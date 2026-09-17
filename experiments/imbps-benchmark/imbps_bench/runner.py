"""Benchmark orchestration."""

import gc
import json
import math
import os
import random
import socket
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Sequence, Tuple

import torch

from imbps_bench.kernels import (
    imbps_forward_into,
    make_input,
    make_reference_workspace,
    make_split_workspace,
    make_weights,
    pack_weights,
    reference_forward_into,
    tensor_nbytes,
    theoretical_flops,
    weights_nbytes,
)
from imbps_bench.metadata import collect_metadata
from imbps_bench.reporting import append_raw_rows, summarize_rows, write_raw_header, write_summary


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class RunConfig:
    mlp_kind: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    dtype_name: str
    splits: Sequence[int]
    threads: int
    warmup: int
    repeats: int
    seed: int
    weight_layout: str
    gelu_approximate: str
    output_dir: Path
    max_allocation_gib: float
    skip_correctness: bool
    allow_correctness_failure: bool


def tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float32:
        return 2e-4, 2e-5
    if dtype == torch.bfloat16:
        return 8e-2, 8e-2
    raise ValueError("unsupported dtype: %s" % dtype)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = (actual_f - expected_f).abs()
    max_abs = float(difference.max().item())
    denominator = expected_f.abs().clamp_min(torch.finfo(torch.float32).eps)
    max_rel = float((difference / denominator).max().item())
    return max_abs, max_rel


def estimate_peak_bytes(config: RunConfig) -> int:
    dtype = DTYPES[config.dtype_name]
    itemsize = torch.empty((), dtype=dtype).element_size()
    weight_matrices = 2 if config.mlp_kind == "opt" else 3
    canonical_weights = weight_matrices * config.hidden_size * config.intermediate_size * itemsize
    packed_weights = canonical_weights if config.weight_layout == "prepacked" else 0
    input_bytes = config.tokens * config.hidden_size * itemsize
    output_bytes = input_bytes
    reference_activation_count = 1 if config.mlp_kind == "opt" else 2
    reference_workspace = (
        reference_activation_count * config.tokens * config.intermediate_size * itemsize
        + output_bytes
    )
    smallest_k = min(config.splits)
    max_width = int(math.ceil(config.intermediate_size / smallest_k))
    split_activation_count = 1 if config.mlp_kind == "opt" else 2
    split_workspace = split_activation_count * config.tokens * max_width * itemsize + output_bytes
    # Include an expected-output clone used for correctness and a small safety margin.
    return int(
        (canonical_weights + packed_weights + input_bytes + reference_workspace + split_workspace + output_bytes)
        * 1.10
    )


def _set_threads(threads: int) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits this before inter-op work starts. The captured
        # metadata records the effective value if another library initialized it.
        pass


def _run_id(config: RunConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "%s-%s-m%d-h%d-i%d-t%d-%s" % (
        stamp,
        config.mlp_kind,
        config.tokens,
        config.hidden_size,
        config.intermediate_size,
        config.threads,
        config.dtype_name,
    )


def _base_row(
    config: RunConfig,
    run_id: str,
    split_k: int,
    variant: str,
    iteration: int,
    pair_index: int,
    order_in_pair: int,
    elapsed_ns: int,
    checksum: float,
    packing_ms: float,
    canonical_weight_bytes: int,
    packed_weight_bytes: int,
    workspace_bytes: int,
    input_bytes: int,
    output_bytes: int,
    max_abs_error: float,
    max_rel_error: float,
    allclose: bool,
) -> Dict[str, Any]:
    flops = theoretical_flops(
        config.mlp_kind, config.tokens, config.hidden_size, config.intermediate_size
    )
    latency_seconds = elapsed_ns / 1_000_000_000.0
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pair_index": pair_index,
        "order_in_pair": order_in_pair,
        "variant": variant,
        "mlp_kind": config.mlp_kind,
        "weight_layout": "canonical" if variant == "reference" else config.weight_layout,
        "dtype": config.dtype_name,
        "threads": torch.get_num_threads(),
        "tokens": config.tokens,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "split_k": split_k,
        "iteration": iteration,
        "latency_ns": elapsed_ns,
        "latency_ms": elapsed_ns / 1_000_000.0,
        "tokens_per_second": config.tokens / latency_seconds,
        "gflops": flops / elapsed_ns,
        "checksum": checksum,
        "packing_ms": packing_ms,
        "canonical_weight_bytes": canonical_weight_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "workspace_bytes": workspace_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "allclose": allclose,
        "seed": config.seed,
        "gelu_approximate": config.gelu_approximate,
    }


def run_sweep(config: RunConfig) -> Dict[str, Any]:
    if config.dtype_name not in DTYPES:
        raise ValueError("unsupported dtype: %s" % config.dtype_name)
    if not config.splits:
        raise ValueError("at least one split value is required")
    if any(k <= 0 or k > config.intermediate_size for k in config.splits):
        raise ValueError("every split must be in [1, intermediate_size]")
    if config.warmup < 0 or config.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    predicted_bytes = estimate_peak_bytes(config)
    limit_bytes = int(config.max_allocation_gib * (1024 ** 3))
    if predicted_bytes > limit_bytes:
        raise MemoryError(
            "estimated peak allocation %.2f GiB exceeds --max-allocation-gib %.2f"
            % (predicted_bytes / (1024 ** 3), config.max_allocation_gib)
        )

    _set_threads(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(config)
    raw_path = config.output_dir / "raw.csv"
    summary_path = config.output_dir / "summary.csv"
    metadata_path = config.output_dir / "metadata.json"
    manifest_path = config.output_dir / "manifest.json"
    write_raw_header(raw_path)
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["splits"] = list(config.splits)
    config_dict["estimated_peak_bytes"] = predicted_bytes
    collect_metadata(metadata_path, arguments=config_dict)

    dtype = DTYPES[config.dtype_name]
    print(
        "[imbps] allocating tensors (estimated peak %.2f GiB)" % (predicted_bytes / (1024 ** 3)),
        file=sys.stderr,
        flush=True,
    )
    weights = make_weights(
        config.mlp_kind,
        config.hidden_size,
        config.intermediate_size,
        dtype,
        config.seed,
    )
    x = make_input(config.tokens, config.hidden_size, dtype, config.seed + 1)
    reference_workspace = make_reference_workspace(
        config.mlp_kind,
        config.tokens,
        config.hidden_size,
        config.intermediate_size,
        dtype,
    )
    canonical_weight_bytes = weights_nbytes(weights)
    input_bytes = tensor_nbytes(x)
    output_bytes = config.tokens * config.hidden_size * x.element_size()
    rtol, atol = tolerances(dtype)
    rng = random.Random(config.seed)
    all_rows: List[Dict[str, Any]] = []
    pair_index = 0

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "config": config_dict,
        "paths": {
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
            "metadata_json": str(metadata_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with torch.inference_mode():
        for split_k in config.splits:
            print(
                "[imbps] K=%d: packing, checking, warming up, and timing" % split_k,
                file=sys.stderr,
                flush=True,
            )
            packed = pack_weights(weights, split_k, config.weight_layout)
            split_workspace = make_split_workspace(
                config.mlp_kind,
                config.tokens,
                config.hidden_size,
                packed.max_width,
                dtype,
            )

            max_abs_error = math.nan
            max_rel_error = math.nan
            is_allclose = True
            if not config.skip_correctness:
                expected = reference_forward_into(
                    x, weights, reference_workspace, config.gelu_approximate
                ).clone()
                actual = imbps_forward_into(
                    x, packed, split_workspace, config.gelu_approximate
                )
                max_abs_error, max_rel_error = error_metrics(actual, expected)
                is_allclose = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
                del expected
                if not is_allclose and not config.allow_correctness_failure:
                    raise RuntimeError(
                        "correctness failed for K=%d: max_abs=%g max_rel=%g (rtol=%g atol=%g)"
                        % (split_k, max_abs_error, max_rel_error, rtol, atol)
                    )

            def run_reference() -> torch.Tensor:
                return reference_forward_into(
                    x, weights, reference_workspace, config.gelu_approximate
                )

            def run_imbps() -> torch.Tensor:
                return imbps_forward_into(
                    x, packed, split_workspace, config.gelu_approximate
                )

            for _ in range(config.warmup):
                run_reference()
                run_imbps()

            split_rows: List[Dict[str, Any]] = []
            for iteration in range(config.repeats):
                order = ["reference", "imbps"]
                rng.shuffle(order)
                for order_in_pair, variant in enumerate(order):
                    function = run_reference if variant == "reference" else run_imbps
                    started = perf_counter_ns()
                    output = function()
                    elapsed_ns = perf_counter_ns() - started
                    checksum = float(output[0, 0].float().item())
                    row = _base_row(
                        config=config,
                        run_id=run_id,
                        split_k=split_k,
                        variant=variant,
                        iteration=iteration,
                        pair_index=pair_index,
                        order_in_pair=order_in_pair,
                        elapsed_ns=elapsed_ns,
                        checksum=checksum,
                        packing_ms=packed.packing_ms if variant == "imbps" else 0.0,
                        canonical_weight_bytes=canonical_weight_bytes,
                        packed_weight_bytes=packed.packed_bytes if variant == "imbps" else 0,
                        workspace_bytes=(
                            split_workspace.nbytes
                            if variant == "imbps"
                            else reference_workspace.nbytes
                        ),
                        input_bytes=input_bytes,
                        output_bytes=output_bytes,
                        max_abs_error=max_abs_error,
                        max_rel_error=max_rel_error,
                        allclose=is_allclose,
                    )
                    split_rows.append(row)
                pair_index += 1
            append_raw_rows(raw_path, split_rows)
            all_rows.extend(split_rows)
            reference_times = [
                float(row["latency_ms"])
                for row in split_rows
                if row["variant"] == "reference"
            ]
            imbps_times = [
                float(row["latency_ms"])
                for row in split_rows
                if row["variant"] == "imbps"
            ]
            reference_median = statistics.median(reference_times)
            imbps_median = statistics.median(imbps_times)
            print(
                "[imbps] K=%d: reference %.3f ms, IMBPS %.3f ms, speedup %.3fx"
                % (split_k, reference_median, imbps_median, reference_median / imbps_median),
                file=sys.stderr,
                flush=True,
            )
            del split_workspace
            del packed
            gc.collect()

    summaries = summarize_rows(all_rows)
    write_summary(summary_path, summaries)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["row_count"] = len(all_rows)
    manifest["summary_count"] = len(summaries)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    candidates = [row for row in summaries if row["variant"] == "imbps"]
    best = max(candidates, key=lambda row: float(row["speedup_vs_reference"])) if candidates else None
    return {
        "run_id": run_id,
        "output_dir": str(config.output_dir),
        "raw_csv": str(raw_path),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
        "manifest_json": str(manifest_path),
        "best_imbps": best,
    }
