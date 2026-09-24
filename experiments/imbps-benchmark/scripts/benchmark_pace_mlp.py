#!/usr/bin/env python3
"""Benchmark AMD PACE's fused IMBPS MLP operator with paired K sweeps.

The timed comparison keeps the operator, tensors, thread count, and placement
fixed and changes only the parameter split K.  K=1 is therefore the direct
PACE unsplit control.  A conventional PyTorch MLP is evaluated outside the
timed region solely to quantify numerical error.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def sysfs_value(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None


def sampled_percentile(
    values: torch.Tensor, q: float, max_samples: int = 1_000_000
) -> tuple[float, int]:
    """Return a deterministic sampled percentile for outputs too large for quantile."""
    flat = values.reshape(-1)
    if flat.numel() > max_samples:
        stride = (flat.numel() + max_samples - 1) // max_samples
        flat = flat[::stride][:max_samples]
    return float(torch.quantile(flat, q).item()), flat.numel()


def invoke_pace(
    x: torch.Tensor,
    up_weights: list[torch.Tensor],
    up_biases: list[torch.Tensor],
    down_weights: list[torch.Tensor],
    down_bias: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Call the documented PACE v1.0 non-gated fused MLP signature."""
    return torch.ops.pace.mlp_mlp_fusion(
        x,
        up_weights,
        up_biases,
        down_weights,
        down_bias,
        activation.capitalize(),
        None,
        None,
    )


def timed_ms(function: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    start = time.perf_counter_ns()
    output = function()
    elapsed = time.perf_counter_ns() - start
    return elapsed / 1_000_000.0, output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--splits", default="1,4,8,16,23")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--data-scale", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [int(value) for value in args.splits.split(",")]
    dimensions = (args.tokens, args.hidden_size, args.intermediate_size)
    if min(*dimensions, args.threads, args.repeats, *splits) <= 0:
        raise ValueError("dimensions, threads, repeats, and K must be positive")
    if args.warmup < 0 or args.data_scale <= 0:
        raise ValueError("warmup must be non-negative and data scale positive")
    if len(splits) != len(set(splits)):
        raise ValueError("duplicate K values are not allowed")

    # Importing PACE registers torch.ops.pace.*.
    import pace  # noqa: F401

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPES[args.dtype]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print("[pace] allocating canonical tensors", flush=True)
    x = (
        torch.rand(dimensions[0], dimensions[1], generator=generator, dtype=dtype)
        * args.data_scale
    )
    up_weight = torch.rand(
        dimensions[2], dimensions[1], generator=generator, dtype=dtype
    ) * args.data_scale
    up_bias = torch.rand(dimensions[2], generator=generator, dtype=dtype) * args.data_scale
    down_weight = torch.rand(
        dimensions[1], dimensions[2], generator=generator, dtype=dtype
    ) * args.data_scale
    down_bias = torch.rand(dimensions[1], generator=generator, dtype=dtype) * args.data_scale

    expected: torch.Tensor | None = None
    if not args.skip_reference:
        print("[pace] computing untimed PyTorch numerical reference", flush=True)
        with torch.inference_mode():
            hidden = F.linear(x, up_weight, up_bias)
            hidden = F.gelu(hidden) if args.activation == "gelu" else F.relu(hidden)
            expected = F.linear(hidden, down_weight, down_bias)
            del hidden
        gc.collect()

    raw_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    packed: dict[
        int,
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], float],
    ] = {}
    correctness: dict[int, dict[str, object]] = {}
    latencies_by_k: dict[int, list[float]] = {split_k: [] for split_k in splits}

    with torch.inference_mode():
        for split_k in splits:
            print(f"[pace] packing K={split_k}", flush=True)
            pack_start = time.perf_counter_ns()
            up_weights = [part.contiguous() for part in up_weight.chunk(split_k, dim=0)]
            up_biases = [part.contiguous() for part in up_bias.chunk(split_k, dim=0)]
            down_weights = [part.contiguous() for part in down_weight.chunk(split_k, dim=1)]
            packing_ms = (time.perf_counter_ns() - pack_start) / 1_000_000.0
            packed[split_k] = (up_weights, up_biases, down_weights, packing_ms)

        def invoke(split_k: int) -> torch.Tensor:
            up_weights, up_biases, down_weights, _ = packed[split_k]
            return invoke_pace(
                x,
                up_weights,
                up_biases,
                down_weights,
                down_bias,
                args.activation,
            )

        print("[pace] warming every K", flush=True)
        for _ in range(args.warmup):
            for split_k in splits:
                actual = invoke(split_k)

        print("[pace] checking numerical error", flush=True)
        for split_k in splits:
            actual = invoke(split_k)
            if expected is None:
                metrics: dict[str, object] = {
                    "max_abs_error": float("nan"),
                    "mean_abs_error": float("nan"),
                    "sampled_p99_abs_error": float("nan"),
                    "p99_sample_size": 0,
                    "fraction_abs_error_ge_0.1": float("nan"),
                    "allclose_rtol_1e-2_atol_1e-2": False,
                }
            else:
                actual_f = actual.float()
                expected_f = expected.float()
                delta = (actual_f - expected_f).abs()
                p99_abs, p99_samples = sampled_percentile(delta, 0.99)
                metrics = {
                    "max_abs_error": float(delta.max().item()),
                    "mean_abs_error": float(delta.mean().item()),
                    "sampled_p99_abs_error": p99_abs,
                    "p99_sample_size": p99_samples,
                    "fraction_abs_error_ge_0.1": float((delta >= 0.1).float().mean().item()),
                    "allclose_rtol_1e-2_atol_1e-2": bool(
                        torch.allclose(actual_f, expected_f, rtol=1e-2, atol=1e-2)
                    ),
                }
                del actual_f, expected_f, delta
            correctness[split_k] = metrics

        # Each repeat is a paired block containing every K exactly once.  The
        # randomized order prevents first-run, thermal, and frequency drift from
        # being systematically assigned to a particular K.
        rng = random.Random(args.seed)
        print("[pace] starting randomized paired measurements", flush=True)
        for repeat in range(args.repeats):
            order = list(splits)
            rng.shuffle(order)
            for order_in_pair, split_k in enumerate(order):
                latency_ms, actual = timed_ms(lambda k=split_k: invoke(k))
                latencies_by_k[split_k].append(latency_ms)
                raw_rows.append(
                    {
                        "pair_index": repeat,
                        "order_in_pair": order_in_pair,
                        "split_k": split_k,
                        "latency_ms": latency_ms,
                        "checksum": float(actual.float().sum().item()),
                    }
                )

    median_by_k: dict[int, float] = {}
    for split_k in splits:
        latencies = latencies_by_k[split_k]
        median_ms = statistics.median(latencies)
        median_by_k[split_k] = median_ms
        _, _, _, packing_ms = packed[split_k]
        summaries.append(
            {
                "split_k": split_k,
                "median_ms": median_ms,
                "mean_ms": statistics.mean(latencies),
                "min_ms": min(latencies),
                "p95_ms": sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)],
                "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "speedup_vs_k1": "",
                "packing_ms": packing_ms,
                **correctness[split_k],
            }
        )

    if 1 not in median_by_k:
        raise ValueError("include K=1 to define the unsplit PACE control")
    baseline = median_by_k[1]
    for row in summaries:
        row["speedup_vs_k1"] = baseline / float(row["median_ms"])

    write_csv(args.output_dir / "raw.csv", raw_rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "command": sys.argv,
        "configuration": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "splits": splits,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "pace_version": getattr(sys.modules.get("pace"), "__version__", None),
        },
        "process": {
            "affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OMP_WAIT_POLICY",
                "GOMP_CPU_AFFINITY",
                "LD_PRELOAD",
            )
        },
        "system": {
            "lscpu": command_output(["lscpu"]),
            "lscpu_cache": command_output(["lscpu", "-C"]),
            "lscpu_topology": command_output(
                ["lscpu", "-e=CPU,NODE,SOCKET,CACHE,ONLINE"]
            ),
            "numactl_hardware": command_output(["numactl", "--hardware"]),
            "numactl_show": command_output(["numactl", "--show"]),
            "cpupower_frequency_info": command_output(["cpupower", "frequency-info"]),
            "smt_active": sysfs_value("/sys/devices/system/cpu/smt/active"),
            "cpu0_scaling_governor": sysfs_value(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            ),
            "cpu_boost": sysfs_value("/sys/devices/system/cpu/cpufreq/boost"),
            "transparent_hugepage_enabled": sysfs_value(
                "/sys/kernel/mm/transparent_hugepage/enabled"
            ),
            "transparent_hugepage_defrag": sysfs_value(
                "/sys/kernel/mm/transparent_hugepage/defrag"
            ),
            "numa_balancing": sysfs_value("/proc/sys/kernel/numa_balancing"),
            "perf_event_paranoid": sysfs_value(
                "/proc/sys/kernel/perf_event_paranoid"
            ),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "pace_git_commit": os.environ.get("PACE_GIT_COMMIT"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
