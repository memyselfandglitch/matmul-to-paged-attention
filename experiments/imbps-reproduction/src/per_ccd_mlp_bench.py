#!/usr/bin/env python3
"""Prototype parallel IMBPS by assigning parameter blocks to CCD workers.

AMD-PACE v1.0 marks its split loop as a future parallelization point.  This
runner leaves the installed PACE package untouched and evaluates that idea with
persistent processes: one process and OpenMP team per CCD, disjoint blocks of
the intermediate dimension, shared input, and an explicit cross-CCD reduction.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import queue
import random
import resource
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .cache_model import MIB
    from .pace_mlp_bench import (
        allocator_mappings,
        collect_thread_affinity,
        percentile,
        summarize_thread_affinity,
    )
    from .select_llc_cpus import discover_groups_with_source
except ImportError:  # Direct execution: python3 src/per_ccd_mlp_bench.py
    from cache_model import MIB
    from pace_mlp_bench import (
        allocator_mappings,
        collect_thread_affinity,
        percentile,
        summarize_thread_affinity,
    )
    from select_llc_cpus import discover_groups_with_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
EXPECTED_PACE_VERSION = "1.0.0"


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="opt30b_parallel_imbps")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--rows", type=comma_ints, default=None)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20251001)
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0 or args.warmups < 0 or args.barrier_timeout <= 0:
        parser.error("rounds/timeout must be positive and warmups nonnegative")
    return args


def load_claim(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    try:
        return registry, registry["per_ccd"][name]
    except KeyError as error:
        choices = ", ".join(sorted(registry.get("per_ccd", {})))
        raise ValueError(f"unknown per-CCD claim {name!r}; choose {choices}") from error


def select_variant(claim: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [row for row in claim["variants"] if row["variant_id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"variant {identifier!r} is not uniquely preregistered")
    return dict(matches[0])


def chunk_ranges(total: int, chunks: int) -> list[tuple[int, int]]:
    """Match torch.chunk's ceil-sized contiguous blocks."""
    if min(total, chunks) <= 0:
        raise ValueError("total and chunks must be positive")
    width = (total + chunks - 1) // chunks
    return [
        (start, min(total, start + width))
        for start in range(0, total, width)
    ]


def assign_blocks(total_splits: int, workers: int) -> list[list[int]]:
    ranges = chunk_ranges(total_splits, workers)
    return [list(range(start, end)) for start, end in ranges]


def per_ccd_working_set(
    rows: int,
    hidden: int,
    intermediate: int,
    total_splits: int,
    bytes_per_element: int = 2,
) -> dict[str, float | bool]:
    ranges = chunk_ranges(intermediate, total_splits)
    widest = max(end - start for start, end in ranges)
    input_bytes = bytes_per_element * rows * hidden
    intermediate_bytes = bytes_per_element * rows * widest
    one_weight_block_bytes = bytes_per_element * hidden * widest
    total = input_bytes + intermediate_bytes + one_weight_block_bytes
    return {
        "widest_intermediate_block": widest,
        "input_mib": input_bytes / MIB,
        "intermediate_block_mib": intermediate_bytes / MIB,
        "one_weight_block_mib": one_weight_block_bytes / MIB,
        "modeled_working_set_mib": total / MIB,
    }


def numeric_comparison(reference: Any, candidate: Any) -> dict[str, float]:
    import torch

    difference = (reference.float() - candidate.float()).abs().reshape(-1)
    return {
        "max_abs": float(difference.max().item()),
        "q99_abs": float(torch.quantile(difference, 0.99).item()),
        "mean_abs": float(difference.mean().item()),
        "output_coordinate_argmax_agreement": float(
            (reference.float().argmax(dim=-1) == candidate.float().argmax(dim=-1))
            .float()
            .mean()
            .item()
        ),
    }


def direct_imbps(torch: Any, source: Any, weight1: Any, weight2: Any, splits: int):
    ranges = chunk_ranges(weight1.shape[0], splits)
    weights1 = [weight1[start:end].contiguous() for start, end in ranges]
    weights2 = [weight2[:, start:end].contiguous() for start, end in ranges]
    output = torch.ops.pace.mlp_mlp_fusion(
        source, weights1, None, weights2, None, "relu", None, None
    )
    return output


def worker_main(
    worker_index: int,
    cpus: tuple[int, ...],
    block_ids: list[int],
    ranges: list[tuple[int, int]],
    source: Any,
    weight1: Any,
    weight2: Any,
    partials: Any,
    start_barrier: Any,
    done_barrier: Any,
    stop_event: Any,
    ready_queue: Any,
    error_queue: Any,
) -> None:
    try:
        os.sched_setaffinity(0, set(cpus))
        os.environ["OMP_NUM_THREADS"] = str(len(cpus))
        os.environ["OMP_PROC_BIND"] = "close"
        os.environ["OMP_PLACES"] = "cores"
        os.environ["OMP_DYNAMIC"] = "false"
        os.environ["NUMEXPR_MAX_THREADS"] = str(len(cpus))
        os.environ["NUMEXPR_NUM_THREADS"] = str(len(cpus))

        import torch
        import pace  # noqa: F401

        torch.set_num_threads(len(cpus))
        torch.set_num_interop_threads(1)
        local_weight1 = [
            weight1[ranges[index][0] : ranges[index][1]].contiguous()
            for index in block_ids
        ]
        local_weight2 = [
            weight2[:, ranges[index][0] : ranges[index][1]].contiguous()
            for index in block_ids
        ]

        with torch.inference_mode():
            output = torch.ops.pace.mlp_mlp_fusion(
                source,
                local_weight1,
                None,
                local_weight2,
                None,
                "relu",
                None,
                None,
            )
            partials[worker_index].copy_(output)

        masks, affinity_union = collect_thread_affinity()
        if set(affinity_union) != set(cpus):
            raise RuntimeError(
                f"worker {worker_index} did not cover exactly CCD CPUs "
                f"{list(cpus)}: {affinity_union}"
            )
        mapped_allocators = allocator_mappings()
        if os.environ.get("REQUIRE_TCMALLOC") == "1" and not any(
            "tcmalloc" in path.lower() for path in mapped_allocators
        ):
            raise RuntimeError(f"worker {worker_index} did not map tcmalloc")
        ready_queue.put(
            {
                "worker": worker_index,
                "pid": os.getpid(),
                "cpus": list(cpus),
                "block_ids": block_ids,
                "torch_threads": torch.get_num_threads(),
                "affinity": affinity_union,
                "thread_affinity_masks": summarize_thread_affinity(masks),
                "allocator_mappings": mapped_allocators,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        while True:
            start_barrier.wait()
            if stop_event.is_set():
                return
            with torch.inference_mode():
                output = torch.ops.pace.mlp_mlp_fusion(
                    source,
                    local_weight1,
                    None,
                    local_weight2,
                    None,
                    "relu",
                    None,
                    None,
                )
                partials[worker_index].copy_(output)
            done_barrier.wait()
    except BaseException:
        error_queue.put(
            {"worker": worker_index, "traceback": traceback.format_exc()}
        )
        try:
            start_barrier.abort()
            done_barrier.abort()
        except BaseException:
            pass


def check_worker_errors(error_queue: Any) -> None:
    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except queue.Empty:
            break
    if errors:
        raise RuntimeError(f"per-CCD worker failure: {errors}")


def run_per_ccd(
    torch: Any,
    source: Any,
    weight1: Any,
    weight2: Any,
    ccd_groups: list[tuple[int, ...]],
    total_splits: int,
    warmups: int,
    rounds: int,
    timeout: float,
) -> tuple[list[float], Any, list[dict[str, Any]]]:
    worker_count = len(ccd_groups)
    ranges = chunk_ranges(weight1.shape[0], total_splits)
    assignments = assign_blocks(total_splits, worker_count)
    if len(ranges) != total_splits:
        raise ValueError("intermediate dimension produced fewer chunks than requested")

    # Spawn transfers these storages once and every worker maps the same pages.
    torch.multiprocessing.set_sharing_strategy("file_system")
    source.share_memory_()
    weight1.share_memory_()
    weight2.share_memory_()
    partials = torch.empty(
        (worker_count, source.shape[0], source.shape[1]), dtype=source.dtype
    ).share_memory_()
    reduction = torch.empty_like(source, dtype=torch.float32)
    final_output = torch.empty_like(source)

    context = mp.get_context("spawn")
    start_barrier = context.Barrier(worker_count + 1, timeout=timeout)
    done_barrier = context.Barrier(worker_count + 1, timeout=timeout)
    stop_event = context.Event()
    ready_queue = context.Queue()
    error_queue = context.Queue()
    processes = []
    for index, cpus in enumerate(ccd_groups):
        process = context.Process(
            target=worker_main,
            args=(
                index,
                cpus,
                assignments[index],
                ranges,
                source,
                weight1,
                weight2,
                partials,
                start_barrier,
                done_barrier,
                stop_event,
                ready_queue,
                error_queue,
            ),
        )
        process.start()
        processes.append(process)

    worker_reports = []
    try:
        deadline = time.monotonic() + timeout
        while len(worker_reports) < worker_count:
            check_worker_errors(error_queue)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("per-CCD workers did not become ready")
            try:
                worker_reports.append(ready_queue.get(timeout=min(1.0, remaining)))
            except queue.Empty:
                continue

        def execute_once() -> float:
            started = time.perf_counter_ns()
            start_barrier.wait()
            done_barrier.wait()
            reduction.zero_()
            for partial in partials:
                reduction.add_(partial)
            final_output.copy_(reduction)
            return (time.perf_counter_ns() - started) / 1_000_000

        for _ in range(warmups):
            execute_once()
        durations = [execute_once() for _ in range(rounds)]
        check_worker_errors(error_queue)
        return durations, final_output.clone(), sorted(
            worker_reports, key=lambda report: report["worker"]
        )
    finally:
        stop_event.set()
        try:
            start_barrier.wait(timeout=5)
        except BaseException:
            pass
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        check_worker_errors(error_queue)


def build_model(
    torch: Any,
    mode: str,
    hidden: int,
    intermediate: int,
    total_splits: int,
    weight1: Any,
    weight2: Any,
):
    from pace.ops.base import BackendBase
    from pace.ops.enum import BackendType
    from pace.ops.mlp import MergedMLP

    backend = BackendType.TPP if mode == "tpp" else BackendType.IMBPS
    os.environ["IMBPS_BLOCK_SIZE"] = str(total_splits)
    model = MergedMLP(
        in_features=hidden,
        out_features=intermediate,
        bias=False,
        activation="relu",
        gate=False,
        dtype=torch.bfloat16,
        backend_impl=backend,
    )
    model.eval()
    with torch.no_grad():
        model.up_proj.linear.weight.copy_(weight1)
        model.down_proj.weight.copy_(weight2)
    for module in model.modules():
        module_backend = getattr(module, "backend", None)
        if isinstance(module_backend, BackendBase):
            module_backend.preprocess(module)
    return model


def main() -> None:
    args = parse_args()
    registry, claim = load_claim(args.claim)
    variant = select_variant(claim, args.variant)
    model_record = registry["models"][claim["model"]]
    rows = args.rows or [int(value) for value in claim["rows"]]
    hidden = int(model_record["hidden"])
    intermediate = int(model_record["intermediate"])
    total_splits = int(variant["total_splits"])
    ccd_count = int(variant["ccd_count"])

    # Capture and validate the launch mask before importing PyTorch.  With
    # OMP_PROC_BIND=close, OpenMP may pin the main thread to one CPU during the
    # import/runtime initialization even though its worker team spans the
    # complete 96-CPU socket.
    launch_affinity = set(os.sched_getaffinity(0))
    discovered, topology_source, topology_candidates = discover_groups_with_source(
        launch_affinity,
        expected_groups=int(claim["expected_ccds"]),
        expected_group_size=int(claim["cores_per_ccd"]),
    )
    selected_groups = discovered[:ccd_count]

    import torch
    import pace  # noqa: F401

    threads = args.threads or len(launch_affinity)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    pace_version = importlib.metadata.version("pace")
    if pace_version != EXPECTED_PACE_VERSION:
        raise RuntimeError(
            f"expected PACE {EXPECTED_PACE_VERSION}, found {pace_version}"
        )
    mapped_allocators = allocator_mappings()
    if os.environ.get("REQUIRE_TCMALLOC") == "1" and not any(
        "tcmalloc" in path.lower() for path in mapped_allocators
    ):
        raise RuntimeError("REQUIRE_TCMALLOC=1 but tcmalloc is not mapped")

    weight_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    weight1 = torch.empty((intermediate, hidden), dtype=torch.bfloat16).uniform_(
        -0.02, 0.02, generator=weight_generator
    )
    weight2 = torch.empty((hidden, intermediate), dtype=torch.bfloat16).uniform_(
        -0.02, 0.02, generator=weight_generator
    )
    model = None
    if variant["mode"] in {"tpp", "global_imbps"}:
        model = build_model(
            torch,
            variant["mode"],
            hidden,
            intermediate,
            total_splits,
            weight1,
            weight2,
        )

    records = []
    rng = random.Random(args.seed)
    row_order = list(rows)
    rng.shuffle(row_order)
    for row_index, active_rows in enumerate(row_order, start=1):
        source_generator = torch.Generator(device="cpu").manual_seed(
            args.seed + active_rows * 101
        )
        source = torch.empty(
            (active_rows, hidden), dtype=torch.bfloat16
        ).uniform_(-0.02, 0.02, generator=source_generator)
        worker_reports: list[dict[str, Any]] = []
        numerics = None
        if variant["mode"] == "per_ccd":
            with torch.inference_mode():
                reference = direct_imbps(
                    torch, source, weight1, weight2, total_splits
                )
            durations, output, worker_reports = run_per_ccd(
                torch,
                source,
                weight1,
                weight2,
                selected_groups,
                total_splits,
                args.warmups,
                args.rounds,
                args.barrier_timeout,
            )
            numerics = numeric_comparison(reference, output)
        else:
            assert model is not None
            with torch.inference_mode():
                for _ in range(args.warmups):
                    output = model(source)
                durations = []
                for _ in range(args.rounds):
                    started = time.perf_counter_ns()
                    output = model(source)
                    durations.append((time.perf_counter_ns() - started) / 1_000_000)

        cache_model = per_ccd_working_set(
            active_rows, hidden, intermediate, total_splits
        )
        cache_model["ccd_cache_mib"] = float(claim["ccd_cache_mib"])
        cache_model["fits_one_ccd_strict"] = (
            cache_model["modeled_working_set_mib"] < claim["ccd_cache_mib"]
        )
        records.append(
            {
                "active_rows": active_rows,
                "row_execution_order": row_index,
                "durations_ms": durations,
                "median_ms": statistics.median(durations),
                "q1_ms": percentile(durations, 0.25),
                "q3_ms": percentile(durations, 0.75),
                "output": {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "sum_fp32": float(output.float().sum().item()),
                    "max_abs_fp32": float(output.float().abs().max().item()),
                },
                "numerics_vs_global_same_k": numerics,
                "cache_model_per_ccd": cache_model,
                "workers": worker_reports,
            }
        )
        print(
            f"variant={args.variant} mode={variant['mode']} rows={active_rows} "
            f"P={ccd_count} K={total_splits} median_ms={statistics.median(durations):.3f}",
            flush=True,
        )

    masks, affinity_union = collect_thread_affinity()
    result = {
        "schema_version": 1,
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": "parallel_per_ccd_imbps",
        "scope_note": (
            "Standalone OPT-30B MLP. Per-CCD latency includes persistent-worker "
            "synchronization, PACE fused block computation, publication of partial "
            "outputs, and cross-CCD reduction; process/model setup is excluded."
        ),
        "claim": args.claim,
        "variant": variant,
        "expected_variants": claim["variants"],
        "model": {
            "name": claim["model"],
            "hidden": hidden,
            "intermediate": intermediate,
            "dtype": claim["dtype"],
            "activation": claim["activation"],
            "bias": claim["bias"],
        },
        "runner": {
            "rows": rows,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "seed": args.seed,
            "barrier_timeout": args.barrier_timeout,
        },
        "topology": {
            "launch_affinity": sorted(launch_affinity),
            "selected_topology_source": topology_source,
            "topology_candidates": topology_candidates,
            "discovered_ccd_groups": [list(group) for group in discovered],
            "selected_ccd_groups": [list(group) for group in selected_groups],
            "ccd_count": ccd_count,
            "cores_per_ccd": claim["cores_per_ccd"],
            "ccd_cache_mib": claim["ccd_cache_mib"],
        },
        "records": sorted(records, key=lambda record: record["active_rows"]),
        "environment": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "torch_version": torch.__version__,
            "pace_version": pace_version,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "affinity": affinity_union,
            "main_thread_affinity": sorted(os.sched_getaffinity(0)),
            "process_thread_count": len(masks),
            "thread_affinity_masks": summarize_thread_affinity(masks),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "libxsmm_block_size": os.environ.get("LIBXSMM_BLOCK_SIZE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "ld_preload": os.environ.get("LD_PRELOAD"),
            "allocator_mappings": mapped_allocators,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
