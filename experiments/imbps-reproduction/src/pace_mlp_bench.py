#!/usr/bin/env python3
"""Run one isolated PACE TPP or IMBPS OPT-style MLP timing case."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tpp", "imbps"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--activation", choices=("gelu", "relu"), default="relu")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="after warmup, write PID/TIDs here and wait for --start-file",
    )
    parser.add_argument("--start-file", type=Path, default=None)
    parser.add_argument("--start-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if min(args.batch, args.sequence, args.hidden, args.intermediate, args.splits) <= 0:
        parser.error("dimensions and splits must be positive")
    if args.intermediate % args.splits != 0:
        parser.error("intermediate must be divisible by splits")
    if args.warmups < 0 or args.iterations <= 0:
        parser.error("warmups must be nonnegative and iterations positive")
    if (args.ready_file is None) != (args.start_file is None):
        parser.error("--ready-file and --start-file must be supplied together")
    if args.backend == "tpp" and args.dtype != "bf16":
        parser.error("PACE v1.0 TPP linear kernels are registered for BF16 only")
    return args


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wait_for_profiler(ready_file: Path, start_file: Path, timeout: float) -> None:
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    tids = sorted(int(path.name) for path in Path("/proc/self/task").iterdir())
    ready_file.write_text(
        json.dumps({"pid": os.getpid(), "tids": tids}) + "\n", encoding="utf-8"
    )
    deadline = time.monotonic() + timeout
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"profiler did not create {start_file} within {timeout}s")
        time.sleep(0.01)


def main() -> None:
    args = parse_args()
    os.environ["IMBPS_BLOCK_SIZE"] = str(args.splits)

    import torch
    from pace.ops.base import BackendBase
    from pace.ops.enum import BackendType
    from pace.ops.mlp import MergedMLP

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    affinity = set(os.sched_getaffinity(0))
    if len(affinity) < torch.get_num_threads():
        raise RuntimeError(
            "OpenMP oversubscription: "
            f"torch has {torch.get_num_threads()} threads but process affinity "
            f"contains only {len(affinity)} CPUs ({sorted(affinity)})"
        )
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    backend = BackendType.IMBPS if args.backend == "imbps" else BackendType.TPP
    pace_version = importlib.metadata.version("pace")
    if pace_version != "1.0.0":
        raise RuntimeError(f"expected PACE 1.0.0, found {pace_version}")

    setup_started = time.perf_counter()
    model = MergedMLP(
        in_features=args.hidden,
        out_features=args.intermediate,
        bias=not args.no_bias,
        activation=args.activation,
        gate=False,
        dtype=dtype,
        backend_impl=backend,
    )
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)
        source = torch.empty(
            args.batch * args.sequence, args.hidden, dtype=dtype
        ).uniform_(-0.02, 0.02)

    for module in model.modules():
        module_backend = getattr(module, "backend", None)
        if isinstance(module_backend, BackendBase):
            module_backend.preprocess(module)

    with torch.inference_mode():
        output = None
        for _ in range(args.warmups):
            output = model(source)

        setup_seconds = time.perf_counter() - setup_started

        if args.ready_file is not None and args.start_file is not None:
            wait_for_profiler(args.ready_file, args.start_file, args.start_timeout)

        measurement_started = time.perf_counter()
        durations_ms: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            output = model(source)
            elapsed = time.perf_counter_ns() - started
            durations_ms.append(elapsed / 1_000_000)
        measurement_wall_seconds = time.perf_counter() - measurement_started

    assert output is not None
    median_ms = statistics.median(durations_ms)
    mean_ms = statistics.fmean(durations_ms)
    stdev_ms = statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    q1_ms = percentile(durations_ms, 0.25)
    q3_ms = percentile(durations_ms, 0.75)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": {
            "backend": args.backend,
            "batch": args.batch,
            "sequence": args.sequence,
            "rows": args.batch * args.sequence,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "splits": args.splits,
            "dtype": args.dtype,
            "activation": args.activation,
            "bias": not args.no_bias,
            "seed": args.seed,
        },
        "run": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "durations_ms": durations_ms,
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "stdev_ms": stdev_ms,
            "q1_ms": q1_ms,
            "q3_ms": q3_ms,
            "iqr_ms": q3_ms - q1_ms,
            "coefficient_of_variation": stdev_ms / mean_ms if mean_ms else None,
            "setup_seconds": setup_seconds,
            "measurement_wall_seconds": measurement_wall_seconds,
        },
        "output": {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "sum_fp32": float(output.float().sum().item()),
            "max_abs_fp32": float(output.float().abs().max().item()),
        },
        "environment": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "affinity": sorted(affinity),
            "torch_version": torch.__version__,
            "pace_version": pace_version,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "gomp_cpu_affinity": os.environ.get("GOMP_CPU_AFFINITY"),
            "imbps_block_size": os.environ.get("IMBPS_BLOCK_SIZE"),
            "libxsmm_block_size": os.environ.get("LIBXSMM_BLOCK_SIZE"),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "median_ms": median_ms}))


if __name__ == "__main__":
    main()
