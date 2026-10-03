#!/usr/bin/env python3
"""Run one warmed-up PACE MLP case under a package-scoped AMD uProf pass."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = PROJECT_ROOT / "src" / "pace_mlp_bench.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uprof-bin", type=Path, required=True)
    parser.add_argument("--access-mode", choices=("perf", "msr"), default="perf")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--package", type=int, default=0)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--backend", choices=("tpp", "imbps"), required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=28672)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--setup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--profiler-start-timeout-seconds", type=float, default=30)
    parser.add_argument("--measurement-timeout-seconds", type=float, default=7200)
    parser.add_argument("--arm-delay-seconds", type=float, default=0.25)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def build_worker_command(
    args: argparse.Namespace, ready: Path, start: Path, output: Path
) -> list[str]:
    return [
        "numactl",
        f"--physcpubind={args.cpu_list}",
        "--membind=0",
        sys.executable,
        str(WORKER_PATH),
        "--backend",
        args.backend,
        "--batch",
        str(args.batch),
        "--sequence",
        str(args.sequence),
        "--hidden",
        str(args.hidden),
        "--intermediate",
        str(args.intermediate),
        "--splits",
        str(args.split),
        "--dtype",
        args.dtype,
        "--activation",
        args.activation,
        "--threads",
        str(args.threads),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--ready-file",
        str(ready),
        "--start-file",
        str(start),
        "--start-timeout",
        str(args.setup_timeout_seconds),
        "--output",
        str(output),
    ]


def build_profiler_command(
    args: argparse.Namespace, target_pid: int, csv_path: Path
) -> list[str]:
    if args.access_mode != "perf":
        raise ValueError("PID attachment is supported only in perf mode")
    return [
        str(args.uprof_bin),
        "-X",
        "-m",
        args.metrics,
        "-A",
        "package",
        "-C",
        "-p",
        str(target_pid),
        "-o",
        str(csv_path),
    ]


def terminate(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_profiler_start(
    process: subprocess.Popen[str], log_path: Path, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"uProf exited with {returncode} before measurement; see {log_path}"
            )
        if log_path.is_file() and "Profiling started." in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"uProf did not report profiling start within {timeout}s; see {log_path}"
    )


def main() -> None:
    args = parse_args()
    if not args.uprof_bin.is_file() or not os.access(args.uprof_bin, os.X_OK):
        raise SystemExit(f"uProf binary is not executable: {args.uprof_bin}")
    if args.backend == "tpp" and args.split != 1:
        raise SystemExit("TPP must use split=1")
    if args.intermediate % args.split != 0:
        raise SystemExit(
            f"intermediate={args.intermediate} is not divisible by split={args.split}"
        )

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    ready = result_dir / "worker.ready.json"
    start = result_dir / "worker.start"
    benchmark_path = result_dir / "benchmark.json"
    csv_path = result_dir / "uprof.csv"
    log_path = result_dir / "uprof.log"
    worker_log_path = result_dir / "worker.log"
    worker = build_worker_command(args, ready, start, benchmark_path)
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "case": {
            "backend": args.backend,
            "split": args.split,
            "batch": args.batch,
            "sequence": args.sequence,
            "rows": args.batch * args.sequence,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "dtype": args.dtype,
            "activation": args.activation,
        },
        "profiler": {
            "binary": str(args.uprof_bin),
            "access_mode": args.access_mode,
            "metrics": args.metrics.split(","),
            "scope": "PID-attached core metrics with all-package uncore reporting",
            "primary_package": args.package,
            "background_control_package": 1 if args.package == 0 else 0,
            "aggregation": "package",
            "cumulative": True,
            "attach_mode": "pid_after_warmup",
            "arm_delay_seconds": args.arm_delay_seconds,
        },
        "measurement": {
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "warmups": args.warmups,
            "iterations": args.iterations,
        },
        "environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "process_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with worker_log_path.open("w", encoding="utf-8") as worker_log:
        worker_process = subprocess.Popen(
            worker,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        setup_deadline = time.monotonic() + args.setup_timeout_seconds
        while not ready.exists():
            returncode = worker_process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"worker exited with {returncode} before readiness; see {worker_log_path}"
                )
            if time.monotonic() >= setup_deadline:
                terminate(worker_process)
                raise TimeoutError(
                    f"worker did not become ready; see {worker_log_path}"
                )
            time.sleep(0.1)

        ready_payload = json.loads(ready.read_text(encoding="utf-8"))
        target_pid = int(ready_payload["pid"])
        command = build_profiler_command(args, target_pid, csv_path)
        manifest["profiler"]["target_pid"] = target_pid
        manifest["profiler"]["command"] = command
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        with log_path.open("w", encoding="utf-8") as log:
            profiler = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_profiler_start(
                    profiler, log_path, args.profiler_start_timeout_seconds
                )
            except (RuntimeError, TimeoutError):
                terminate(worker_process)
                if profiler.poll() is None:
                    terminate(profiler)
                raise
            time.sleep(args.arm_delay_seconds)
            start.write_text("start\n", encoding="utf-8")
            try:
                worker_returncode = worker_process.wait(
                    timeout=args.measurement_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                terminate(worker_process)
                terminate(profiler)
                raise TimeoutError(f"worker measurement timed out; see {worker_log_path}")

            if worker_returncode != 0:
                terminate(profiler)
                raise RuntimeError(
                    f"worker exited with {worker_returncode}; see {worker_log_path}"
                )
            try:
                profiler_returncode = profiler.wait(timeout=60)
            except subprocess.TimeoutExpired:
                terminate(profiler)
                raise TimeoutError(f"uProf did not stop after target exit; see {log_path}")

    if profiler_returncode != 0:
        raise RuntimeError(f"uProf exited with {profiler_returncode}; see {log_path}")
    if not benchmark_path.is_file():
        raise RuntimeError(f"worker result is missing: {benchmark_path}")
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"uProf CSV is missing or empty: {csv_path}")

    manifest["status"] = "complete"
    manifest["profiler"]["returncode"] = profiler_returncode
    manifest["outputs"] = {
        "benchmark": str(benchmark_path),
        "uprof_csv": str(csv_path),
        "uprof_log": str(log_path),
        "worker_log": str(worker_log_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "median_ms": benchmark["run"]["median_ms"],
                "measurement_wall_seconds": benchmark["run"]["measurement_wall_seconds"],
                "uprof_csv_bytes": csv_path.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
