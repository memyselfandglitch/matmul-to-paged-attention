#!/usr/bin/env python3
"""Profile one pre-warmed PACE MLP with package-scoped AMD uProf counters."""

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
GATE_PATH = PROJECT_ROOT / "src" / "uprof_measurement_gate.py"


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
    parser.add_argument("--setup-timeout-seconds", type=float, default=600)
    parser.add_argument("--profiler-start-timeout-seconds", type=float, default=30)
    parser.add_argument("--measurement-timeout-seconds", type=float, default=7200)
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
        str(args.measurement_timeout_seconds),
        "--output",
        str(output),
    ]


def build_gate_command(
    start: Path, done: Path, timing: Path, timeout_seconds: float
) -> list[str]:
    return [
        sys.executable,
        str(GATE_PATH),
        "--start-file",
        str(start),
        "--done-file",
        str(done),
        "--timing-file",
        str(timing),
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def build_profiler_command(
    args: argparse.Namespace, gate: list[str], csv_path: Path
) -> list[str]:
    access_args = ["-X"] if args.access_mode == "perf" else ["--msr"]
    return [
        str(args.uprof_bin),
        *access_args,
        "-m",
        args.metrics,
        "-c",
        f"package={args.package}",
        "-C",
        "-o",
        str(csv_path),
        "--",
        *gate,
    ]


def terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_for_file(
    path: Path,
    process: subprocess.Popen[str],
    timeout: float,
    description: str,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"{description} process exited with {returncode}; see {log_path}"
            )
        if time.monotonic() >= deadline:
            terminate(process)
            raise TimeoutError(f"timed out waiting for {path}; see {log_path}")
        time.sleep(0.05)


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
    uprof_log_path = result_dir / "uprof.log"
    worker_log_path = result_dir / "worker.log"
    gate_timing_path = result_dir / "gate-timing.json"
    worker_command = build_worker_command(args, ready, start, benchmark_path)
    gate_command = build_gate_command(
        start, benchmark_path, gate_timing_path, args.measurement_timeout_seconds
    )
    profiler_command = build_profiler_command(args, gate_command, csv_path)
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
            "scope": f"system counters restricted to package={args.package}",
            "aggregation": "native component rows",
            "launch_mode": "profiled gate controlling an external pre-warmed worker",
            "command": profiler_command,
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
        worker = subprocess.Popen(
            worker_command,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_file(
                ready,
                worker,
                args.setup_timeout_seconds,
                "worker",
                worker_log_path,
            )
        except Exception:
            terminate(worker)
            raise

        manifest["measurement"]["worker_pid"] = worker.pid
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with uprof_log_path.open("w", encoding="utf-8") as uprof_log:
            profiler = subprocess.Popen(
                profiler_command,
                stdout=uprof_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_file(
                    start,
                    profiler,
                    args.profiler_start_timeout_seconds,
                    "uProf/gate",
                    uprof_log_path,
                )
                worker_returncode = worker.wait(
                    timeout=args.measurement_timeout_seconds
                )
            except Exception:
                terminate(worker)
                terminate(profiler)
                raise

            if worker_returncode != 0:
                terminate(profiler)
                raise RuntimeError(
                    f"worker exited with {worker_returncode}; see {worker_log_path}"
                )
            try:
                profiler_returncode = profiler.wait(timeout=60)
            except subprocess.TimeoutExpired:
                terminate(profiler)
                raise TimeoutError(
                    f"uProf did not stop after the gate completed; see {uprof_log_path}"
                )

    if profiler_returncode != 0:
        raise RuntimeError(
            f"uProf exited with {profiler_returncode}; see {uprof_log_path}"
        )
    if not benchmark_path.is_file():
        raise RuntimeError(f"worker result is missing: {benchmark_path}")
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise RuntimeError(f"uProf CSV is missing or empty: {csv_path}")
    if not gate_timing_path.is_file():
        raise RuntimeError(f"gate timing is missing: {gate_timing_path}")

    gate_timing = json.loads(gate_timing_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["profiler"]["returncode"] = profiler_returncode
    manifest["outputs"] = {
        "benchmark": str(benchmark_path),
        "uprof_csv": str(csv_path),
        "uprof_log": str(uprof_log_path),
        "worker_log": str(worker_log_path),
        "gate_timing": str(gate_timing_path),
    }
    manifest["measurement"]["counter_active_seconds"] = gate_timing[
        "active_seconds"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "median_ms": benchmark["run"]["median_ms"],
                "measurement_wall_seconds": benchmark["run"]["measurement_wall_seconds"],
                "counter_active_seconds": gate_timing["active_seconds"],
                "uprof_csv_bytes": csv_path.stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
