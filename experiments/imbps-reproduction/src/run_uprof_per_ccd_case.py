#!/usr/bin/env python3
"""Profile one pre-warmed per-CCD benchmark with CCD-aggregated uProf data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_uprof_case import (
        build_gate_command,
        terminate,
        wait_for_file,
    )
except ImportError:  # Direct execution
    from run_uprof_case import build_gate_command, terminate, wait_for_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = PROJECT_ROOT / "src" / "per_ccd_mlp_bench.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uprof-bin", type=Path, required=True)
    parser.add_argument("--access-mode", choices=("perf", "msr"), default="perf")
    parser.add_argument("--metrics", default="ipc,l2,l3")
    parser.add_argument("--package", type=int, default=0)
    parser.add_argument("--aggregation", choices=("ccd", "ccx"), default="ccd")
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-measurement-seconds", type=float, default=5.0)
    parser.add_argument("--setup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--measurement-timeout-seconds", type=float, default=7200)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    ready = result_dir / "worker.ready.json"
    start = result_dir / "worker.start"
    done = result_dir / "worker.measurement.done"
    benchmark = result_dir / "benchmark.json"
    csv_path = result_dir / "uprof.csv"
    worker_log_path = result_dir / "worker.log"
    uprof_log_path = result_dir / "uprof.log"
    gate_timing = result_dir / "gate-timing.json"

    worker_command = [
        "numactl",
        f"--physcpubind={args.cpu_list}",
        "--membind=0",
        sys.executable,
        str(WORKER_PATH),
        "--claim",
        args.claim,
        "--variant",
        args.variant,
        "--rows",
        str(args.rows),
        "--threads",
        str(args.threads),
        "--warmups",
        str(args.warmups),
        "--rounds",
        str(args.rounds),
        "--min-measurement-seconds",
        str(args.min_measurement_seconds),
        "--ready-file",
        str(ready),
        "--start-file",
        str(start),
        "--done-file",
        str(done),
        "--start-timeout",
        str(args.measurement_timeout_seconds),
        "--output",
        str(benchmark),
    ]
    gate_command = build_gate_command(
        start, done, gate_timing, args.measurement_timeout_seconds
    )
    access = ["-X"] if args.access_mode == "perf" else ["--msr"]
    profiler_command = [
        str(args.uprof_bin),
        *access,
        "-m",
        args.metrics,
        "-c",
        f"package={args.package}",
        "-A",
        args.aggregation,
        "-C",
        "-o",
        str(csv_path),
        "--",
        *gate_command,
    ]
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "case": {
            "claim": args.claim,
            "variant": args.variant,
            "rows": args.rows,
        },
        "profiler": {
            "binary": str(args.uprof_bin),
            "access_mode": args.access_mode,
            "metrics": args.metrics.split(","),
            "scope": f"package={args.package}",
            "aggregation": args.aggregation,
            "command": profiler_command,
        },
        "measurement": {
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "warmups": args.warmups,
            "requested_rounds": args.rounds,
            "minimum_seconds": args.min_measurement_seconds,
        },
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with worker_log_path.open("w", encoding="utf-8") as worker_log:
        worker = subprocess.Popen(
            worker_command, stdout=worker_log, stderr=subprocess.STDOUT, text=True
        )
        try:
            wait_for_file(
                ready,
                worker,
                args.setup_timeout_seconds,
                "per-CCD worker",
                worker_log_path,
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
                        30,
                        "uProf/gate",
                        uprof_log_path,
                    )
                    worker_returncode = worker.wait(
                        timeout=args.measurement_timeout_seconds
                    )
                    if worker_returncode != 0:
                        terminate(profiler)
                        profiler_returncode = profiler.wait(timeout=10)
                    else:
                        profiler_returncode = profiler.wait(timeout=60)
                except Exception:
                    terminate(worker)
                    terminate(profiler)
                    raise
        except Exception:
            terminate(worker)
            raise

    if worker_returncode != 0:
        raise RuntimeError(
            f"worker exited with {worker_returncode}; see {worker_log_path}"
        )
    if profiler_returncode != 0:
        raise RuntimeError(
            f"uProf exited with {profiler_returncode}; see {uprof_log_path}"
        )
    if not benchmark.is_file() or not csv_path.is_file() or not gate_timing.is_file():
        raise RuntimeError("per-CCD profiling outputs are incomplete")

    result = json.loads(benchmark.read_text(encoding="utf-8"))
    gate = json.loads(gate_timing.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["measurement"]["actual_invocations"] = len(
        result["records"][0]["durations_ms"]
    )
    manifest["measurement"]["counter_active_seconds"] = gate["active_seconds"]
    manifest["outputs"] = {
        "benchmark": str(benchmark),
        "uprof_csv": str(csv_path),
        "worker_log": str(worker_log_path),
        "uprof_log": str(uprof_log_path),
        "gate_timing": str(gate_timing),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result_dir)


if __name__ == "__main__":
    main()
