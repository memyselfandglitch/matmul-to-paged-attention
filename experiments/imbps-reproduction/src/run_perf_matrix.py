#!/usr/bin/env python3
"""Collect process-scoped perf counters after PACE warmup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_standalone_matrix import CLAIMS_PATH, PROJECT_ROOT, WORKER_PATH, build_cases, load_claim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claim",
        choices=("table_ii", "table_viii", "decode_exploratory"),
        required=True,
    )
    parser.add_argument("--models", default=None)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--events", default="cycles,instructions,cache-references,cache-misses")
    parser.add_argument("--l3-event", default=None)
    parser.add_argument("--l2-event", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--setup-timeout-seconds", type=float, default=1800)
    parser.add_argument("--measurement-timeout-seconds", type=float, default=7200)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.models:
        args.models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.l3_event and args.l2_event:
        parser.error("choose only one of --l3-event and --l2-event per run")
    named_cache_event = args.l3_event or args.l2_event
    if named_cache_event:
        args.events = f"{args.events},{named_cache_event}"
    return args


def parse_perf(path: Path) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        raw_value = fields[0].strip()
        event = fields[2].strip()
        if not event or raw_value.startswith("<"):
            continue
        try:
            counters[event] = float(raw_value)
        except ValueError:
            continue
    return counters


def main() -> None:
    args = parse_args()
    if shutil.which("perf") is None:
        raise SystemExit("perf is not installed or not on PATH")
    registry, claim = load_claim(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.result_dir or PROJECT_ROOT / "results" / f"{args.claim}-perf-{timestamp}"
    raw_dir = result_dir / "raw"
    log_dir = result_dir / "logs"
    control_dir = result_dir / "control"
    for directory in (raw_dir, log_dir, control_dir):
        directory.mkdir(parents=True, exist_ok=False)

    cases = build_cases(registry, claim)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = (
            f"{index:02d}-{case['model']}-b{case['batch']}-"
            f"{case['backend']}-k{case['split']}"
        )
        output_path = raw_dir / f"{case_id}.json"
        perf_path = raw_dir / f"{case_id}-perf.csv"
        ready_path = control_dir / f"{case_id}.ready"
        start_path = control_dir / f"{case_id}.start"
        worker_log_path = log_dir / f"{case_id}-worker.log"
        perf_log_path = log_dir / f"{case_id}-perf.log"
        command = [
            sys.executable,
            str(WORKER_PATH),
            "--backend",
            case["backend"],
            "--batch",
            str(case["batch"]),
            "--sequence",
            str(case["sequence"]),
            "--hidden",
            str(case["hidden"]),
            "--intermediate",
            str(case["intermediate"]),
            "--splits",
            str(case["split"]),
            "--dtype",
            case["dtype"],
            "--activation",
            case["activation"],
            "--warmups",
            str(args.warmups),
            "--iterations",
            str(args.iterations),
            "--output",
            str(output_path),
            "--ready-file",
            str(ready_path),
            "--start-file",
            str(start_path),
            "--start-timeout",
            str(args.setup_timeout_seconds),
        ]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))

        with worker_log_path.open("w", encoding="utf-8") as worker_log:
            worker = subprocess.Popen(command, stdout=worker_log, stderr=subprocess.STDOUT, text=True)
            setup_deadline = time.monotonic() + args.setup_timeout_seconds
            while not ready_path.exists():
                if worker.poll() is not None:
                    raise RuntimeError(f"worker {case_id} exited before profiler attach; see {worker_log_path}")
                if time.monotonic() >= setup_deadline:
                    worker.terminate()
                    raise TimeoutError(f"worker {case_id} did not become ready; see {worker_log_path}")
                time.sleep(0.1)
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            tids = ",".join(str(tid) for tid in ready["tids"])
            perf_command = [
                "perf",
                "stat",
                "--no-big-num",
                "-x",
                ";",
                "-e",
                args.events,
                "-o",
                str(perf_path),
                "-t",
                tids,
            ]
            with perf_log_path.open("w", encoding="utf-8") as perf_log:
                perf_process = subprocess.Popen(
                    perf_command, stdout=perf_log, stderr=subprocess.STDOUT, text=True
                )
                time.sleep(0.5)
                start_path.write_text("start\n", encoding="utf-8")
                try:
                    returncode = worker.wait(timeout=args.measurement_timeout_seconds)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    raise TimeoutError(f"measurement timed out for {case_id}")
                if returncode != 0:
                    raise RuntimeError(f"worker {case_id} failed; see {worker_log_path}")
                try:
                    perf_returncode = perf_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    perf_process.terminate()
                    perf_returncode = perf_process.wait(timeout=10)
                if perf_returncode != 0:
                    raise RuntimeError(f"perf failed for {case_id}; see {perf_log_path}")

        benchmark = json.loads(output_path.read_text(encoding="utf-8"))
        counters = parse_perf(perf_path)
        row = {
            "claim": args.claim,
            "model": case["model"],
            "batch": case["batch"],
            "sequence": case["sequence"],
            "activation": case["activation"],
            "backend": case["backend"],
            "split": case["split"],
            "median_ms": benchmark["run"]["median_ms"],
            "profile_scope": "all worker TIDs present after warmup",
            "events_requested": args.events,
            **counters,
        }
        rows.append(row)
        print(f"profiled {case_id}: {len(counters)} counters", flush=True)

    all_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    with (result_dir / "counters.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=all_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": args.claim,
        "paper": registry["paper"],
        "pace": registry["pace"],
        "events": args.events.split(","),
        "l3_event": args.l3_event,
        "l2_event": args.l2_event,
        "scope_warning": (
            "Generic cache-misses is not labeled L2 or L3. Results are a cache-level "
            "replication only when the corresponding named event is validated on this "
            "CPU/kernel and has a documented task scope."
        ),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
