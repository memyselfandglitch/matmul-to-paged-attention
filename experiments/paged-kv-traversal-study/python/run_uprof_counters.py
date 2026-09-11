#!/usr/bin/env python3
"""Collect AMD uProf MSR counters for isolated matched-layout decode cases."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


METRICS = {
    "Retired Instructions": "retired_instructions",
    "Cycles Not in Halt": "cycles_not_halted",
    "IPC (Sys + User)": "ipc",
    "L3 Access": "l3_access",
    "L3 Miss": "l3_miss",
    "L3 Access (pti)": "l3_access_per_thousand_instructions",
    "L3 Miss (pti)": "l3_miss_per_thousand_instructions",
    "L3 Miss %": "l3_miss_percent",
    "Ave L3 Miss Latency (ns)": "l3_miss_latency_ns",
    "L3 Miss Latency From Local Memory or I/O (%)": "l3_local_memory_percent",
    "L3 Miss Latency From Remote Memory or I/O (%)": "l3_remote_memory_percent",
    "Total Mem Bw (GB/s)": "memory_bandwidth_gb_s",
    "Total Mem RdBw (GB/s)": "memory_read_bandwidth_gb_s",
    "Total Mem WrBw (GB/s)": "memory_write_bandwidth_gb_s",
}


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_uprof_report(path: Path) -> dict[str, float]:
    report = path.read_text(encoding="utf-8")
    if "Data Collection Driver:,MSR Driver" not in report:
        raise RuntimeError(f"{path} was not collected with the MSR driver")

    values: dict[str, float] = {}
    for row in csv.reader(report.splitlines()):
        if len(row) < 2:
            continue
        output_name = METRICS.get(row[0].strip())
        if output_name is None:
            continue
        try:
            values[output_name] = float(row[1].strip())
        except ValueError:
            continue

    missing = sorted(set(METRICS.values()) - set(values))
    if missing:
        raise RuntimeError(f"uProf report {path} lacks metrics: {', '.join(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--uprof-pcm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--run-lengths", default="8,4,3,2,1")
    parser.add_argument("--layouts", default="BHND,HBND")
    parser.add_argument("--block-seeds", default="0,1,2")
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument("--ccx", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    run_lengths = [int(value) for value in comma_list(args.run_lengths)]
    block_seeds = [int(value) for value in comma_list(args.block_seeds)]
    layouts = comma_list(args.layouts)
    if any(length <= 0 or length > args.blocks for length in run_lengths):
        parser.error("run lengths must be between 1 and --blocks")
    if not layouts or any(layout not in {"BHND", "HBND"} for layout in layouts):
        parser.error("--layouts must contain BHND and/or HBND")
    if not block_seeds:
        parser.error("--block-seeds must not be empty")
    if not args.uprof_pcm.is_file():
        parser.error(f"AMD uProf PCM binary not found: {args.uprof_pcm}")
    if shutil.which("taskset") is None:
        parser.error("taskset is required to pin the measured process")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    output_csv = args.output_dir / "uprof-counters.csv"
    fields = (
        "blocks",
        "tokens",
        "heads",
        "block_size",
        "head_dim",
        "run_length",
        "block_seed",
        "layout",
        "profile_core",
        "profile_ccx",
        "median_ms",
        "gib_per_second",
        "useful_flops_per_kv_byte",
        *METRICS.values(),
    )

    binary = args.binary.resolve()
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run_length in run_lengths:
            for block_seed in block_seeds:
                for layout in layouts:
                    case_name = f"{layout.lower()}-run{run_length}-seed{block_seed}"
                    timing_csv = raw_dir / f"{case_name}-timing.csv"
                    uprof_csv = raw_dir / f"{case_name}-uprof.csv"
                    benchmark = [
                        str(binary),
                        "--memory-layout",
                        layout,
                        "--traversal",
                        layout,
                        "--block-order",
                        "fragmented",
                        "--block-run-length",
                        str(run_length),
                        "--block-seed",
                        str(block_seed),
                        "--blocks",
                        str(args.blocks),
                        "--heads",
                        str(args.heads),
                        "--block-size",
                        str(args.block_size),
                        "--head-dim",
                        str(args.head_dim),
                        "--warmups",
                        str(args.warmups),
                        "--repeats",
                        str(args.repeats),
                        "--no-cache-scrub",
                        "--csv",
                        str(timing_csv),
                    ]
                    command = [
                        str(args.uprof_pcm.resolve()),
                        "--msr",
                        "-m",
                        "ipc,l3,memory",
                        "-c",
                        f"ccx={args.ccx}",
                        "-C",
                        "-q",
                        "-o",
                        str(uprof_csv),
                        "--",
                        "taskset",
                        "-c",
                        str(args.core),
                        *benchmark,
                    ]
                    completed = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"uProf failed for {case_name}\n"
                            f"stdout:\n{completed.stdout}\n"
                            f"stderr:\n{completed.stderr}"
                        )

                    with timing_csv.open(newline="", encoding="utf-8") as source:
                        timing_rows = list(csv.DictReader(source))
                    if len(timing_rows) != 1:
                        raise RuntimeError(f"{case_name} returned {len(timing_rows)} timing rows")
                    timing = timing_rows[0]
                    counters = parse_uprof_report(uprof_csv)
                    writer.writerow(
                        {
                            "blocks": args.blocks,
                            "tokens": args.blocks * args.block_size,
                            "heads": args.heads,
                            "block_size": args.block_size,
                            "head_dim": args.head_dim,
                            "run_length": run_length,
                            "block_seed": block_seed,
                            "layout": layout,
                            "profile_core": args.core,
                            "profile_ccx": args.ccx,
                            "median_ms": timing["median_ms"],
                            "gib_per_second": timing["gib_per_second"],
                            "useful_flops_per_kv_byte": timing[
                                "useful_flops_per_kv_byte"
                            ],
                            **counters,
                        }
                    )
                    destination.flush()
                    print(
                        f"run={run_length:<2} seed={block_seed} layout={layout} "
                        f"time={float(timing['median_ms']):.3f} ms "
                        f"L3-miss={counters['l3_miss_percent']:.2f}%"
                    )

    print(f"Combined measurements: {output_csv}")


if __name__ == "__main__":
    main()
