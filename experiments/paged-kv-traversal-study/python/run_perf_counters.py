#!/usr/bin/env python3
"""Collect Linux perf counters for isolated matched-layout decode cases."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


EVENTS = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
    "dTLB-loads",
    "dTLB-load-misses",
)


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_perf(path: Path) -> dict[str, float]:
    counters: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        raw_value, event = fields[0].strip(), fields[2].strip().split(":", 1)[0]
        if raw_value.startswith("<"):
            continue
        try:
            counters[event] = float(raw_value)
        except ValueError:
            continue
    return counters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--run-lengths", default="512,32,16,8,4,1")
    parser.add_argument("--layouts", default="BHND,HBND")
    parser.add_argument("--block-seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if shutil.which("perf") is None:
        parser.error("Linux perf is not installed or is not on PATH")
    run_lengths = [int(value) for value in parse_list(args.run_lengths)]
    layouts = parse_list(args.layouts)
    if any(length <= 0 or length > args.blocks for length in run_lengths):
        parser.error("run lengths must be between 1 and --blocks")
    if not layouts or any(layout not in {"BNHD", "BHND", "HBND"} for layout in layouts):
        parser.error("layouts must contain only BNHD, BHND, or HBND")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_fields = (
        "blocks",
        "tokens",
        "heads",
        "block_size",
        "head_dim",
        "run_length",
        "block_seed",
        "layout",
        "median_ms",
        "gib_per_second",
        "useful_flops_per_kv_byte",
    )
    fieldnames = (*metadata_fields, *EVENTS)

    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        with tempfile.TemporaryDirectory(prefix="paged-kv-perf-") as temporary:
            temporary_dir = Path(temporary)
            for run_length in run_lengths:
                for layout in layouts:
                    measurement_csv = temporary_dir / "measurement.csv"
                    perf_output = temporary_dir / "perf.txt"
                    benchmark = [
                        str(args.binary),
                        "--memory-layout",
                        layout,
                        "--traversal",
                        layout,
                        "--block-order",
                        "fragmented",
                        "--block-run-length",
                        str(run_length),
                        "--block-seed",
                        str(args.block_seed),
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
                        str(measurement_csv),
                    ]
                    command = [
                        "perf",
                        "stat",
                        "--no-big-num",
                        "-x",
                        ";",
                        "-e",
                        ",".join(EVENTS),
                        "-o",
                        str(perf_output),
                        "--",
                        *benchmark,
                    ]
                    completed = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    if completed.returncode != 0:
                        details = perf_output.read_text(encoding="utf-8") if perf_output.exists() else ""
                        raise RuntimeError(
                            "perf failed. Check kernel.perf_event_paranoid or cluster "
                            f"permissions.\n{completed.stderr}\n{details}"
                        )

                    with measurement_csv.open(newline="", encoding="utf-8") as source:
                        measurements = list(csv.DictReader(source))
                    if len(measurements) != 1:
                        raise RuntimeError("isolated benchmark did not return exactly one row")
                    measurement = measurements[0]
                    counters = parse_perf(perf_output)
                    row: dict[str, str | int | float] = {
                        "blocks": args.blocks,
                        "tokens": args.blocks * args.block_size,
                        "heads": args.heads,
                        "block_size": args.block_size,
                        "head_dim": args.head_dim,
                        "run_length": run_length,
                        "block_seed": args.block_seed,
                        "layout": layout,
                        "median_ms": measurement["median_ms"],
                        "gib_per_second": measurement["gib_per_second"],
                        "useful_flops_per_kv_byte": measurement[
                            "useful_flops_per_kv_byte"
                        ],
                    }
                    row.update(counters)
                    writer.writerow(row)
                    destination.flush()
                    missing = sorted(set(EVENTS) - set(counters))
                    suffix = f"; unsupported: {','.join(missing)}" if missing else ""
                    print(f"run_length={run_length:<5} layout={layout}{suffix}")

    print(f"Perf measurements: {args.output}")


if __name__ == "__main__":
    main()
