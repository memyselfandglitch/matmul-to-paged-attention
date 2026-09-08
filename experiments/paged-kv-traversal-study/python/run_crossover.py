#!/usr/bin/env python3
"""Measure matched layouts over a controlled physical-block fragmentation curve."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


LAYOUTS = ("BNHD", "BHND", "HBND")
MEASUREMENT_FIELDS = (
    "memory_layout",
    "traversal",
    "median_ms",
    "gib_per_second",
    "gflops",
    "useful_flops_per_kv_byte",
    "checksum",
    "max_abs_error",
)


def parse_run_lengths(value: str, blocks: int) -> list[int]:
    lengths = [int(item) for item in value.split(",")]
    if not lengths or any(length <= 0 or length > blocks for length in lengths):
        raise argparse.ArgumentTypeError(
            f"run lengths must be between 1 and the block count ({blocks})"
        )
    return list(dict.fromkeys(lengths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--run-lengths", default="512,256,128,64,32,16,8,4,2,1")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    if min(
        args.blocks,
        args.heads,
        args.block_size,
        args.head_dim,
        args.trials,
        args.warmups,
        args.repeats,
    ) <= 0:
        parser.error("all numeric dimensions and repetition counts must be positive")
    run_lengths = parse_run_lengths(args.run_lengths, args.blocks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "blocks",
        "tokens",
        "heads",
        "block_size",
        "head_dim",
        "run_length",
        "num_runs",
        "trial",
        *MEASUREMENT_FIELDS,
    )

    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()

        with tempfile.TemporaryDirectory(prefix="paged-kv-crossover-") as temporary:
            temporary_csv = Path(temporary) / "measurement.csv"
            total = len(run_lengths) * args.trials
            completed = 0
            for run_length in run_lengths:
                for trial in range(args.trials):
                    command = [
                        str(args.binary),
                        "--stage",
                        "matched",
                        "--block-order",
                        "fragmented",
                        "--block-run-length",
                        str(run_length),
                        "--block-seed",
                        str(trial),
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
                        "--csv",
                        str(temporary_csv),
                    ]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

                    with temporary_csv.open(newline="", encoding="utf-8") as source:
                        measurements = list(csv.DictReader(source))
                    if {row["memory_layout"] for row in measurements} != set(LAYOUTS):
                        raise RuntimeError("matched run did not return all three layouts")

                    for measurement in measurements:
                        row = {
                            "blocks": args.blocks,
                            "tokens": args.blocks * args.block_size,
                            "heads": args.heads,
                            "block_size": args.block_size,
                            "head_dim": args.head_dim,
                            "run_length": run_length,
                            "num_runs": (args.blocks + run_length - 1) // run_length,
                            "trial": trial,
                        }
                        row.update(measurement)
                        writer.writerow(row)
                    destination.flush()
                    completed += 1
                    print(
                        f"[{completed:>3}/{total}] run_length={run_length:<5} "
                        f"trial={trial}"
                    )

    print(f"Raw crossover measurements: {args.output}")


if __name__ == "__main__":
    main()
