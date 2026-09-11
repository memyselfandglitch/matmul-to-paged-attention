#!/usr/bin/env python3
"""Rank Phase 1 traversals for sequential and shuffled block tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {path}")

    parsed: dict[str, dict[str, float | str]] = {}
    for row in rows:
        traversal = row["traversal"]
        parsed[traversal] = {
            "memory_layout": row["memory_layout"],
            "median_ms": float(row["median_ms"]),
            "gib_per_second": float(row["gib_per_second"]),
            "gflops": float(row["gflops"]),
            "max_abs_error": float(row["max_abs_error"]),
        }
    return parsed


def report(label: str, rows: dict[str, dict[str, float | str]]) -> None:
    ordered = sorted(rows.items(), key=lambda item: float(item[1]["median_ms"]))
    best_ms = float(ordered[0][1]["median_ms"])

    print(label)
    print(
        f"{'rank':>4}  {'traversal':<10} {'median ms':>11} "
        f"{'vs best':>10} {'GiB/s':>10} {'GFLOP/s':>10}"
    )
    for rank, (traversal, row) in enumerate(ordered, start=1):
        median_ms = float(row["median_ms"])
        bandwidth = float(row["gib_per_second"])
        throughput = float(row["gflops"])
        print(
            f"{rank:>4}  {traversal:<10} {median_ms:>11.3f} "
            f"{median_ms / best_ms:>9.3f}x {bandwidth:>10.3f} "
            f"{throughput:>10.3f}"
        )

    block_ms = float(rows["BHND"]["median_ms"])
    head_ms = float(rows["HBND"]["median_ms"])
    matched_ms = float(rows["BNHD"]["median_ms"])
    print(f"winner: {ordered[0][0]}")
    print(f"head-first / block-first: {head_ms / block_ms:.3f}x")
    print(f"block-first / NHD-matched: {block_ms / matched_ms:.3f}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequential", type=Path)
    parser.add_argument("shuffled", type=Path)
    args = parser.parse_args()

    sequential = load(args.sequential)
    shuffled = load(args.shuffled)
    required = {"BHND", "HBND", "BNHD"}
    for path, rows in ((args.sequential, sequential), (args.shuffled, shuffled)):
        if set(rows) != required:
            raise ValueError(f"{path} must contain exactly {sorted(required)}")
        if any(float(row["max_abs_error"]) != 0.0 for row in rows.values()):
            raise ValueError(f"correctness error reported in {path}")

    report("Sequential physical block table", sequential)
    report("Shuffled physical block table", shuffled)

    sequential_winner = min(
        sequential, key=lambda traversal: float(sequential[traversal]["median_ms"])
    )
    shuffled_winner = min(
        shuffled, key=lambda traversal: float(shuffled[traversal]["median_ms"])
    )
    if sequential_winner == shuffled_winner:
        print(f"Both block-table patterns favour {sequential_winner}.")
    else:
        print(
            "The winner changes with block placement: "
            f"{sequential_winner} sequential, {shuffled_winner} shuffled."
        )
        print("Do not select a Phase 2 memory layout from only one pattern.")


if __name__ == "__main__":
    main()
