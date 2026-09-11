#!/usr/bin/env python3
"""Analyse the complete Phase 2 physical-layout/traversal matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LAYOUTS = ("BNHD", "BHND", "HBND")


def load(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {path}")

    parsed: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["memory_layout"], row["traversal"])
        if key in parsed:
            raise ValueError(f"duplicate measurement {key} in {path}")
        parsed[key] = {
            "median_ms": float(row["median_ms"]),
            "gib_per_second": float(row["gib_per_second"]),
            "gflops": float(row["gflops"]),
            "max_abs_error": float(row["max_abs_error"]),
        }

    expected = {(memory, traversal) for memory in LAYOUTS for traversal in LAYOUTS}
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ValueError(f"{path} has the wrong matrix; missing={missing}, extra={extra}")
    if any(row["max_abs_error"] != 0.0 for row in parsed.values()):
        raise ValueError(f"correctness error reported in {path}")
    return parsed


def winner(
    rows: dict[tuple[str, str], dict[str, float]], memory: str
) -> tuple[str, float]:
    traversal = min(
        LAYOUTS, key=lambda order: rows[(memory, order)]["median_ms"]
    )
    return traversal, rows[(memory, traversal)]["median_ms"]


def report_pattern(
    label: str, rows: dict[tuple[str, str], dict[str, float]]
) -> None:
    print(label)
    print("Median decode time (ms); rows are physical memory layouts")
    print(
        f"{'memory':<8}"
        + "".join(f"{traversal:>12}" for traversal in LAYOUTS)
        + f"  {'winner':>8}  {'matched/best':>13}"
    )

    matched_wins = 0
    for memory in LAYOUTS:
        best_traversal, best_ms = winner(rows, memory)
        matched_ms = rows[(memory, memory)]["median_ms"]
        if best_traversal == memory:
            matched_wins += 1
        values = "".join(
            f"{rows[(memory, traversal)]['median_ms']:>12.3f}"
            for traversal in LAYOUTS
        )
        print(
            f"{memory:<8}{values}  {best_traversal:>8}  "
            f"{matched_ms / best_ms:>12.3f}x"
        )

    print("\nUseful GFLOP/s; rows are physical memory layouts")
    print(f"{'memory':<8}" + "".join(f"{traversal:>12}" for traversal in LAYOUTS))
    for memory in LAYOUTS:
        values = "".join(
            f"{rows[(memory, traversal)]['gflops']:>12.3f}"
            for traversal in LAYOUTS
        )
        print(f"{memory:<8}{values}")

    global_pair = min(rows, key=lambda pair: rows[pair]["median_ms"])
    global_ms = rows[global_pair]["median_ms"]
    print(f"matched traversal wins: {matched_wins}/{len(LAYOUTS)} layouts")
    global_result = rows[global_pair]
    print(
        "global winner: "
        f"memory={global_pair[0]}, traversal={global_pair[1]}, {global_ms:.3f} ms, "
        f"{global_result['gib_per_second']:.3f} GiB/s, "
        f"{global_result['gflops']:.3f} GFLOP/s"
    )
    print("Head-first versus block-first within each physical layout:")
    for memory in LAYOUTS:
        block_ms = rows[(memory, "BHND")]["median_ms"]
        head_ms = rows[(memory, "HBND")]["median_ms"]
        print(f"  {memory}: HBND/BHND = {head_ms / block_ms:.3f}x")
    print()


def report_block_table_sensitivity(
    sequential: dict[tuple[str, str], dict[str, float]],
    shuffled: dict[tuple[str, str], dict[str, float]],
) -> None:
    print("Block-table sensitivity (shuffled/sequential time)")
    print(f"{'memory':<8}{'traversal':<12}{'ratio':>10}{'change':>12}")
    for memory in LAYOUTS:
        for traversal in LAYOUTS:
            sequential_ms = sequential[(memory, traversal)]["median_ms"]
            shuffled_ms = shuffled[(memory, traversal)]["median_ms"]
            ratio = shuffled_ms / sequential_ms
            print(
                f"{memory:<8}{traversal:<12}{ratio:>9.3f}x"
                f"{(ratio - 1.0) * 100.0:>+11.1f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequential", type=Path)
    parser.add_argument("shuffled", type=Path)
    args = parser.parse_args()

    sequential = load(args.sequential)
    shuffled = load(args.shuffled)

    report_pattern("Sequential physical block table", sequential)
    report_pattern("Shuffled physical block table", shuffled)
    report_block_table_sensitivity(sequential, shuffled)


if __name__ == "__main__":
    main()
