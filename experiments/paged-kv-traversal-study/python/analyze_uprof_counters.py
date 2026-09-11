#!/usr/bin/env python3
"""Summarise repeated AMD uProf measurements around the traversal crossover."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def median(rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {args.input}")

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["run_length"]), row["layout"])].append(row)

    print("AMD uProf MSR results (median across independent block-table seeds)")
    print(
        f"{'run':>4} {'layout':>6} {'n':>3} {'time ms':>9} {'GiB/s':>8} "
        f"{'IPC':>6} {'L3 miss':>9} {'L3 miss pti':>12} "
        f"{'L3 latency':>12} {'DRAM GB/s':>10}"
    )
    for run_length in sorted({key[0] for key in grouped}, reverse=True):
        for layout in ("BHND", "HBND"):
            group = grouped.get((run_length, layout), [])
            if not group:
                continue
            print(
                f"{run_length:>4} {layout:>6} {len(group):>3} "
                f"{median(group, 'median_ms'):>9.3f} "
                f"{median(group, 'gib_per_second'):>8.3f} "
                f"{median(group, 'ipc'):>6.3f} "
                f"{median(group, 'l3_miss_percent'):>8.2f}% "
                f"{median(group, 'l3_miss_per_thousand_instructions'):>12.2f} "
                f"{median(group, 'l3_miss_latency_ns'):>10.2f} ns "
                f"{median(group, 'memory_bandwidth_gb_s'):>10.2f}"
            )

    print()
    print("Paired HBND relative to BHND")
    print(
        f"{'run':>4} {'time':>10} {'L3 miss pp':>12} "
        f"{'L3 miss pti':>13} {'IPC':>10} {'winner':>8}"
    )
    for run_length in sorted({key[0] for key in grouped}, reverse=True):
        bhnd = grouped.get((run_length, "BHND"), [])
        hbnd = grouped.get((run_length, "HBND"), [])
        if not bhnd or not hbnd:
            continue
        time_ratio = median(hbnd, "median_ms") / median(bhnd, "median_ms")
        miss_points = median(hbnd, "l3_miss_percent") - median(
            bhnd, "l3_miss_percent"
        )
        miss_pti_ratio = median(
            hbnd, "l3_miss_per_thousand_instructions"
        ) / median(bhnd, "l3_miss_per_thousand_instructions")
        ipc_ratio = median(hbnd, "ipc") / median(bhnd, "ipc")
        if time_ratio < 0.98:
            winner = "HBND"
        elif time_ratio > 1.02:
            winner = "BHND"
        else:
            winner = "tie"
        print(
            f"{run_length:>4} {time_ratio:>9.3f}x "
            f"{miss_points:>+11.2f} {miss_pti_ratio:>12.3f}x "
            f"{ipc_ratio:>9.3f}x {winner:>8}"
        )

    print()
    print(
        "L3 and DRAM counters are collected by hardware scope, not attributed "
        "exclusively to the process. Interpret them only from otherwise quiet, "
        "paired runs on the same core/CCX."
    )


if __name__ == "__main__":
    main()
