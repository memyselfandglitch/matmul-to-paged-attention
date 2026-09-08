#!/usr/bin/env python3
"""Summarise timing, bandwidth, cache and TLB counters from Linux perf."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return float(value) if value else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {args.input}")

    print("Matched-layout timing and hardware counters")
    print(
        f"{'run':>6} {'layout':>7} {'ms':>9} {'GiB/s':>8} {'IPC':>7} "
        f"{'cache miss':>12} {'L1D miss':>10} {'LLC miss':>10} {'dTLB miss':>11}"
    )
    for row in rows:
        ipc = ratio(number(row, "instructions"), number(row, "cycles"))
        cache_miss = ratio(number(row, "cache-misses"), number(row, "cache-references"))
        l1_miss = ratio(
            number(row, "L1-dcache-load-misses"), number(row, "L1-dcache-loads")
        )
        llc_miss = ratio(number(row, "LLC-load-misses"), number(row, "LLC-loads"))
        dtlb_miss = ratio(
            number(row, "dTLB-load-misses"), number(row, "dTLB-loads")
        )
        print(
            f"{int(row['run_length']):>6} {row['layout']:>7} "
            f"{float(row['median_ms']):>9.3f} "
            f"{float(row['gib_per_second']):>8.3f} {decimal(ipc):>7} "
            f"{percentage(cache_miss):>12} {percentage(l1_miss):>10} "
            f"{percentage(llc_miss):>10} {percentage(dtlb_miss):>11}"
        )

    intensities = sorted({float(row["useful_flops_per_kv_byte"]) for row in rows})
    print()
    print(f"Useful arithmetic intensity: {intensities} FLOP/KV byte")
    print(
        "Counters cover process startup plus warmups and repetitions; the high "
        "repetition count makes the decode kernels dominate the totals."
    )


if __name__ == "__main__":
    main()
