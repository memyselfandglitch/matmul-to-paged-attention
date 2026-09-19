#!/usr/bin/env python3
"""Compute paired speedups, bootstrap intervals, and exact sign tests."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


FIELDS = (
    "metric",
    "split_k",
    "pairs",
    "reference_median",
    "imbps_median",
    "ratio_of_medians_speedup",
    "paired_speedup_median",
    "paired_speedup_ci95_low",
    "paired_speedup_ci95_high",
    "paired_time_change_pct_median",
    "imbps_faster_pairs",
    "exact_sign_test_p_two_sided",
)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _sign_test_two_sided(successes: int, trials: int) -> float:
    tail = sum(math.comb(trials, index) for index in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--metrics", required=True, help="comma-separated numeric columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    output_rows = []
    for metric in metrics:
        for split_k in sorted({int(row["split_k"]) for row in rows}):
            subset = [row for row in rows if int(row["split_k"]) == split_k]
            by_pair: dict[str, dict[str, float]] = {}
            for row in subset:
                pair_key = row.get("iteration", "")
                by_pair.setdefault(pair_key, {})[row["variant"]] = float(row[metric])
            pairs = [
                (variants["reference"], variants["imbps"])
                for variants in by_pair.values()
                if "reference" in variants and "imbps" in variants
            ]
            if not pairs:
                raise ValueError(f"no complete pairs for K={split_k}, metric={metric}")
            reference = [pair[0] for pair in pairs]
            imbps = [pair[1] for pair in pairs]
            speedups = [ref / candidate for ref, candidate in pairs]
            bootstrapped = []
            for _ in range(args.bootstrap_samples):
                sample = [speedups[rng.randrange(len(speedups))] for _ in speedups]
                bootstrapped.append(statistics.median(sample))
            faster = sum(candidate < ref for ref, candidate in pairs)
            output_rows.append(
                {
                    "metric": metric,
                    "split_k": split_k,
                    "pairs": len(pairs),
                    "reference_median": statistics.median(reference),
                    "imbps_median": statistics.median(imbps),
                    "ratio_of_medians_speedup": statistics.median(reference) / statistics.median(imbps),
                    "paired_speedup_median": statistics.median(speedups),
                    "paired_speedup_ci95_low": _percentile(bootstrapped, 0.025),
                    "paired_speedup_ci95_high": _percentile(bootstrapped, 0.975),
                    "paired_time_change_pct_median": statistics.median(
                        (candidate / ref - 1.0) * 100.0 for ref, candidate in pairs
                    ),
                    "imbps_faster_pairs": faster,
                    "exact_sign_test_p_two_sided": _sign_test_two_sided(faster, len(pairs)),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            "%s K=%-2d speedup=%7.4fx CI95=[%7.4f,%7.4f] faster=%d/%d p=%g"
            % (
                row["metric"],
                row["split_k"],
                row["paired_speedup_median"],
                row["paired_speedup_ci95_low"],
                row["paired_speedup_ci95_high"],
                row["imbps_faster_pairs"],
                row["pairs"],
                row["exact_sign_test_p_two_sided"],
            )
        )


if __name__ == "__main__":
    main()
