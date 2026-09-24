#!/usr/bin/env python3
"""Analyze a native PACE K sweep with held-out surrogate validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from imbps_bench.pace_analysis import bootstrap_median_ci, fit_empirical_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--training-k", default="1,2,4,8,16")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260924)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_k = {int(value) for value in args.training_k.split(",")}
    with args.raw.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_k: dict[int, dict[int, float]] = {}
    for row in source_rows:
        split_k = int(row["split_k"])
        pair_index = int(row["pair_index"])
        by_k.setdefault(split_k, {})[pair_index] = float(row["latency_ms"])
    if 1 not in by_k:
        raise ValueError("K=1 is required")
    missing_training = sorted(training_k - set(by_k))
    if missing_training:
        raise ValueError(f"training K values absent from sweep: {missing_training}")

    medians = {
        split_k: statistics.median(observations.values())
        for split_k, observations in by_k.items()
    }
    fit = fit_empirical_surrogate(
        {split_k: medians[split_k] for split_k in sorted(training_k)}
    )
    measured_best_k = min(medians, key=medians.get)
    predicted_best_k = min(medians, key=lambda split_k: fit.predict(split_k))
    baseline = medians[1]
    output_rows = []
    for split_k in sorted(by_k):
        values = list(by_k[split_k].values())
        ci_low, ci_high = bootstrap_median_ci(
            values,
            samples=args.bootstrap_samples,
            seed=args.seed + split_k,
        )
        common_pairs = sorted(set(by_k[split_k]) & set(by_k[measured_best_k]))
        paired_deltas = [
            by_k[split_k][pair] - by_k[measured_best_k][pair]
            for pair in common_pairs
        ]
        delta_low, delta_high = bootstrap_median_ci(
            paired_deltas,
            samples=args.bootstrap_samples,
            seed=args.seed + 10_000 + split_k,
        )
        prediction = fit.predict(split_k)
        median_ms = medians[split_k]
        output_rows.append(
            {
                "split_k": split_k,
                "role": "train" if split_k in training_k else "held_out",
                "observations": len(values),
                "median_ms": median_ms,
                "median_ci95_low_ms": ci_low,
                "median_ci95_high_ms": ci_high,
                "speedup_vs_k1": baseline / median_ms,
                "predicted_ms": prediction,
                "prediction_error_ms": prediction - median_ms,
                "prediction_abs_pct_error": abs(prediction - median_ms) / median_ms * 100.0,
                "paired_delta_vs_measured_best_ms": statistics.median(paired_deltas),
                "paired_delta_ci95_low_ms": delta_low,
                "paired_delta_ci95_high_ms": delta_high,
            }
        )

    held_out_errors = [
        row["prediction_abs_pct_error"]
        for row in output_rows
        if row["role"] == "held_out"
    ]
    metadata = {
        "warning": (
            "c+a/K+bK is an empirical surrogate, not a derivation of the "
            "cache-spill model"
        ),
        "training_k": sorted(training_k),
        "held_out_k": sorted(set(by_k) - training_k),
        "fit": {
            "c": fit.c,
            "a": fit.a,
            "b": fit.b,
            "continuous_optimum": fit.continuous_optimum,
        },
        "measured_best_k": measured_best_k,
        "predicted_best_among_swept_k": predicted_best_k,
        "held_out_mean_abs_pct_error": (
            statistics.mean(held_out_errors) if held_out_errors else None
        ),
        "held_out_max_abs_pct_error": max(held_out_errors) if held_out_errors else None,
        "held_out_rmse_ms": (
            math.sqrt(
                statistics.mean(
                    row["prediction_error_ms"] ** 2
                    for row in output_rows
                    if row["role"] == "held_out"
                )
            )
            if held_out_errors
            else None
        ),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    args.output_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
