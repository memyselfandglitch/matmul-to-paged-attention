#!/usr/bin/env python3
"""Combine cache and traffic uProf matrices into mechanism summaries."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--traffic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def bootstrap_median_ci(
    values: list[float], seed: int = 20261003, draws: int = 10_000
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.median(sample))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def metric_summary(values: list[float], seed: int) -> dict[str, float]:
    low, high = bootstrap_median_ci(values, seed=seed)
    return {
        "median": statistics.median(values),
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_ratios(
    rows: list[dict[str, Any]], metric: str, *, baseline_over_candidate: bool
) -> list[float]:
    baseline = {
        int(row["round"]): float(row[metric])
        for row in rows
        if row["backend"] == "tpp" and int(row["split"]) == 1
    }
    candidates = [row for row in rows if row["backend"] != "tpp"]
    ratios = []
    for row in candidates:
        round_number = int(row["round"])
        if round_number not in baseline:
            raise ValueError(f"missing TPP round {round_number} for {metric}")
        candidate = float(row[metric])
        ratios.append(
            baseline[round_number] / candidate
            if baseline_over_candidate
            else candidate / baseline[round_number]
        )
    return ratios


def summarize_pass(rows: list[dict[str, Any]], pass_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    by_shape: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["pass"] != pass_name:
            raise ValueError(f"expected {pass_name} row, got {row['pass']}")
        shape = (row["model"], int(row["batch"]), int(row["sequence"]))
        key = (*shape, row["backend"], int(row["split"]))
        grouped[key].append(row)
        by_shape[shape].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model, batch, sequence, backend, split = key
        case_rows = grouped[key]
        shape = (model, batch, sequence)
        expected_rounds = sorted({int(row["round"]) for row in by_shape[shape]})
        actual_rounds = sorted(int(row["round"]) for row in case_rows)
        if actual_rounds != expected_rounds:
            raise ValueError(
                f"incomplete rounds for B={batch} {backend} K={split}: "
                f"{actual_rounds} != {expected_rounds}"
            )
        seed = 20261003 + batch * 100 + split
        summary: dict[str, Any] = {
            "pass": pass_name,
            "model": model,
            "batch": batch,
            "sequence": sequence,
            "rows": batch * sequence,
            "backend": backend,
            "split": split,
            "rounds": len(case_rows),
            "cache_mib": case_rows[0]["cache_mib"],
            "equation12_working_set_mib": case_rows[0][
                "equation12_working_set_mib"
            ],
            "equation12_fits": case_rows[0]["equation12_fits"],
        }
        metric_names = ["median_ms"]
        if pass_name == "cache":
            metric_names.extend(
                [
                    "ipc",
                    "l2_access_pti",
                    "l2_miss_pti",
                    "l3_access_per_invocation",
                    "l3_miss_per_invocation",
                    "l3_hit_per_invocation",
                    "l3_miss_percent",
                    "l3_hit_percent",
                    "l3_miss_latency_ns",
                ]
            )
        else:
            metric_names.extend(
                [
                    "total_mem_bw_gbps",
                    "read_mem_bw_gbps",
                    "write_mem_bw_gbps",
                    "dram_bytes_per_invocation",
                    "dram_read_bytes_per_invocation",
                    "dram_write_bytes_per_invocation",
                    "measured_dram_ai_flops_per_byte",
                ]
            )
        for metric_index, metric in enumerate(metric_names):
            values = [float(row[metric]) for row in case_rows]
            for suffix, value in metric_summary(values, seed + metric_index).items():
                summary[f"{metric}_{suffix}"] = value

        if backend != "tpp":
            batch_rows = [
                row
                for row in by_shape[shape]
                if row["backend"] == "tpp"
                or (row["backend"] == backend and int(row["split"]) == split)
            ]
            ratio_specs = [("timing_speedup", "median_ms", True)]
            if pass_name == "cache":
                ratio_specs.extend(
                    [
                        (
                            "l3_miss_reduction_factor",
                            "l3_miss_per_invocation",
                            True,
                        ),
                        (
                            "l3_access_reduction_factor",
                            "l3_access_per_invocation",
                            True,
                        ),
                    ]
                )
            else:
                ratio_specs.extend(
                    [
                        (
                            "dram_byte_reduction_factor",
                            "dram_bytes_per_invocation",
                            True,
                        ),
                        (
                            "dram_ai_increase_factor",
                            "measured_dram_ai_flops_per_byte",
                            False,
                        ),
                    ]
                )
            for ratio_index, (name, metric, direction) in enumerate(ratio_specs):
                values = paired_ratios(
                    batch_rows, metric, baseline_over_candidate=direction
                )
                for suffix, value in metric_summary(
                    values, seed + 100 + ratio_index
                ).items():
                    summary[f"{name}_{suffix}"] = value
        summaries.append(summary)
    return summaries


def mechanism_checks(
    cache_summary: list[dict[str, Any]], traffic_summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def tied_splits(
        candidates: list[dict[str, Any]], metric: str, *, minimize: bool
    ) -> tuple[int, list[int]]:
        best = (min if minimize else max)(
            candidates, key=lambda row: row[f"{metric}_median"]
        )
        best_interval = (
            best[f"{metric}_ci95_low"],
            best[f"{metric}_ci95_high"],
        )
        ties = []
        for candidate in candidates:
            interval = (
                candidate[f"{metric}_ci95_low"],
                candidate[f"{metric}_ci95_high"],
            )
            if max(best_interval[0], interval[0]) <= min(
                best_interval[1], interval[1]
            ):
                ties.append(int(candidate["split"]))
        return int(best["split"]), sorted(ties)

    checks = []
    shapes = sorted(
        {
            (row["model"], int(row["batch"]), int(row["sequence"]))
            for row in cache_summary
        }
    )
    for model, batch, sequence in shapes:
        cache_candidates = [
            row
            for row in cache_summary
            if row["model"] == model
            and row["batch"] == batch
            and row["sequence"] == sequence
            and row["backend"] == "imbps"
        ]
        traffic_candidates = [
            row
            for row in traffic_summary
            if row["model"] == model
            and row["batch"] == batch
            and row["sequence"] == sequence
            and row["backend"] == "imbps"
        ]
        fastest_cache, fastest_cache_ties = tied_splits(
            cache_candidates, "median_ms", minimize=True
        )
        fastest_traffic, fastest_traffic_ties = tied_splits(
            traffic_candidates, "median_ms", minimize=True
        )
        min_misses, min_miss_ties = tied_splits(
            cache_candidates, "l3_miss_per_invocation", minimize=True
        )
        max_hit_rate, max_hit_rate_ties = tied_splits(
            cache_candidates, "l3_hit_percent", minimize=False
        )
        min_dram, min_dram_ties = tied_splits(
            traffic_candidates, "dram_bytes_per_invocation", minimize=True
        )
        max_ai, max_ai_ties = tied_splits(
            traffic_candidates, "measured_dram_ai_flops_per_byte", minimize=False
        )
        cache_common = sorted(
            set(fastest_cache_ties) & set(min_miss_ties) & set(max_hit_rate_ties)
        )
        traffic_common = sorted(
            set(fastest_traffic_ties) & set(min_dram_ties) & set(max_ai_ties)
        )
        checks.append(
            {
                "model": model,
                "batch": batch,
                "sequence": sequence,
                "rows": batch * sequence,
                "fastest_k_cache_pass": fastest_cache,
                "fastest_k_cache_pass_ties": fastest_cache_ties,
                "fastest_k_traffic_pass": fastest_traffic,
                "fastest_k_traffic_pass_ties": fastest_traffic_ties,
                "minimum_l3_miss_k": min_misses,
                "minimum_l3_miss_k_ties": min_miss_ties,
                "maximum_l3_hit_rate_k": max_hit_rate,
                "maximum_l3_hit_rate_k_ties": max_hit_rate_ties,
                "minimum_dram_byte_k": min_dram,
                "minimum_dram_byte_k_ties": min_dram_ties,
                "maximum_measured_dram_ai_k": max_ai,
                "maximum_measured_dram_ai_k_ties": max_ai_ties,
                "cache_pass_common_best_k": cache_common,
                "traffic_pass_common_best_k": traffic_common,
                "cache_pass_rank_agreement": bool(cache_common),
                "traffic_pass_rank_agreement": bool(traffic_common),
            }
        )
    return checks


def compare_k4_k8(rows: list[dict[str, Any]], pass_name: str) -> list[dict[str, Any]]:
    metrics = (
        ["median_ms", "l3_miss_per_invocation", "l3_hit_percent"]
        if pass_name == "cache"
        else [
            "median_ms",
            "dram_bytes_per_invocation",
            "measured_dram_ai_flops_per_byte",
        ]
    )
    comparisons = []
    shapes = sorted(
        {
            (row["model"], int(row["batch"]), int(row["sequence"]))
            for row in rows
        }
    )
    for model, batch, sequence in shapes:
        candidates = {
            (int(row["round"]), int(row["split"])): row
            for row in rows
            if row["model"] == model
            and int(row["batch"]) == batch
            and int(row["sequence"]) == sequence
            and row["backend"] == "imbps"
        }
        rounds = sorted(
            round_number
            for round_number, split in candidates
            if split == 4
        )
        for metric_index, metric in enumerate(metrics):
            differences = [
                float(candidates[(round_number, 4)][metric])
                - float(candidates[(round_number, 8)][metric])
                for round_number in rounds
            ]
            low, high = bootstrap_median_ci(
                differences, seed=20262003 + batch * 10 + metric_index
            )
            comparisons.append(
                {
                    "pass": pass_name,
                    "model": model,
                    "batch": batch,
                    "sequence": sequence,
                    "rows": batch * sequence,
                    "metric": metric,
                    "definition": "K4 minus K8",
                    "rounds": len(rounds),
                    "median_difference": statistics.median(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci95_excludes_zero": low > 0 or high < 0,
                }
            )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cache_rows = json.loads((args.cache_dir / "records.json").read_text())
    traffic_rows = json.loads((args.traffic_dir / "records.json").read_text())
    cache_summary = summarize_pass(cache_rows, "cache")
    traffic_summary = summarize_pass(traffic_rows, "traffic")
    checks = mechanism_checks(cache_summary, traffic_summary)
    pairwise = compare_k4_k8(cache_rows, "cache") + compare_k4_k8(
        traffic_rows, "traffic"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = cache_summary + traffic_summary
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "mechanism-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "mechanism-checks.csv", checks)
    (args.output_dir / "k4-vs-k8.json").write_text(
        json.dumps(pairwise, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "k4-vs-k8.csv", pairwise)
    print(json.dumps(checks, indent=2))
    print(f"uProf mechanism analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
