#!/usr/bin/env python3
"""Summarize global and parallel per-CCD IMBPS variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def ratio_ci(
    baseline: list[float], candidate: list[float], seed: int, draws: int = 10_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    for _ in range(draws):
        left = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        right = [candidate[rng.randrange(len(candidate))] for _ in candidate]
        ratios.append(statistics.median(left) / statistics.median(right))
    return quantile(ratios, 0.025), quantile(ratios, 0.975)


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no per-CCD result.json files below {root}")
    fragments = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(result.get("status") != "complete" for result in fragments):
        raise ValueError("per-CCD matrix contains an incomplete result")
    claims = {result["claim"] for result in fragments}
    if len(claims) != 1:
        raise ValueError(f"mixed claims: {sorted(claims)}")

    # Shape-isolated matrices intentionally produce one result fragment per
    # (variant, active_rows). Merge those fragments before applying the same
    # completeness and summary logic used by the original all-shapes process.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in fragments:
        grouped.setdefault(result["variant"]["variant_id"], []).append(result)
    results = []
    for variant_id, pieces in grouped.items():
        merged = dict(pieces[0])
        records = []
        for piece in pieces:
            if piece["variant"] != merged["variant"]:
                raise ValueError(f"variant metadata differs across {variant_id}")
            records.extend(piece["records"])
        active_rows = [int(record["active_rows"]) for record in records]
        if len(active_rows) != len(set(active_rows)):
            raise ValueError(f"duplicate active-row fragment for {variant_id}")
        merged["records"] = sorted(records, key=lambda row: int(row["active_rows"]))
        results.append(merged)

    observed = {result["variant"]["variant_id"] for result in results}
    expected_sets = {
        tuple(sorted(row["variant_id"] for row in result["expected_variants"]))
        for result in results
    }
    if len(expected_sets) != 1:
        raise ValueError("results disagree about expected variants")
    expected = set(next(iter(expected_sets)))
    if observed != expected:
        raise ValueError(
            f"incomplete per-CCD matrix; missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    row_sets = {
        tuple(sorted(int(record["active_rows"]) for record in result["records"]))
        for result in results
    }
    if len(row_sets) != 1:
        raise ValueError("active-row matrix differs across variants")
    return results


def record_for(result: dict[str, Any], active_rows: int) -> dict[str, Any]:
    return next(
        record for record in result["records"] if record["active_rows"] == active_rows
    )


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {result["variant"]["variant_id"]: result for result in results}
    by_mode_k = {
        (result["variant"]["mode"], int(result["variant"]["total_splits"])): result
        for result in results
        if result["variant"]["mode"] in {"tpp", "global_imbps"}
    }
    active_rows_values = sorted(
        int(record["active_rows"]) for record in results[0]["records"]
    )
    rows = []
    for active_rows in active_rows_values:
        tpp_record = record_for(by_id["tpp-all96"], active_rows)
        tpp_times = [float(value) for value in tpp_record["durations_ms"]]
        for result in sorted(results, key=lambda item: item["variant"]["variant_id"]):
            variant = result["variant"]
            record = record_for(result, active_rows)
            times = [float(value) for value in record["durations_ms"]]
            speedup_tpp = statistics.median(tpp_times) / statistics.median(times)
            low, high = ratio_ci(
                tpp_times,
                times,
                seed=20261011 + active_rows * 31 + int(variant["total_splits"]),
            )
            global_same_k = by_mode_k.get(
                ("global_imbps", int(variant["total_splits"]))
            )
            speedup_same_k = None
            if variant["mode"] in {"per_ccd", "per_ccd_threads"} and global_same_k is not None:
                global_record = record_for(global_same_k, active_rows)
                speedup_same_k = statistics.median(
                    [float(value) for value in global_record["durations_ms"]]
                ) / statistics.median(times)
            scaling_speedup = None
            scaling_efficiency = None
            if variant["mode"] in {"per_ccd", "per_ccd_threads"}:
                serial_candidates = [
                    item
                    for item in results
                    if item["variant"]["mode"] == variant["mode"]
                    and int(item["variant"]["ccd_count"]) == 1
                ]
                if len(serial_candidates) > 1:
                    raise ValueError(
                        f"expected at most one single-CCD baseline for {variant['mode']}"
                    )
                if serial_candidates:
                    serial_record = record_for(serial_candidates[0], active_rows)
                    serial_times = [
                        float(value) for value in serial_record["durations_ms"]
                    ]
                    scaling_speedup = statistics.median(
                        serial_times
                    ) / statistics.median(times)
                    scaling_efficiency = scaling_speedup / int(variant["ccd_count"])
            numerics = record["numerics_vs_global_same_k"] or {}
            cache_model = record["cache_model_per_ccd"]
            rows.append(
                {
                    "claim": result["claim"],
                    "active_rows": active_rows,
                    "variant_id": variant["variant_id"],
                    "mode": variant["mode"],
                    "ccd_count": variant["ccd_count"],
                    "cores_used": (
                        int(variant["ccd_count"])
                        * int(result["topology"]["cores_per_ccd"])
                    ),
                    "total_splits": variant["total_splits"],
                    "inner_threads": variant.get("inner_threads"),
                    "reduction_strategy": variant.get("reduction_strategy"),
                    "splits_per_ccd": (
                        int(variant["total_splits"]) / int(variant["ccd_count"])
                    ),
                    "median_ms": statistics.median(times),
                    "q1_ms": record["q1_ms"],
                    "q3_ms": record["q3_ms"],
                    "speedup_vs_tpp_all96": speedup_tpp,
                    "speedup_vs_tpp_all96_ci95_low": low,
                    "speedup_vs_tpp_all96_ci95_high": high,
                    "speedup_vs_global_imbps_same_k": speedup_same_k,
                    "parallel_speedup_vs_one_ccd": scaling_speedup,
                    "parallel_efficiency_vs_one_ccd": scaling_efficiency,
                    "modeled_per_ccd_working_set_mib": cache_model[
                        "modeled_working_set_mib"
                    ],
                    "fits_32_mib_per_ccd_strict": cache_model[
                        "fits_one_ccd_strict"
                    ],
                    "max_abs_vs_global_same_k": numerics.get("max_abs"),
                    "q99_abs_vs_global_same_k": numerics.get("q99_abs"),
                    "argmax_agreement_vs_global_same_k": numerics.get(
                        "output_coordinate_argmax_agreement"
                    ),
                    "timing_includes_cross_ccd_reduction": variant["mode"]
                    in {"per_ccd", "per_ccd_threads"},
                    "median_compute_and_sync_ms": record.get(
                        "median_compute_and_sync_ms"
                    ),
                    "median_reduction_and_conversion_ms": record.get(
                        "median_reduction_and_conversion_ms"
                    ),
                    "median_worker_operator_min_ms": record.get(
                        "median_worker_operator_min_ms"
                    ),
                    "median_worker_operator_max_ms": record.get(
                        "median_worker_operator_max_ms"
                    ),
                    "median_worker_operator_imbalance": record.get(
                        "median_worker_operator_imbalance"
                    ),
                    "median_worker_publication_max_ms": record.get(
                        "median_worker_publication_max_ms"
                    ),
                    "median_worker_reduction_max_ms": record.get(
                        "median_worker_reduction_max_ms"
                    ),
                    "reduction_fraction_of_latency": (
                        float(record["median_reduction_and_conversion_ms"])
                        / statistics.median(times)
                        if record.get("median_reduction_and_conversion_ms")
                        is not None
                        else None
                    ),
                    "ci_method": "independent bootstrap of iteration latencies",
                }
            )
    return sorted(rows, key=lambda row: (row["active_rows"], row["variant_id"]))


def write_outputs(
    root: Path, results: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> Path:
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment_kind": "parallel_per_ccd_imbps",
        "claim": results[0]["claim"],
        "scope_note": results[0]["scope_note"],
        "limitations": [
            "The matrix may contain process-based and persistent-pthread prototypes of the PACE v1.0 split-loop TODO; neither patches the PACE kernel.",
            "Per-CCD latency includes synchronization, partial-output publication, and final reduction, but excludes worker/thread and weight setup.",
            "The 32 MiB fit calculation applies to one active split block plus input and intermediate activation on one CCD.",
            "TPP/global and per-CCD variants run in separate serialized Slurm tasks; confidence intervals therefore use independent samples.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    results = load_results(args.result_root)
    rows = summarize(results)
    print(write_outputs(args.result_root, results, rows))


if __name__ == "__main__":
    main()
