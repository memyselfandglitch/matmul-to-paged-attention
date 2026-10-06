#!/usr/bin/env python3
"""Summarize serialized PACE KV-cache target-verification backend runs."""

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


def independent_speedup_ci(
    baseline_rounds: list[float],
    candidate_rounds: list[float],
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float]:
    """Bootstrap median(base)/median(candidate) without claiming pairing."""
    if not baseline_rounds or not candidate_rounds:
        raise ValueError("round medians cannot be empty")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        baseline = [
            baseline_rounds[rng.randrange(len(baseline_rounds))]
            for _ in baseline_rounds
        ]
        candidate = [
            candidate_rounds[rng.randrange(len(candidate_rounds))]
            for _ in candidate_rounds
        ]
        estimates.append(statistics.median(baseline) / statistics.median(candidate))
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def correctness_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[float, float]:
    baseline_top1 = baseline["top1_token_ids"]
    candidate_top1 = candidate["top1_token_ids"]
    candidate_top5 = candidate["top5_token_ids"]
    if not (len(baseline_top1) == len(candidate_top1) == len(candidate_top5)):
        raise ValueError("numeric signatures have incompatible lengths")
    count = len(baseline_top1)
    if count == 0:
        raise ValueError("numeric signature is empty")
    top1_agreement = sum(
        expected == observed
        for expected, observed in zip(baseline_top1, candidate_top1)
    ) / count
    baseline_in_top5 = sum(
        expected in observed
        for expected, observed in zip(baseline_top1, candidate_top5)
    ) / count
    return top1_agreement, baseline_in_top5


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no backend result.json files under {root}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    incomplete = [
        str(path)
        for path, result in zip(paths, results)
        if result.get("status") != "complete"
    ]
    if incomplete:
        raise ValueError(f"incomplete target-verification outputs: {incomplete}")
    claims = {result["claim"] for result in results}
    if len(claims) != 1:
        raise ValueError(f"mixed claims in one summary: {sorted(claims)}")
    variants = {(result["backend"], int(result["split"])) for result in results}
    if len(variants) != len(results):
        raise ValueError("duplicate backend/split result")
    expected_sets = {
        tuple(
            sorted(
                (variant["backend"], int(variant["split"]))
                for variant in result["expected_variants"]
            )
        )
        for result in results
    }
    if len(expected_sets) != 1:
        raise ValueError("backend outputs disagree about the expected variants")
    expected = set(next(iter(expected_sets)))
    if variants != expected:
        missing = sorted(expected - variants)
        extra = sorted(variants - expected)
        raise ValueError(
            f"backend matrix is incomplete or unexpected; missing={missing}, extra={extra}"
        )

    model_fingerprints = {
        (
            result["model"].get("name"),
            result["model"].get("requested_reference"),
            result["model"].get("resolved_path"),
            result["model"].get("snapshot_commit"),
        )
        for result in results
    }
    if len(model_fingerprints) != 1:
        raise ValueError("backend outputs were produced from different model artifacts")

    case_sets = {
        tuple(sorted(record["case"]["case_id"] for record in result["records"]))
        for result in results
    }
    if len(case_sets) != 1:
        raise ValueError("backend outputs do not contain the same case matrix")
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for result in results:
        for record in result["records"]:
            key = (record["case"]["case_id"], result["backend"], int(result["split"]))
            if key in records:
                raise ValueError(f"duplicate case/backend record: {key}")
            records[key] = {"result": result, "record": record}

    case_ids = sorted({key[0] for key in records})
    variants = sorted(
        {(key[1], key[2]) for key in records},
        key=lambda item: (item[0] != "tpp", item[1]),
    )
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        missing = [
            (backend, split)
            for backend, split in variants
            if (case_id, backend, split) not in records
        ]
        if missing:
            raise ValueError(f"case {case_id} is missing variants: {missing}")
        baseline_entry = records[(case_id, "tpp", 1)]
        baseline_record = baseline_entry["record"]
        baseline_rounds = [
            float(round_record["median_ms"])
            for round_record in baseline_record["rounds"]
        ]
        baseline_median = statistics.median(baseline_rounds)
        available = [
            (backend, split, records[(case_id, backend, split)])
            for backend, split in variants
            if (case_id, backend, split) in records
        ]
        fastest = min(
            available,
            key=lambda item: statistics.median(
                [float(row["median_ms"]) for row in item[2]["record"]["rounds"]]
            ),
        )
        for backend, split, entry in available:
            record = entry["record"]
            round_medians = [
                float(round_record["median_ms"])
                for round_record in record["rounds"]
            ]
            median_ms = statistics.median(round_medians)
            if backend == "tpp":
                ci_low = ci_high = speedup = 1.0
                top1_agreement = baseline_in_top5 = 1.0
            else:
                speedup = baseline_median / median_ms
                ci_low, ci_high = independent_speedup_ci(
                    baseline_rounds,
                    round_medians,
                    seed=20251001 + record["case"]["active_rows"] * 37 + split,
                )
                top1_agreement, baseline_in_top5 = correctness_metrics(
                    baseline_record["numerics"], record["numerics"]
                )
            case = record["case"]
            rows.append(
                {
                    "claim": entry["result"]["claim"],
                    "case_id": case_id,
                    "mode": case["mode"],
                    "batch": case["batch"],
                    "draft_tokens": case["draft_tokens"],
                    "active_rows": case["active_rows"],
                    "context_tokens": case["context_tokens"],
                    "backend": backend,
                    "split": split,
                    "rounds": len(round_medians),
                    "iterations": record["run"]["iterations"],
                    "median_ms": median_ms,
                    "q1_ms": record["run"]["q1_ms"],
                    "q3_ms": record["run"]["q3_ms"],
                    "median_rollback_ms": record["run"]["median_rollback_ms"],
                    "active_rows_per_second": case["active_rows"] / (median_ms / 1000),
                    "baseline_median_ms": baseline_median,
                    "speedup_vs_tpp": speedup,
                    "speedup_ci95_low": ci_low,
                    "speedup_ci95_high": ci_high,
                    "speedup_supports_faster": ci_low > 1.0,
                    "top1_agreement": top1_agreement,
                    "baseline_top1_in_candidate_top5": baseline_in_top5,
                    "nominal_kv_cache_gib": record["cache"]["nominal_bytes"]
                    / (1024**3),
                    "allocated_kv_cache_gib": record["cache"]["allocated_bytes"]
                    / (1024**3),
                    "prefill_ms": record["cache"]["prefill_ms"],
                    "empirical_best_backend": fastest[0],
                    "empirical_best_split": fastest[1],
                    "is_empirical_best": backend == fastest[0] and split == fastest[1],
                    "ci_method": "independent bootstrap of per-round medians",
                }
            )
    return sorted(rows, key=lambda row: (row["active_rows"], row["case_id"], row["split"]))


def write_outputs(root: Path, results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Path:
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "experiment_kind": "kv_cache_target_verification",
        "scope_note": results[0]["scope_note"],
        "claim": results[0]["claim"],
        "model": results[0]["model"],
        "backend_results": [
            {
                "backend": result["backend"],
                "split": result["split"],
                "model_load_seconds": result["runner"]["model_load_seconds"],
                "max_rss_kib": result["environment"]["max_rss_kib"],
            }
            for result in sorted(results, key=lambda item: int(item["split"]))
        ],
        "limitations": [
            "Draft-model generation and acceptance are excluded.",
            (
                "Prefill is performed to populate the KV cache but excluded "
                "from target-forward latency."
            ),
            (
                "Backend tasks load the model independently, so confidence "
                "intervals use an independent rather than paired bootstrap."
            ),
            (
                "This pilot uses one model load per backend; a confirmatory "
                "run should repeat backend blocks."
            ),
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
    output = write_outputs(args.result_root, results, rows)
    print(output)


if __name__ == "__main__":
    main()
