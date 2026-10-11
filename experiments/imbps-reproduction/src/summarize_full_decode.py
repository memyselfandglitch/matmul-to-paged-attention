#!/usr/bin/env python3
"""Combine complete PACE normal/PARD generation variants."""

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
    values = []
    for _ in range(draws):
        base_sample = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        candidate_sample = [
            candidate[rng.randrange(len(candidate))] for _ in candidate
        ]
        values.append(
            statistics.median(base_sample) / statistics.median(candidate_sample)
        )
    return quantile(values, 0.025), quantile(values, 0.975)


def variant_key(result: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        result["mode"],
        int(result["draft_tokens"]),
        result["backend"],
        int(result["split"]),
    )


def expected_key(variant: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        variant["mode"],
        int(variant["draft_tokens"]),
        variant["backend"],
        int(variant["split"]),
    )


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no full-decode result.json files below {root}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(result.get("status") != "complete" for result in results):
        raise ValueError("full-decode result matrix contains an incomplete result")
    claims = {result["claim"] for result in results}
    if len(claims) != 1:
        raise ValueError(f"mixed claims: {sorted(claims)}")
    observed = {variant_key(result) for result in results}
    if len(observed) != len(results):
        raise ValueError("duplicate full-decode variant")
    expected_sets = {
        tuple(sorted(expected_key(variant) for variant in result["expected_variants"]))
        for result in results
    }
    if len(expected_sets) != 1:
        raise ValueError("results disagree about the preregistered matrix")
    expected = set(next(iter(expected_sets)))
    if observed != expected:
        raise ValueError(
            f"incomplete full-decode matrix; missing={sorted(expected-observed)} "
            f"extra={sorted(observed-expected)}"
        )
    target_fingerprints = {
        (
            result["model"]["target_resolved_path"],
            result["model"]["target_snapshot_commit"],
        )
        for result in results
    }
    if len(target_fingerprints) != 1:
        raise ValueError("target model differs across variants")
    draft_fingerprints = {
        (
            result["model"]["draft_resolved_path"],
            result["model"]["draft_snapshot_commit"],
        )
        for result in results
        if result["mode"] == "pard"
    }
    if len(draft_fingerprints) != 1:
        raise ValueError("draft model differs across PARD variants")
    context_sets = {
        tuple(sorted(int(record["context_tokens"]) for record in result["records"]))
        for result in results
    }
    if len(context_sets) != 1:
        raise ValueError("context matrix differs across variants")
    return results


def context_record(result: dict[str, Any], context: int) -> dict[str, Any]:
    return next(
        record for record in result["records"] if record["context_tokens"] == context
    )


def round_values(record: dict[str, Any], field: str) -> list[float]:
    return [float(round_record[field]) for round_record in record["rounds"]]


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {variant_key(result): result for result in results}
    contexts = sorted(
        int(record["context_tokens"]) for record in results[0]["records"]
    )
    rows = []
    for context in contexts:
        absolute_baseline = context_record(
            by_variant[("normal", 0, "tpp", 1)], context
        )
        absolute_times = round_values(absolute_baseline, "total_ms")
        for key, result in sorted(by_variant.items()):
            mode, draft_tokens, backend, split = key
            record = context_record(result, context)
            times = round_values(record, "total_ms")
            same_mode_baseline = context_record(
                by_variant[(mode, draft_tokens, "tpp", 1)], context
            )
            same_mode_times = round_values(same_mode_baseline, "total_ms")
            normal_same_backend = context_record(
                by_variant[("normal", 0, backend, split)], context
            )
            normal_same_backend_times = round_values(normal_same_backend, "total_ms")
            total_speedup = statistics.median(absolute_times) / statistics.median(times)
            total_low, total_high = ratio_ci(
                absolute_times,
                times,
                seed=20261011 + context * 13 + draft_tokens * 17 + split,
            )
            mode_speedup = statistics.median(same_mode_times) / statistics.median(times)
            mode_low, mode_high = ratio_ci(
                same_mode_times,
                times,
                seed=20261012 + context * 19 + draft_tokens * 23 + split,
            )
            speculative_speedup = (
                statistics.median(normal_same_backend_times) / statistics.median(times)
                if mode == "pard"
                else 1.0
            )
            accepted = [
                round_record["accepted_draft_fraction"]
                for round_record in record["rounds"]
                if round_record["accepted_draft_fraction"] is not None
            ]
            kept = [
                round_record["mean_kept_tokens_per_speculative_iteration"]
                for round_record in record["rounds"]
                if round_record["mean_kept_tokens_per_speculative_iteration"]
                is not None
            ]
            baseline_hashes = [
                round_record["signature"]["generated_token_sha256"]
                for round_record in same_mode_baseline["rounds"]
            ]
            candidate_hashes = [
                round_record["signature"]["generated_token_sha256"]
                for round_record in record["rounds"]
            ]
            sequence_agreement = sum(
                left == right for left, right in zip(baseline_hashes, candidate_hashes)
            ) / len(candidate_hashes)
            rows.append(
                {
                    "claim": result["claim"],
                    "context_tokens": context,
                    "requested_output_tokens": record["rounds"][0][
                        "requested_output_tokens"
                    ],
                    "mode": mode,
                    "draft_tokens": draft_tokens,
                    "backend": backend,
                    "split": split,
                    "rounds": len(times),
                    "median_total_ms": statistics.median(times),
                    "median_ttft_ms": record["summary"]["median_ttft_ms"],
                    "median_output_tokens_per_second": record["summary"][
                        "median_output_tokens_per_second"
                    ],
                    "speedup_vs_normal_tpp": total_speedup,
                    "speedup_vs_normal_tpp_ci95_low": total_low,
                    "speedup_vs_normal_tpp_ci95_high": total_high,
                    "speedup_vs_same_mode_tpp": mode_speedup,
                    "speedup_vs_same_mode_tpp_ci95_low": mode_low,
                    "speedup_vs_same_mode_tpp_ci95_high": mode_high,
                    "pard_speedup_vs_normal_same_backend": speculative_speedup,
                    "median_accepted_draft_fraction": (
                        statistics.median(accepted) if accepted else None
                    ),
                    "median_kept_tokens_per_speculative_iteration": (
                        statistics.median(kept) if kept else None
                    ),
                    "generated_sequence_agreement_with_same_mode_tpp": sequence_agreement,
                    "ci_method": "independent bootstrap of per-run medians",
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["context_tokens"],
            row["mode"] != "normal",
            row["draft_tokens"],
            row["backend"] != "tpp",
            row["split"],
        ),
    )


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
        "experiment_kind": "full_autoregressive_and_pard_generation",
        "claim": results[0]["claim"],
        "target_model": results[0]["model"]["target_requested_reference"],
        "timing_boundary": results[0]["runner"]["timing_boundary"],
        "limitations": [
            "PACE v1.0 PARD supports greedy batch-one generation only.",
            "The same OperatorConfig, including the MLP backend and K, is applied to both target and PARD draft models.",
            "Each array task loads its model independently, so intervals use an independent bootstrap.",
            "The timed generate call includes prompt prefill; TTFT and total output throughput are reported separately.",
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
