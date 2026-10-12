#!/usr/bin/env python3
"""Summarize normal batched generation across batch, context, and IMBPS K."""

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
        baseline_sample = [baseline[rng.randrange(len(baseline))] for _ in baseline]
        candidate_sample = [
            candidate[rng.randrange(len(candidate))] for _ in candidate
        ]
        ratios.append(
            statistics.median(baseline_sample)
            / statistics.median(candidate_sample)
        )
    return quantile(ratios, 0.025), quantile(ratios, 0.975)


def variant_key(result: dict[str, Any]) -> tuple[int, str, int]:
    return int(result["batch"]), result["backend"], int(result["split"])


def backend_key(variant: dict[str, Any]) -> tuple[str, int]:
    return variant["backend"], int(variant["split"])


def context_record(result: dict[str, Any], context: int) -> dict[str, Any]:
    return next(
        record for record in result["records"] if int(record["context_tokens"]) == context
    )


def round_values(record: dict[str, Any], path: tuple[str, ...]) -> list[float]:
    values = []
    for round_record in record["rounds"]:
        value: Any = round_record
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no result.json files below {root}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(result.get("status") != "complete" for result in results):
        raise ValueError("batched-decode matrix contains an incomplete result")
    if any(result.get("mode") != "normal" for result in results):
        raise ValueError("batched-decode summary accepts normal generation only")
    claims = {result["claim"] for result in results}
    if len(claims) != 1:
        raise ValueError(f"mixed claims: {sorted(claims)}")
    observed = {variant_key(result) for result in results}
    if len(observed) != len(results):
        raise ValueError("duplicate batch/backend/K result")
    expected_backend_sets = {
        tuple(
            sorted(
                backend_key(variant)
                for variant in result["expected_variants"]
                if variant["mode"] == "normal"
            )
        )
        for result in results
    }
    if len(expected_backend_sets) != 1:
        raise ValueError("results disagree about the backend/K matrix")
    expected_backends = set(next(iter(expected_backend_sets)))
    for batch in sorted({int(result["batch"]) for result in results}):
        batch_observed = {
            (result["backend"], int(result["split"]))
            for result in results
            if int(result["batch"]) == batch
        }
        if batch_observed != expected_backends:
            raise ValueError(
                f"incomplete batch={batch} matrix; "
                f"missing={sorted(expected_backends-batch_observed)} "
                f"extra={sorted(batch_observed-expected_backends)}"
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
    context_sets = {
        tuple(sorted(int(record["context_tokens"]) for record in result["records"]))
        for result in results
    }
    if len(context_sets) != 1:
        raise ValueError("context matrix differs across variants")
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {variant_key(result): result for result in results}
    batches = sorted({int(result["batch"]) for result in results})
    contexts = sorted(
        int(record["context_tokens"]) for record in results[0]["records"]
    )
    rows = []
    for batch in batches:
        for context in contexts:
            baseline = context_record(by_variant[(batch, "tpp", 1)], context)
            baseline_total = round_values(baseline, ("total_ms",))
            baseline_ttft = round_values(
                baseline, ("emission", "time_to_first_emission_ms")
            )
            baseline_tail = round_values(
                baseline, ("emission", "post_first_emission_ms")
            )
            for (candidate_batch, backend, split), result in sorted(by_variant.items()):
                if candidate_batch != batch:
                    continue
                record = context_record(result, context)
                total = round_values(record, ("total_ms",))
                ttft = round_values(
                    record, ("emission", "time_to_first_emission_ms")
                )
                tail = round_values(record, ("emission", "post_first_emission_ms"))
                seed = 20261012 + batch * 1009 + context * 17 + split
                total_low, total_high = ratio_ci(baseline_total, total, seed)
                ttft_low, ttft_high = ratio_ci(baseline_ttft, ttft, seed + 1)
                tail_low, tail_high = ratio_ci(baseline_tail, tail, seed + 2)
                output_tokens = int(record["rounds"][0]["requested_output_tokens"])
                tail_steps = max(1, output_tokens - 1)
                baseline_hashes = [
                    item["signature"]["generated_token_sha256"]
                    for item in baseline["rounds"]
                ]
                candidate_hashes = [
                    item["signature"]["generated_token_sha256"]
                    for item in record["rounds"]
                ]
                agreement = sum(
                    left == right
                    for left, right in zip(baseline_hashes, candidate_hashes)
                ) / len(candidate_hashes)
                l2_proxy = result["l2_equation_13_proxy"]
                rows.append(
                    {
                        "claim": result["claim"],
                        "batch": batch,
                        "context_tokens": context,
                        "requested_output_tokens_per_sequence": output_tokens,
                        "backend": backend,
                        "split": split,
                        "rounds": len(total),
                        "nominal_prompt_kv_mib": record["nominal_kv_cache"][
                            "prompt_mib"
                        ],
                        "nominal_final_kv_mib": record["nominal_kv_cache"][
                            "final_mib"
                        ],
                        "equation_13_strict_lower_bound": l2_proxy[
                            "strict_lower_bound"
                        ],
                        "equation_13_power_of_two_k": l2_proxy[
                            "next_power_of_two_candidate"
                        ],
                        "median_total_ms": statistics.median(total),
                        "median_ttft_ms": statistics.median(ttft),
                        "median_decode_tail_ms": statistics.median(tail),
                        "median_decode_step_ms": statistics.median(tail) / tail_steps,
                        "median_aggregate_decode_ms_per_token": (
                            statistics.median(tail) / (batch * tail_steps)
                        ),
                        "median_output_tokens_per_second": record["summary"][
                            "median_output_tokens_per_second"
                        ],
                        "median_post_first_tokens_per_second": record["summary"][
                            "median_post_first_tokens_per_second"
                        ],
                        "total_speedup_vs_tpp": (
                            statistics.median(baseline_total)
                            / statistics.median(total)
                        ),
                        "total_speedup_ci95_low": total_low,
                        "total_speedup_ci95_high": total_high,
                        "ttft_speedup_vs_tpp": (
                            statistics.median(baseline_ttft)
                            / statistics.median(ttft)
                        ),
                        "ttft_speedup_ci95_low": ttft_low,
                        "ttft_speedup_ci95_high": ttft_high,
                        "decode_tail_speedup_vs_tpp": (
                            statistics.median(baseline_tail) / statistics.median(tail)
                        ),
                        "decode_tail_speedup_ci95_low": tail_low,
                        "decode_tail_speedup_ci95_high": tail_high,
                        "generated_sequence_agreement_with_tpp": agreement,
                        "ci_method": "independent bootstrap of per-run medians",
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["batch"],
            row["context_tokens"],
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
        "experiment_kind": "batched_normal_autoregressive_generation",
        "claim": results[0]["claim"],
        "target_model": results[0]["model"]["target_requested_reference"],
        "primary_metric": "decode_tail_speedup_vs_tpp",
        "timing_boundary": results[0]["runner"]["timing_boundary"],
        "limitations": [
            "The Equation-13 K is an aggregate-private-L2 proxy, not a shared-cache guarantee.",
            "Qwen2.5 has a gated MLP, while Equation 13 is used in its paper-style two-GEMM form.",
            "Prompts are synthetic and every sequence in a batch has equal length; no padding is used.",
            "The offline batch is not a continuous-batching serving workload.",
            "The clean timing run does not collect hardware counters.",
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
