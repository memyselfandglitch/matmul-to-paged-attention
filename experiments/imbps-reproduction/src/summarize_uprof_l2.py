#!/usr/bin/env python3
"""Summarize an L2-only AMD uProf matrix without L3 or DF counters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .summarize_uprof_matrix import summarize_pass, write_csv
except ImportError:
    from summarize_uprof_matrix import summarize_pass, write_csv


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


def l2_checks(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks = []
    shapes = sorted(
        {
            (row["model"], int(row["batch"]), int(row["sequence"]))
            for row in summary
        }
    )
    for model, batch, sequence in shapes:
        candidates = [
            row
            for row in summary
            if row["model"] == model
            and int(row["batch"]) == batch
            and int(row["sequence"]) == sequence
            and row["backend"] == "imbps"
        ]
        fastest, fastest_ties = tied_splits(
            candidates, "median_ms", minimize=True
        )
        minimum_misses, minimum_miss_ties = tied_splits(
            candidates, "l2_miss_per_invocation", minimize=True
        )
        maximum_hit_rate, maximum_hit_rate_ties = tied_splits(
            candidates, "l2_hit_percent", minimize=False
        )
        common = sorted(
            set(fastest_ties)
            & set(minimum_miss_ties)
            & set(maximum_hit_rate_ties)
        )
        equation_k = candidates[0]["equation13_author_power_of_two_candidate"]
        checks.append(
            {
                "model": model,
                "batch": batch,
                "sequence": sequence,
                "rows": batch * sequence,
                "cache_level": "l2",
                "cache_scope": candidates[0]["cache_scope"],
                "equation13_author_power_of_two_candidate": equation_k,
                "fastest_profiled_k": fastest,
                "fastest_profiled_k_ties": fastest_ties,
                "minimum_l2_miss_k": minimum_misses,
                "minimum_l2_miss_k_ties": minimum_miss_ties,
                "maximum_l2_hit_rate_k": maximum_hit_rate,
                "maximum_l2_hit_rate_k_ties": maximum_hit_rate_ties,
                "timing_and_l2_common_best_k": common,
                "timing_and_l2_rank_agreement": bool(common),
                "equation_candidate_is_l2_best_or_tied": (
                    equation_k in minimum_miss_ties
                    and equation_k in maximum_hit_rate_ties
                ),
                "timing_note": "uProf-instrumented; use job 9974 for clean timing",
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads((args.l2_dir / "records.json").read_text(encoding="utf-8"))
    summary = summarize_pass(rows, "l2")
    checks = l2_checks(summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "mechanism-checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "mechanism-checks.csv", checks)
    print(json.dumps(checks, indent=2))
    print(f"uProf L2-only analysis: {args.output_dir}")


if __name__ == "__main__":
    main()
