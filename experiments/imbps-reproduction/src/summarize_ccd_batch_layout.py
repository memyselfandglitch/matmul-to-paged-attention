#!/usr/bin/env python3
"""Compare whole-item and intermediate-channel placement on matched batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

try:
    from .per_ccd_mlp_bench import load_claim, validate_batch_shape
    from .run_standalone_matrix import bootstrap_median_ci, quantile
except ImportError:
    from per_ccd_mlp_bench import load_claim, validate_batch_shape
    from run_standalone_matrix import bootstrap_median_ci, quantile


CLAIM = "opt30b_per_ccd_batch_layout"
ITEM = "batch-items-t8"
CHANNEL = "thread-p12-k12-sharded-t8"
TPP = "tpp-all96"
GLOBAL = "global-k6-all96"
MARGIN = 1.02


def selected_shapes(claim: dict[str, Any], shapes: str) -> list[tuple[int, int]]:
    if not shapes:
        return list(product(claim["batches"], claim["rows_per_item"]))
    selected = []
    for shape in shapes.split("+"):
        try:
            b, q = map(int, shape.split(":"))
        except ValueError as error:
            raise ValueError("expected selected shapes in B:Q+B:Q format") from error
        validate_batch_shape(claim, b, b * q)
        selected.append((b, q))
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate selected shapes")
    return selected


def comparison(baseline: list[float], candidate: list[float]) -> tuple[float, float, float]:
    if len(baseline) != len(candidate):
        raise ValueError("latency samples must be paired")
    ratios = [left / right for left, right in zip(baseline, candidate)]
    return statistics.median(ratios), *bootstrap_median_ci(ratios)


def summarize_shape(
    pairs: list[dict[str, dict[str, Any]]],
    batch: int,
    rows_per_item: int,
    ccds: int,
) -> dict[str, Any]:
    names = {ITEM: "whole_item", CHANNEL: "channel", TPP: "tpp", GLOBAL: "global_k6"}
    times = {variant: [pair[variant]["median_ms"] for pair in pairs] for variant in names}
    speed, low, high = comparison(times[ITEM], times[CHANNEL])
    if low > MARGIN:
        winner = "channels"
        faster_ccd = CHANNEL
    elif high < 1 / MARGIN:
        winner = "whole_item"
        faster_ccd = ITEM
    else:
        winner = "unclear"
        faster_ccd = None
    m = batch * rows_per_item
    if rows_per_item == 1:
        workload = "decode_shaped"
    elif rows_per_item <= 16:
        workload = "verification_shaped"
    else:
        workload = "prefill_shaped"
    row = {
        "batch_items": batch,
        "rows_per_item": rows_per_item,
        "active_rows": m,
        "workload_shape": workload,
        "pairs": len(pairs),
        "whole_item_ccds": min(batch, ccds),
        "whole_item_max_items_per_ccd": math.ceil(batch / ccds),
        "channel_ccds": ccds,
        "channel_reduction_ms": statistics.median(
            pair[CHANNEL]["median_reduction_and_conversion_ms"] for pair in pairs
        ),
        "channel_speedup_vs_whole_item": speed,
        "channel_speedup_ci95_low": low,
        "channel_speedup_ci95_high": high,
        "timing_winner": winner,
    }
    for variant, name in names.items():
        latency = statistics.median(times[variant])
        row[f"{name}_ms"] = latency
        row[f"{name}_rows_per_second"] = 1000 * m / latency
        row[f"{name}_p95_ms"] = statistics.median(
            quantile(pair[variant]["durations_ms"], 0.95) for pair in pairs
        )
    row["fastest_measured"] = min(names.values(), key=lambda name: row[f"{name}_ms"])

    preferred = None
    for variant in (ITEM, CHANNEL):
        name = names[variant]
        numerics = [pair[variant]["numerics_vs_unsplit"] for pair in pairs]
        row[f"{name}_bitwise_equal_to_unsplit"] = all(
            entry["bitwise_equal"] for entry in numerics
        )
        row[f"{name}_max_abs_vs_unsplit"] = max(entry["max_abs"] for entry in numerics)
        row[f"{name}_worst_q99_abs_vs_unsplit"] = max(entry["q99_abs"] for entry in numerics)
        baseline_wins = []
        for baseline in (TPP, GLOBAL):
            metric = f"{name}_speedup_vs_{names[baseline]}"
            ratio, lower, upper = comparison(times[baseline], times[variant])
            row[metric] = ratio
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
            baseline_wins.append(lower > MARGIN)
        if faster_ccd == variant and all(baseline_wins):
            preferred = name
    row["preferred_ccd_layout"] = preferred or "unclear_or_socket_wide"
    return row


def switch_brackets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report observed switches, including reversals, along both axes."""
    brackets = []
    for fixed, sweep in (("rows_per_item", "batch_items"), ("batch_items", "rows_per_item")):
        for value in sorted({row[fixed] for row in rows}):
            decisive = sorted(
                (row[sweep], row["timing_winner"])
                for row in rows
                if row[fixed] == value and row["timing_winner"] != "unclear"
            )
            switches = [
                (left, right)
                for left, right in zip(decisive, decisive[1:])
                if left[1] != right[1]
            ]
            for (lower, lower_winner), (upper, upper_winner) in switches:
                brackets.append(
                    {
                        fixed: value,
                        "sweep": sweep,
                        "lower": lower,
                        "upper": upper,
                        "lower_winner": lower_winner,
                        "upper_winner": upper_winner,
                        "switches_in_slice": len(switches),
                    }
                )
    return brackets


def equal_row_reversals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rows = defaultdict(list)
    for row in rows:
        if row["timing_winner"] != "unclear":
            by_rows[row["active_rows"]].append(
                {"B": row["batch_items"], "Q": row["rows_per_item"],
                 "winner": row["timing_winner"]}
            )
    return [
        {"active_rows": active_rows, "shapes": matched}
        for active_rows, matched in sorted(by_rows.items())
        if len({shape["winner"] for shape in matched}) > 1
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shape-index", type=int)
    mode.add_argument("--shape-count", action="store_true")
    mode.add_argument("--result-root", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--shapes", default="", help="optional B:Q+B:Q subset or retest shapes")
    args = parser.parse_args()
    _, claim = load_claim(CLAIM)
    try:
        shapes = selected_shapes(claim, args.shapes)
    except ValueError as error:
        parser.error(str(error))
    if args.shape_count:
        print(len(shapes))
        return
    if args.shape_index is not None:
        if not 0 <= args.shape_index < len(shapes):
            parser.error("shape index is outside the configured sweep")
        print(*shapes[args.shape_index])
        return
    if args.pairs < 3:
        parser.error("at least three paired rounds are required")
    root = args.result_root
    rows = []
    for batch, q in shapes:
        case_dir = root / f"b{batch}-q{q}"
        pairs = []
        for index in range(1, args.pairs + 1):
            pair = {}
            for variant in (ITEM, CHANNEL, TPP, GLOBAL):
                path = case_dir / f"pair-{index}-{variant}.json"
                payload = json.loads(path.read_text())
                record, = payload["records"]
                if (
                    payload["status"] != "complete"
                    or payload["claim"] != CLAIM
                    or payload["variant"]["variant_id"] != variant
                    or (record["batch_items"], record["rows_per_item"]) != (batch, q)
                ):
                    raise ValueError(f"inconsistent batch-layout record: {path}")
                pair[variant] = record
            pairs.append(pair)
        rows.append(summarize_shape(pairs, batch, q, claim["expected_ccds"]))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    policy = {
        "status": "exploratory",
        "decision_speedup_ratio": MARGIN,
        "metric": "paired median batch-latency speedup with 95% bootstrap interval",
        "layout_switch_brackets": switch_brackets(rows),
        "equal_row_reversals": equal_row_reversals(rows),
        "holdout_required": "Retest with new paired rounds and intermediate shapes before adopting a crossover.",
        "numerics": "Check BF16 reference errors before selecting a policy. If bitwise identity is required, exclude layouts that fail the bitwise check.",
    }
    (root / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(root / "summary.csv")
    print(root / "policy.json")


if __name__ == "__main__":
    main()
