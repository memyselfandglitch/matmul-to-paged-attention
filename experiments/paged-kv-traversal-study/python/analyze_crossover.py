#!/usr/bin/env python3
"""Locate the HBND-to-BHND crossover on a block-fragmentation curve."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


LAYOUTS = ("BNHD", "BHND", "HBND")


def describe_result(preference: str, ratio: float, agreeing: int, trials: int) -> str:
    """Turn a paired timing result into a compact presentation statement."""
    if preference == "tie":
        return "Effective tie"

    magnitude = abs(ratio - 1.0)
    if agreeing == trials and magnitude >= 0.05:
        return f"{preference} clearly faster"
    if agreeing == trials:
        return f"{preference} faster"

    contrary = trials - agreeing
    if contrary == 1:
        return f"{preference} faster, one outlier"
    return f"{preference} faster, {contrary} contrary trials"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--tie-threshold-pct",
        type=float,
        default=2.0,
        help="treat BHND/HBND differences below this percentage as a tie",
    )
    args = parser.parse_args()
    if args.tie_threshold_pct < 0.0:
        parser.error("--tie-threshold-pct cannot be negative")
    tie_fraction = args.tie_threshold_pct / 100.0

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    bandwidths: dict[tuple[int, str], list[float]] = defaultdict(list)
    throughputs: dict[tuple[int, str], list[float]] = defaultdict(list)
    paired: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    num_runs: dict[int, int] = {}
    intensities: set[float] = set()
    with args.input.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no measurements in {args.input}")

    for row in rows:
        memory = row["memory_layout"]
        traversal = row["traversal"]
        if memory not in LAYOUTS or traversal != memory:
            raise ValueError("crossover input must contain matched cases only")
        if float(row["max_abs_error"]) != 0.0:
            raise ValueError("correctness error reported in crossover input")
        run_length = int(row["run_length"])
        trial = int(row["trial"])
        median_ms = float(row["median_ms"])
        grouped[(run_length, memory)].append(median_ms)
        bandwidths[(run_length, memory)].append(float(row["gib_per_second"]))
        throughputs[(run_length, memory)].append(float(row["gflops"]))
        paired[(run_length, trial)][memory] = median_ms
        num_runs[run_length] = int(row["num_runs"])
        intensities.add(float(row["useful_flops_per_kv_byte"]))

    run_lengths = sorted(num_runs, reverse=True)
    for run_length in run_lengths:
        for layout in LAYOUTS:
            if (run_length, layout) not in grouped:
                raise ValueError(f"missing {layout} at run length {run_length}")

    summary_rows: list[dict[str, str | int | float]] = []
    print("Matched-layout crossover versus contiguous physical-block run length")
    print(
        f"{'run':>7} {'runs':>7} {'BNHD ms':>11} {'BHND ms':>11} "
        f"{'HBND ms':>11} {'winner':>8} {'HBND/BHND':>12} {'pair':>7}"
    )
    previous_decisive_preference: str | None = None
    previous_decisive_run_length: int | None = None
    transitions: list[tuple[int, int, str, str]] = []
    for run_length in run_lengths:
        medians = {
            layout: statistics.median(grouped[(run_length, layout)])
            for layout in LAYOUTS
        }
        median_bandwidths = {
            layout: statistics.median(bandwidths[(run_length, layout)])
            for layout in LAYOUTS
        }
        median_throughputs = {
            layout: statistics.median(throughputs[(run_length, layout)])
            for layout in LAYOUTS
        }
        winner = min(medians, key=medians.get)
        trial_ratios = [
            measurements["HBND"] / measurements["BHND"]
            for (length, _), measurements in paired.items()
            if length == run_length and set(measurements) == set(LAYOUTS)
        ]
        if not trial_ratios:
            raise ValueError(f"no complete paired trials at run length {run_length}")
        ratio = medians["HBND"] / medians["BHND"]
        if ratio < 1.0 - tie_fraction:
            preference = "HBND"
        elif ratio > 1.0 + tie_fraction:
            preference = "BHND"
        else:
            preference = "tie"
        if preference != "tie":
            if (
                previous_decisive_preference is not None
                and preference != previous_decisive_preference
            ):
                assert previous_decisive_run_length is not None
                transitions.append(
                    (
                        previous_decisive_run_length,
                        run_length,
                        previous_decisive_preference,
                        preference,
                    )
                )
            previous_decisive_preference = preference
            previous_decisive_run_length = run_length
        hbnd_trial_wins = sum(trial_ratio < 1.0 for trial_ratio in trial_ratios)
        bhnd_trial_wins = sum(trial_ratio > 1.0 for trial_ratio in trial_ratios)
        if hbnd_trial_wins >= bhnd_trial_wins:
            trial_winner = "HBND"
            trial_wins = hbnd_trial_wins
        else:
            trial_winner = "BHND"
            trial_wins = bhnd_trial_wins
        trial_agreement = f"{trial_winner} {trial_wins}/{len(trial_ratios)}"
        agreeing_with_preference = (
            hbnd_trial_wins if preference == "HBND" else bhnd_trial_wins
        )
        interpretation = describe_result(
            preference, ratio, agreeing_with_preference, len(trial_ratios)
        )
        print(
            f"{run_length:>7} {num_runs[run_length]:>7} "
            f"{medians['BNHD']:>11.3f} {medians['BHND']:>11.3f} "
            f"{medians['HBND']:>11.3f} {winner:>8} {ratio:>11.3f}x "
            f"{preference:>7}"
        )
        summary_rows.append(
            {
                "run_length": run_length,
                "num_runs": num_runs[run_length],
                "bnhd_median_ms": medians["BNHD"],
                "bhnd_median_ms": medians["BHND"],
                "hbnd_median_ms": medians["HBND"],
                "bnhd_gib_per_second": median_bandwidths["BNHD"],
                "bhnd_gib_per_second": median_bandwidths["BHND"],
                "hbnd_gib_per_second": median_bandwidths["HBND"],
                "bnhd_gflops": median_throughputs["BNHD"],
                "bhnd_gflops": median_throughputs["BHND"],
                "hbnd_gflops": median_throughputs["HBND"],
                "winner": winner,
                "hbnd_over_bhnd": ratio,
                "pair_preference": preference,
                "trials": len(trial_ratios),
                "hbnd_trial_wins": hbnd_trial_wins,
                "bhnd_trial_wins": bhnd_trial_wins,
                "trial_agreement": trial_agreement,
                "interpretation": interpretation,
                "bhnd_min_ms": min(grouped[(run_length, "BHND")]),
                "bhnd_max_ms": max(grouped[(run_length, "BHND")]),
                "hbnd_min_ms": min(grouped[(run_length, "HBND")]),
                "hbnd_max_ms": max(grouped[(run_length, "HBND")]),
            }
        )

    print()
    print("Compact HBND/BHND crossover summary")
    print(
        f"{'run':>5} {'BHND GiB/s':>11} {'HBND GiB/s':>11} "
        f"{'BHND GF/s':>10} {'HBND GF/s':>10} {'HBND/BHND':>12} "
        f"{'trial agreement':>17}  interpretation"
    )
    for row in summary_rows:
        print(
            f"{int(row['run_length']):>5} "
            f"{float(row['bhnd_gib_per_second']):>11.3f} "
            f"{float(row['hbnd_gib_per_second']):>11.3f} "
            f"{float(row['bhnd_gflops']):>10.3f} "
            f"{float(row['hbnd_gflops']):>10.3f} "
            f"{float(row['hbnd_over_bhnd']):>11.3f}x "
            f"{str(row['trial_agreement']):>17}  {row['interpretation']}"
        )

    print()
    print(f"Useful arithmetic intensity: {sorted(intensities)} FLOP/KV byte")
    print(
        f"BHND/HBND differences within {args.tie_threshold_pct:.1f}% are reported "
        "as ties."
    )
    if transitions:
        for longer_run, shorter_run, before, after in transitions:
            print(
                "BHND/HBND preference changes between contiguous run lengths "
                f"{longer_run} and {shorter_run}: {before} -> {after}."
            )
    else:
        print("No BHND/HBND preference switch occurred in the tested run lengths.")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=summary_rows[0].keys(),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
