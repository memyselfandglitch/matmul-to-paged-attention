#!/usr/bin/env python3
"""Run Table II or VIII as isolated, randomized PACE processes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
WORKER_PATH = PROJECT_ROOT / "src" / "pace_mlp_bench.py"


def comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claim",
        choices=("table_ii", "table_viii", "decode_exploratory"),
        required=True,
    )
    parser.add_argument("--models", type=comma_strings, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20251001)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--case-timeout-seconds", type=float, default=7200)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument(
        "--finalize-existing",
        type=Path,
        default=None,
        help="rebuild summary files from a completed result directory without rerunning cases",
    )
    args = parser.parse_args()
    if min(args.rounds, args.iterations) <= 0 or args.warmups < 0:
        parser.error("rounds/iterations must be positive and warmups nonnegative")
    return args


def load_claim(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claim = registry["standalone"][args.claim]
    selected_models = args.models or claim["models"]
    unknown = sorted(set(selected_models) - set(claim["models"]))
    if unknown:
        raise ValueError(f"models not in {args.claim}: {', '.join(unknown)}")
    claim = dict(claim)
    claim["models"] = selected_models
    return registry, claim


def build_cases(registry: dict[str, Any], claim: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    for model_name in claim["models"]:
        model = registry["models"][model_name]
        for batch in claim["batches"]:
            cases.append(
                {
                    "model": model_name,
                    "backend": "tpp",
                    "split": 1,
                    "batch": batch,
                    "sequence": claim["sequence"],
                    "hidden": model["hidden"],
                    "intermediate": model["intermediate"],
                    "activation": model["activation"],
                    "dtype": claim["dtype"],
                }
            )
            for split in claim["splits"]:
                cases.append(
                    {
                        "model": model_name,
                        "backend": "imbps",
                        "split": split,
                        "batch": batch,
                        "sequence": claim["sequence"],
                        "hidden": model["hidden"],
                        "intermediate": model["intermediate"],
                        "activation": model["activation"],
                        "dtype": claim["dtype"],
                    }
                )
    return cases


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
    values: list[float], seed: int = 20251001, draws: int = 10_000
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


def target_for(
    claim_name: str,
    claim: dict[str, Any],
    model: str,
    batch: int,
    split: int,
) -> dict[str, Any]:
    if claim_name == "table_ii":
        target = claim["targets"].get(str(batch), {})
        return target if target.get("split") == split else {}
    if claim_name == "table_viii":
        return claim["targets"].get(model, {}).get(str(split), {})
    return {}


def summarize(
    records: list[dict[str, Any]], claim_name: str, claim: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, int], list[float]] = defaultdict(list)
    per_round: dict[tuple[str, int, str, str, int, int], float] = {}
    for record in records:
        case = record["case"]
        if "split" not in case:
            case["split"] = case["splits"]
        key = (
            case["model"],
            case["batch"],
            case["activation"],
            case["backend"],
            case["split"],
        )
        grouped[key].append(record["run"]["median_ms"])
        per_round[(*key, record["matrix"]["round"])] = record["run"]["median_ms"]

    summaries = []
    for key in sorted(grouped):
        model, batch, activation, backend, split = key
        values = grouped[key]
        baseline_values = grouped[(model, batch, activation, "tpp", 1)]
        baseline_median = statistics.median(baseline_values)
        case_median = statistics.median(values)
        paired_speedups = []
        paired_relative_times = []
        for round_index in sorted(
            record["matrix"]["round"]
            for record in records
            if record["case"]["model"] == model
            and record["case"]["batch"] == batch
            and record["case"]["activation"] == activation
            and record["case"]["backend"] == backend
            and record["case"]["split"] == split
        ):
            baseline_round = per_round[
                (model, batch, activation, "tpp", 1, round_index)
            ]
            case_round = per_round[
                (model, batch, activation, backend, split, round_index)
            ]
            paired_speedups.append(baseline_round / case_round)
            paired_relative_times.append(case_round / baseline_round)
        speedup_ci_low, speedup_ci_high = bootstrap_median_ci(paired_speedups)
        relative_ci_low, relative_ci_high = bootstrap_median_ci(paired_relative_times)
        target = target_for(claim_name, claim, model, batch, split)
        paper_speedup = target.get("speedup")
        paper_relative_time = target.get("relative_time")
        summaries.append(
            {
                "claim": claim_name,
                "model": model,
                "batch": batch,
                "sequence": claim["sequence"],
                "activation": activation,
                "backend": backend,
                "split": split,
                "rounds": len(values),
                "median_ms": case_median,
                "q1_ms": quantile(values, 0.25),
                "q3_ms": quantile(values, 0.75),
                "baseline_median_ms": baseline_median,
                "relative_time": case_median / baseline_median,
                "speedup": baseline_median / case_median,
                "paired_speedup_median": statistics.median(paired_speedups),
                "paired_speedup_ci95_low": speedup_ci_low,
                "paired_speedup_ci95_high": speedup_ci_high,
                "paired_relative_time_median": statistics.median(paired_relative_times),
                "paired_relative_time_ci95_low": relative_ci_low,
                "paired_relative_time_ci95_high": relative_ci_high,
                "paper_speedup": paper_speedup,
                "paper_relative_time": paper_relative_time,
                "paper_timing_target_in_ci95": (
                    speedup_ci_low <= paper_speedup <= speedup_ci_high
                    if paper_speedup is not None
                    else (
                        relative_ci_low <= paper_relative_time <= relative_ci_high
                        if paper_relative_time is not None
                        else None
                    )
                ),
                "paper_l3_miss_reduction_factor": target.get("l3_miss_reduction_factor"),
                "paper_l3_misses_billions": target.get("l3_misses_billions"),
            }
        )
    for row in summaries:
        candidates = [
            candidate
            for candidate in summaries
            if candidate["model"] == row["model"]
            and candidate["batch"] == row["batch"]
            and candidate["activation"] == row["activation"]
            and candidate["backend"] == "imbps"
        ]
        best = min(candidates, key=lambda candidate: candidate["median_ms"])
        if claim_name == "table_ii":
            paper_best = claim["targets"][str(row["batch"])]["split"]
        elif claim_name == "table_viii":
            paper_best = claim["reported_timing_optimum"][row["model"]]
        else:
            paper_best = None
        row["empirical_best_imbps_split"] = best["split"]
        row["is_empirical_best_imbps"] = (
            row["backend"] == "imbps" and row["split"] == best["split"]
        )
        row["paper_reported_best_split"] = paper_best
        row["paper_best_split_reproduced"] = (
            best["split"] == paper_best if paper_best is not None else None
        )
    return summaries


def write_summary(
    result_dir: Path,
    records: list[dict[str, Any]],
    claim_name: str,
    claim: dict[str, Any],
) -> None:
    summary = summarize(records, claim_name, claim)
    if not summary:
        raise ValueError("cannot summarize an empty result set")
    fieldnames = list(summary[0])
    with (result_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def finalize_existing(
    result_dir: Path,
    claim_name: str,
    claim: dict[str, Any],
) -> None:
    result_dir = result_dir.resolve()
    manifest_path = result_dir / "manifest.json"
    raw_dir = result_dir / "raw"
    if not manifest_path.is_file() or not raw_dir.is_dir():
        raise ValueError(f"not a matrix result directory: {result_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("claim") != claim_name:
        raise ValueError(
            f"claim mismatch: requested {claim_name}, manifest has {manifest.get('claim')}"
        )
    raw_paths = sorted(raw_dir.glob("*.json"))
    expected = manifest["runner"]["rounds"] * len(manifest["cases_per_round"])
    if len(raw_paths) != expected:
        raise ValueError(
            f"incomplete result set: expected {expected} raw records, found {len(raw_paths)}"
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
    observed_slots = {
        (record["matrix"]["round"], record["matrix"]["order"])
        for record in records
    }
    if len(observed_slots) != expected:
        raise ValueError("duplicate round/order slots in raw records")
    write_summary(result_dir, records, claim_name, claim)
    print(f"Finalized existing results: {result_dir}")


def main() -> None:
    args = parse_args()
    registry, claim = load_claim(args)
    if args.finalize_existing is not None:
        finalize_existing(args.finalize_existing, args.claim, claim)
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = args.result_dir or PROJECT_ROOT / "results" / f"{args.claim}-{timestamp}"
    raw_dir = result_dir / "raw"
    log_dir = result_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)

    base_cases = build_cases(registry, claim)
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": args.claim,
        "paper": registry["paper"],
        "pace": registry["pace"],
        "runner": {
            "python": sys.executable,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "shuffle_seed": args.seed,
            "threads": args.threads,
            "case_timeout_seconds": args.case_timeout_seconds,
            "retry_policy": "none",
        },
        "cases_per_round": base_cases,
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        cases = [dict(case) for case in base_cases]
        random.Random(args.seed + round_index).shuffle(cases)
        for order_index, case in enumerate(cases, start=1):
            case_id = (
                f"r{round_index:02d}-o{order_index:02d}-{case['model']}-"
                f"b{case['batch']}-{case['backend']}-k{case['split']}"
            )
            output_path = raw_dir / f"{case_id}.json"
            log_path = log_dir / f"{case_id}.log"
            command = [
                sys.executable,
                str(WORKER_PATH),
                "--backend",
                case["backend"],
                "--batch",
                str(case["batch"]),
                "--sequence",
                str(case["sequence"]),
                "--hidden",
                str(case["hidden"]),
                "--intermediate",
                str(case["intermediate"]),
                "--splits",
                str(case["split"]),
                "--dtype",
                case["dtype"],
                "--activation",
                case["activation"],
                "--warmups",
                str(args.warmups),
                "--iterations",
                str(args.iterations),
                "--seed",
                "0",
                "--output",
                str(output_path),
            ]
            if args.threads is not None:
                command.extend(("--threads", str(args.threads)))
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.case_timeout_seconds,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"case {case_id} failed with exit code {completed.returncode}; "
                    f"no retry was attempted; see {log_path}"
                )
            record = json.loads(output_path.read_text(encoding="utf-8"))
            record["case"]["model"] = case["model"]
            record["case"]["split"] = case["split"]
            record["matrix"] = {"round": round_index, "order": order_index}
            output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            print(
                f"round={round_index} order={order_index} model={case['model']} "
                f"batch={case['batch']} backend={case['backend']} K={case['split']} "
                f"median_ms={record['run']['median_ms']:.3f}",
                flush=True,
            )

    write_summary(result_dir, records, args.claim, claim)
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
