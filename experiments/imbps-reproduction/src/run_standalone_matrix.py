#!/usr/bin/env python3
"""Run Table II or VIII as isolated, randomized PACE processes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .cache_model import (
        MIB,
        equation13_lower_bound,
        next_power_of_two_candidate,
        strict_integer_candidate,
        working_set_bytes,
    )
except ImportError:  # Direct execution: python src/run_standalone_matrix.py
    from cache_model import (
        MIB,
        equation13_lower_bound,
        next_power_of_two_candidate,
        strict_integer_candidate,
        working_set_bytes,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
WORKER_PATH = PROJECT_ROOT / "src" / "pace_mlp_bench.py"


def comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claim",
        choices=(
            "table_ii",
            "table_viii",
            "decode_exploratory",
            "decode_l2_opt30b",
            "decode_l2_pilot_opt30b",
            "cache_fit_opt30b",
            "cache_resident_control",
            "autotune_thread_wait",
            "autotune_split_width",
            "autotune_equal_rows",
        ),
        required=True,
    )
    parser.add_argument("--models", type=comma_strings, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--min-measurement-seconds", type=float, default=0.0)
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
    if (
        min(args.rounds, args.iterations) <= 0
        or args.warmups < 0
        or args.min_measurement_seconds < 0
    ):
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
    if "cases" in claim:
        shapes = [
            shape for shape in claim["cases"] if shape["model"] in claim["models"]
        ]
    else:
        shapes = [
            {"model": model_name, "batch": batch, "sequence": claim["sequence"]}
            for model_name in claim["models"]
            for batch in claim["batches"]
        ]
    for shape in shapes:
        model_name = shape["model"]
        model = registry["models"][model_name]
        batch = shape["batch"]
        sequence = shape["sequence"]
        variants = [("tpp", 1)] + [
            ("imbps", split) for split in claim["splits"]
        ]
        for backend, split in variants:
            if model["intermediate"] % split != 0:
                raise ValueError(
                    f"{model_name} intermediate={model['intermediate']} is not "
                    f"divisible by K={split}"
                )
            case = {
                "model": model_name,
                "backend": backend,
                "split": split,
                "batch": batch,
                "sequence": sequence,
                "hidden": model["hidden"],
                "intermediate": model["intermediate"],
                "activation": model["activation"],
                "dtype": claim["dtype"],
            }
            if "cache_mib" in claim:
                bytes_per_element = 2 if claim["dtype"] == "bf16" else 4
                working_set = working_set_bytes(
                    batch,
                    sequence,
                    model["hidden"],
                    model["intermediate"],
                    split,
                    bytes_per_element,
                )
                cache_bytes = round(claim["cache_mib"] * MIB)
                lower_bound = equation13_lower_bound(
                    batch,
                    sequence,
                    model["hidden"],
                    model["intermediate"],
                    bytes_per_element,
                    cache_bytes,
                )
                case["cache_model"] = {
                    "cache_level": claim.get("cache_level", "l3"),
                    "cache_scope": claim.get("cache_scope", "aggregate_shared_cache"),
                    "cache_mib": claim["cache_mib"],
                    "input_mib": working_set.input_bytes / MIB,
                    "split_activation_mib": working_set.split_activation_bytes / MIB,
                    "split_weight_mib": working_set.split_weight_bytes / MIB,
                    "working_set_mib": working_set.total_bytes / MIB,
                    "fits_strict": working_set.total_bytes < cache_bytes,
                    "equation13_strict_lower_bound": (
                        lower_bound if math.isfinite(lower_bound) else None
                    ),
                    "equation13_strict_integer_candidate": (
                        strict_integer_candidate(lower_bound)
                    ),
                    "equation13_author_power_of_two_candidate": (
                        next_power_of_two_candidate(lower_bound)
                    ),
                }
            cases.append(case)
    fit_requirement = claim.get("fit_requirement")
    if fit_requirement is not None:
        if fit_requirement not in {"all", "imbps"}:
            raise ValueError(f"unsupported fit_requirement: {fit_requirement}")
        required_cases = (
            cases
            if fit_requirement == "all"
            else [case for case in cases if case["backend"] == "imbps"]
        )
        violations = [
            case
            for case in required_cases
            if not case.get("cache_model", {}).get("fits_strict", False)
        ]
        if violations:
            descriptions = ", ".join(
                f"{case['model']} B={case['batch']} SL={case['sequence']} "
                f"{case['backend']} K={case['split']}"
                for case in violations
            )
            raise ValueError(
                f"cache-fit claim contains non-fitting required cases: {descriptions}"
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
    grouped: dict[tuple[str, int, int, str, str, int], list[float]] = defaultdict(list)
    per_round: dict[tuple[str, int, int, str, str, int, int], float] = {}
    cache_models: dict[tuple[str, int, int, str, str, int], dict[str, Any]] = {}
    for record in records:
        case = record["case"]
        if "split" not in case:
            case["split"] = case["splits"]
        key = (
            case["model"],
            case["batch"],
            case["sequence"],
            case["activation"],
            case["backend"],
            case["split"],
        )
        grouped[key].append(record["run"]["median_ms"])
        per_round[(*key, record["matrix"]["round"])] = record["run"]["median_ms"]
        if "cache_model" in record:
            cache_models[key] = record["cache_model"]

    summaries = []
    for key in sorted(grouped):
        model, batch, sequence, activation, backend, split = key
        values = grouped[key]
        baseline_values = grouped[(model, batch, sequence, activation, "tpp", 1)]
        baseline_median = statistics.median(baseline_values)
        case_median = statistics.median(values)
        paired_speedups = []
        paired_relative_times = []
        for round_index in sorted(
            record["matrix"]["round"]
            for record in records
            if record["case"]["model"] == model
            and record["case"]["batch"] == batch
            and record["case"]["sequence"] == sequence
            and record["case"]["activation"] == activation
            and record["case"]["backend"] == backend
            and record["case"]["split"] == split
        ):
            baseline_round = per_round[
                (model, batch, sequence, activation, "tpp", 1, round_index)
            ]
            case_round = per_round[
                (model, batch, sequence, activation, backend, split, round_index)
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
                "sequence": sequence,
                "activation": activation,
                "backend": backend,
                "split": split,
                "rounds": len(values),
                "active_rows": batch * sequence,
                "milliseconds_per_active_row": case_median / (batch * sequence),
                "active_rows_per_second": 1000 * batch * sequence / case_median,
                "median_ms": case_median,
                "q1_ms": quantile(values, 0.25),
                "q3_ms": quantile(values, 0.75),
                "baseline_median_ms": baseline_median,
                "relative_time": case_median / baseline_median,
                "speedup": baseline_median / case_median,
                "paired_speedup_median": statistics.median(paired_speedups),
                "paired_speedup_ci95_low": speedup_ci_low,
                "paired_speedup_ci95_high": speedup_ci_high,
                "paired_speedup_ci95_excludes_one": (
                    speedup_ci_low > 1 or speedup_ci_high < 1
                ),
                "paired_speedup_supports_faster": speedup_ci_low > 1,
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
                "cache_mib": cache_models.get(key, {}).get("cache_mib"),
                "cache_level": cache_models.get(key, {}).get("cache_level"),
                "cache_scope": cache_models.get(key, {}).get("cache_scope"),
                "equation12_working_set_mib": cache_models.get(key, {}).get(
                    "working_set_mib"
                ),
                "equation12_fits": cache_models.get(key, {}).get("fits_strict"),
                "equation13_strict_lower_bound": cache_models.get(key, {}).get(
                    "equation13_strict_lower_bound"
                ),
                "equation13_strict_integer_candidate": cache_models.get(key, {}).get(
                    "equation13_strict_integer_candidate"
                ),
                "equation13_author_power_of_two_candidate": cache_models.get(
                    key, {}
                ).get("equation13_author_power_of_two_candidate"),
            }
        )
    draft_token_counts = claim.get("semantic_mappings", {}).get(
        "draft_token_counts", []
    )
    for row in summaries:
        row["normal_decode_batch_equivalent"] = row["active_rows"]
        for draft_tokens in draft_token_counts:
            field = f"speculative_batch_equivalent_gamma{draft_tokens}"
            row[field] = (
                row["active_rows"] // draft_tokens
                if row["active_rows"] % draft_tokens == 0
                else None
            )
    for row in summaries:
        candidates = [
            candidate
            for candidate in summaries
            if candidate["model"] == row["model"]
            and candidate["batch"] == row["batch"]
            and candidate["sequence"] == row["sequence"]
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
        row["equation_candidate_is_empirical_best"] = (
            best["split"] == row["equation13_author_power_of_two_candidate"]
            if row["equation13_author_power_of_two_candidate"] is not None
            else None
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
            "min_measurement_seconds": args.min_measurement_seconds,
            "shuffle_seed": args.seed,
            "threads": args.threads,
            "case_timeout_seconds": args.case_timeout_seconds,
            "retry_policy": "none",
        },
        "environment": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "gomp_cpu_affinity": os.environ.get("GOMP_CPU_AFFINITY"),
            "ld_preload": os.environ.get("LD_PRELOAD"),
            "tcmalloc_prefix": os.environ.get("TCMALLOC_PREFIX"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "process_affinity": sorted(os.sched_getaffinity(0)),
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
                f"b{case['batch']}-s{case['sequence']}-{case['backend']}-k{case['split']}"
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
                "--min-measurement-seconds",
                str(args.min_measurement_seconds),
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
            if "cache_model" in case:
                record["cache_model"] = case["cache_model"]
            record["matrix"] = {"round": round_index, "order": order_index}
            output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)
            print(
                f"round={round_index} order={order_index} model={case['model']} "
                f"batch={case['batch']} sequence={case['sequence']} "
                f"backend={case['backend']} K={case['split']} "
                f"median_ms={record['run']['median_ms']:.3f}",
                flush=True,
            )

    write_summary(result_dir, records, args.claim, claim)
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
