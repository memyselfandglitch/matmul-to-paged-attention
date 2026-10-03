#!/usr/bin/env python3
"""Run randomized Table-II AMD uProf counter rounds and normalize metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
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
    from .uprof_report import parse_uprof_report, require_metric
except ImportError:  # Direct execution: python src/run_uprof_matrix.py
    from cache_model import (
        MIB,
        equation13_lower_bound,
        next_power_of_two_candidate,
        strict_integer_candidate,
        working_set_bytes,
    )
    from uprof_report import parse_uprof_report, require_metric


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
CASE_RUNNER = PROJECT_ROOT / "src" / "run_uprof_case.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claim",
        choices=("table_ii", "cache_fit_opt30b", "server_equation_opt30b"),
        default="table_ii",
    )
    parser.add_argument("--pass-name", choices=("cache", "traffic"), required=True)
    parser.add_argument("--uprof-bin", type=Path, required=True)
    parser.add_argument("--package", type=int, default=0)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def experiment_cases(claim_name: str) -> list[dict[str, Any]]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claim = registry["standalone"][claim_name]
    shapes = (
        [
            {
                "model": claim["models"][0],
                "batch": batch,
                "sequence": claim["sequence"],
            }
            for batch in claim["batches"]
        ]
        if claim_name == "table_ii"
        else claim["cases"]
    )
    cases = []
    for shape in shapes:
        model = registry["models"][shape["model"]]
        variants = [
            ("tpp", 1),
            *(("imbps", split) for split in shape.get("splits", claim["splits"])),
        ]
        cache_mib = float(claim.get("cache_mib", 384))
        lower_bound = equation13_lower_bound(
            shape["batch"],
            shape["sequence"],
            model["hidden"],
            model["intermediate"],
            2 if claim["dtype"] == "bf16" else 4,
            round(cache_mib * MIB),
        )
        for backend, split in variants:
            working_set = working_set_bytes(
                shape["batch"],
                shape["sequence"],
                model["hidden"],
                model["intermediate"],
                split,
                2 if claim["dtype"] == "bf16" else 4,
            )
            cases.append(
                {
                    "model": shape["model"],
                    "batch": shape["batch"],
                    "sequence": shape["sequence"],
                    "hidden": model["hidden"],
                    "intermediate": model["intermediate"],
                    "activation": model["activation"],
                    "dtype": claim["dtype"],
                    "backend": backend,
                    "split": split,
                    "cache_mib": cache_mib,
                    "equation12_working_set_mib": working_set.total_bytes / MIB,
                    "equation12_fits": working_set.total_bytes < cache_mib * MIB,
                    "equation13_strict_lower_bound": (
                        lower_bound if math.isfinite(lower_bound) else None
                    ),
                    "equation13_strict_integer_candidate": strict_integer_candidate(
                        lower_bound
                    ),
                    "equation13_author_power_of_two_candidate": (
                        next_power_of_two_candidate(lower_bound)
                    ),
                }
            )
    return cases


def write_records(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    (result_dir / "records.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (result_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_record(
    case_dir: Path,
    case: dict[str, Any],
    pass_name: str,
    round_number: int,
    order: int,
) -> dict[str, Any]:
    benchmark = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    report = parse_uprof_report(case_dir / "uprof.csv")
    iterations = int(manifest["measurement"]["iterations"])
    active_seconds = float(manifest["measurement"]["counter_active_seconds"])
    rows = case["batch"] * case["sequence"]
    algorithmic_flops = 4 * rows * case["hidden"] * case["intermediate"]
    record: dict[str, Any] = {
        "pass": pass_name,
        "round": round_number,
        "order": order,
        "model": case["model"],
        "batch": case["batch"],
        "sequence": case["sequence"],
        "rows": rows,
        "backend": case["backend"],
        "split": case["split"],
        "iterations": iterations,
        "median_ms": benchmark["run"]["median_ms"],
        "measurement_wall_seconds": benchmark["run"]["measurement_wall_seconds"],
        "counter_active_seconds": active_seconds,
        "algorithmic_flops_per_invocation": algorithmic_flops,
        "cache_mib": case["cache_mib"],
        "equation12_working_set_mib": case["equation12_working_set_mib"],
        "equation12_fits": case["equation12_fits"],
        "equation13_strict_lower_bound": case["equation13_strict_lower_bound"],
        "equation13_strict_integer_candidate": case[
            "equation13_strict_integer_candidate"
        ],
        "equation13_author_power_of_two_candidate": case[
            "equation13_author_power_of_two_candidate"
        ],
        "case_dir": str(case_dir),
    }
    if pass_name == "cache":
        access = require_metric(report, "l3", "L3 Access")
        misses = require_metric(report, "l3", "L3 Miss")
        record.update(
            {
                "ipc": require_metric(report, "core", "IPC (Sys + User)"),
                "l2_access_pti": require_metric(report, "core", "L2 Access (pti)"),
                "l2_miss_pti": require_metric(report, "core", "L2 Miss (pti)"),
                "l3_access_per_invocation": access / iterations,
                "l3_miss_per_invocation": misses / iterations,
                "l3_hit_per_invocation": (access - misses) / iterations,
                "l3_miss_percent": require_metric(report, "l3", "L3 Miss %"),
                "l3_hit_percent": require_metric(report, "l3", "L3 Hit %"),
                "l3_miss_latency_ns": require_metric(
                    report, "l3", "Ave L3 Miss Latency (ns)"
                ),
            }
        )
    else:
        total_bw = require_metric(report, "df", "Total Mem Bw (GB/s)")
        read_bw = require_metric(report, "df", "Total Mem RdBw (GB/s)")
        write_bw = require_metric(report, "df", "Total Mem WrBw (GB/s)")
        dram_bytes = total_bw * 1e9 * active_seconds / iterations
        record.update(
            {
                "total_mem_bw_gbps": total_bw,
                "read_mem_bw_gbps": read_bw,
                "write_mem_bw_gbps": write_bw,
                "dram_bytes_per_invocation": dram_bytes,
                "dram_read_bytes_per_invocation": (
                    read_bw * 1e9 * active_seconds / iterations
                ),
                "dram_write_bytes_per_invocation": (
                    write_bw * 1e9 * active_seconds / iterations
                ),
                "measured_dram_ai_flops_per_byte": algorithmic_flops / dram_bytes,
            }
        )
    return record


def main() -> None:
    args = parse_args()
    metrics = "ipc,l2,l3" if args.pass_name == "cache" else "memory"
    result_dir = args.result_dir.resolve()
    cases_dir = result_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=False)
    cases = experiment_cases(args.claim)
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": args.claim,
        "pass": args.pass_name,
        "metrics": metrics.split(","),
        "rounds": args.rounds,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "seed": args.seed,
        "threads": args.threads,
        "package": args.package,
        "cpu_list": args.cpu_list,
        "uprof_binary": str(args.uprof_bin),
        "case_count_per_round": len(cases),
        "status": "running",
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    records: list[dict[str, Any]] = []
    for round_number in range(1, args.rounds + 1):
        ordered = list(cases)
        random.Random(args.seed + round_number).shuffle(ordered)
        for order, case in enumerate(ordered, start=1):
            case_id = (
                f"r{round_number:02d}-o{order:02d}-{case['model']}-"
                f"b{case['batch']}-{case['backend']}-k{case['split']}"
            )
            case_dir = cases_dir / case_id
            command = [
                sys.executable,
                str(CASE_RUNNER),
                "--uprof-bin",
                str(args.uprof_bin),
                "--access-mode",
                "perf",
                "--metrics",
                metrics,
                "--package",
                str(args.package),
                "--cpu-list",
                args.cpu_list,
                "--backend",
                case["backend"],
                "--split",
                str(case["split"]),
                "--batch",
                str(case["batch"]),
                "--sequence",
                str(case["sequence"]),
                "--hidden",
                str(case["hidden"]),
                "--intermediate",
                str(case["intermediate"]),
                "--dtype",
                case["dtype"],
                "--activation",
                case["activation"],
                "--threads",
                str(args.threads),
                "--warmups",
                str(args.warmups),
                "--iterations",
                str(args.iterations),
                "--result-dir",
                str(case_dir),
            ]
            subprocess.run(command, check=True)
            record = normalized_record(
                case_dir, case, args.pass_name, round_number, order
            )
            records.append(record)
            write_records(result_dir, records)
            print(
                f"pass={args.pass_name} round={round_number} order={order} "
                f"batch={case['batch']} backend={case['backend']} "
                f"K={case['split']} median_ms={record['median_ms']:.3f}",
                flush=True,
            )

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["record_count"] = len(records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"uProf {args.claim} {args.pass_name}: {result_dir}")


if __name__ == "__main__":
    main()
