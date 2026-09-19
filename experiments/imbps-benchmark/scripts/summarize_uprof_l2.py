#!/usr/bin/env python3
"""Summarize minimal Zen 4 L2 AMDuProfPcm reports from matched runs."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List


FILE_PATTERN = re.compile(
    r"r(?P<rep>\d+)-(?P<variant>reference|imbps)-k(?P<split_k>\d+)\.csv$"
)
METRICS = {
    "L2 Demand Access": "l2_demand_access",
    "L2 Demand Miss": "l2_demand_miss",
    "L2 Prefetch Miss": "l2_prefetch_miss",
    "L2 Total Miss": "l2_total_miss",
}
RAW_FIELDS = (
    "rep",
    "variant",
    "split_k",
    "iterations",
    *METRICS.values(),
    *(f"{field}_per_iteration" for field in METRICS.values()),
)
SUMMARY_FIELDS = (
    "variant",
    "split_k",
    "samples",
    "iterations",
    "l2_demand_access_per_iteration_median",
    "l2_demand_miss_per_iteration_median",
    "l2_prefetch_miss_per_iteration_median",
    "l2_total_miss_per_iteration_median",
    "l2_total_miss_per_iteration_min",
    "l2_total_miss_per_iteration_max",
    "l2_total_miss_per_iteration_cv_pct",
    "reference_l2_total_miss_per_iteration_median",
    "l2_total_miss_reduction_vs_reference_pct",
)


def _parse_report(path: Path) -> Dict[str, object]:
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected report filename: {path.name}")
    row: Dict[str, object] = {
        "rep": int(match.group("rep")),
        "variant": match.group("variant"),
        "split_k": int(match.group("split_k")),
    }
    command = ""
    with path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 2:
                continue
            name = fields[0].strip()
            if name == "Command:":
                command = fields[1].strip()
            field = METRICS.get(name)
            if field is not None:
                row[field] = float(fields[1].strip())
    match_iterations = re.search(r"--iterations\s+(\d+)", command)
    if match_iterations is None:
        raise ValueError(f"iterations missing from command in {path}")
    iterations = int(match_iterations.group(1))
    row["iterations"] = iterations
    missing = [field for field in METRICS.values() if field not in row]
    if missing:
        raise ValueError(f"required metrics missing from {path}: {missing}")
    for field in METRICS.values():
        row[f"{field}_per_iteration"] = float(row[field]) / iterations
    return row


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("r*-*.csv"))
    if not paths:
        raise SystemExit(f"no uProf reports found under {args.directory}")
    raw_rows = [_parse_report(path) for path in paths]
    raw_rows.sort(key=lambda row: (str(row["variant"]), int(row["split_k"]), int(row["rep"])))
    reference = statistics.median(
        float(row["l2_total_miss_per_iteration"])
        for row in raw_rows
        if row["variant"] == "reference"
    )
    groups: Dict[tuple[str, int], List[Dict[str, object]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["variant"]), int(row["split_k"])), []).append(row)
    summaries = []
    for (variant, split_k), rows in sorted(groups.items()):
        total_misses = [float(row["l2_total_miss_per_iteration"]) for row in rows]
        total_mean = statistics.fmean(total_misses)
        total_median = statistics.median(total_misses)
        summaries.append(
            {
                "variant": variant,
                "split_k": split_k,
                "samples": len(rows),
                "iterations": int(rows[0]["iterations"]),
                "l2_demand_access_per_iteration_median": statistics.median(
                    float(row["l2_demand_access_per_iteration"]) for row in rows
                ),
                "l2_demand_miss_per_iteration_median": statistics.median(
                    float(row["l2_demand_miss_per_iteration"]) for row in rows
                ),
                "l2_prefetch_miss_per_iteration_median": statistics.median(
                    float(row["l2_prefetch_miss_per_iteration"]) for row in rows
                ),
                "l2_total_miss_per_iteration_median": total_median,
                "l2_total_miss_per_iteration_min": min(total_misses),
                "l2_total_miss_per_iteration_max": max(total_misses),
                "l2_total_miss_per_iteration_cv_pct": (
                    statistics.stdev(total_misses) / total_mean * 100.0
                    if len(total_misses) > 1 and total_mean
                    else 0.0
                ),
                "reference_l2_total_miss_per_iteration_median": reference,
                "l2_total_miss_reduction_vs_reference_pct":
                    (1.0 - total_median / reference) * 100.0,
            }
        )
    _write_csv(args.directory / "raw_metrics.csv", RAW_FIELDS, raw_rows)
    _write_csv(args.directory / "summary.csv", SUMMARY_FIELDS, summaries)
    for row in summaries:
        print(
            "%-9s K=%-2d L2_total_miss/iter=%10.1f reduction=%8.3f%% CV=%6.3f%%"
            % (
                row["variant"],
                row["split_k"],
                row["l2_total_miss_per_iteration_median"],
                row["l2_total_miss_reduction_vs_reference_pct"],
                row["l2_total_miss_per_iteration_cv_pct"],
            )
        )


if __name__ == "__main__":
    main()
