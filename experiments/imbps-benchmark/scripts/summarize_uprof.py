#!/usr/bin/env python3
"""Summarize cumulative AMDuProfPcm CSV reports from matched IMBPS runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


FILE_PATTERN = re.compile(
    r"r(?P<rep>\d+)-(?P<variant>reference|imbps)-k(?P<split_k>\d+)\.csv$"
)

METRICS = {
    "L3 Access": "l3_access",
    "L3 Miss": "l3_miss",
    "L3 Miss %": "l3_miss_pct",
    "L3 Hit %": "l3_hit_pct",
    "Ave L3 Miss Latency (ns)": "l3_miss_latency_ns",
    "L3 Miss Latency From Local Memory or I/O (%)": "l3_local_memory_pct",
    "L3 Miss Latency From Remote Memory or I/O (%)": "l3_remote_memory_pct",
    "L3 Miss Latency From another CCX in same node (%)": "l3_same_node_ccx_pct",
    "L3 Miss Latency From another CCX in remote node (%)": "l3_remote_node_ccx_pct",
    "Total Mem Bw (GB/s)": "memory_bw_gbps",
    "Local DRAM Read Data Bytes(GB/s)": "local_dram_read_gbps",
    "Local DRAM Write Data Bytes(GB/s)": "local_dram_write_gbps",
    "Remote DRAM Read Data Bytes (GB/s)": "remote_dram_read_gbps",
    "Remote DRAM Write Data Bytes (GB/s)": "remote_dram_write_gbps",
}

REQUIRED_METRICS = {
    "l3_access",
    "l3_miss",
    "l3_miss_pct",
    "l3_hit_pct",
}

RAW_FIELDS = (
    "rep",
    "variant",
    "split_k",
    "iterations",
    *METRICS.values(),
    "l3_access_per_iteration",
    "l3_miss_per_iteration",
)

SUMMARY_FIELDS = (
    "variant",
    "split_k",
    "samples",
    "iterations",
    "l3_access_per_iteration_median",
    "l3_access_per_iteration_min",
    "l3_access_per_iteration_max",
    "l3_miss_per_iteration_median",
    "l3_miss_per_iteration_min",
    "l3_miss_per_iteration_max",
    "l3_miss_per_iteration_cv_pct",
    "reference_l3_miss_per_iteration_median",
    "l3_miss_reduction_vs_reference_pct",
    "l3_miss_pct_median",
    "l3_hit_pct_median",
    "l3_miss_latency_ns_median",
    "memory_bw_gbps_median",
    "local_dram_read_gbps_median",
    "local_dram_write_gbps_median",
    "remote_dram_read_gbps_median",
    "remote_dram_write_gbps_median",
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
            value = fields[1].strip()
            if name == "Command:":
                command = value
            metric_name = METRICS.get(name)
            if metric_name is not None:
                row[metric_name] = float(value)
    iterations_match = re.search(r"--iterations\s+(\d+)", command)
    if iterations_match is None:
        raise ValueError(f"iterations missing from command in {path}")
    iterations = int(iterations_match.group(1))
    row["iterations"] = iterations
    missing_required = [name for name in REQUIRED_METRICS if name not in row]
    if missing_required:
        raise ValueError(f"required metrics missing from {path}: {missing_required}")
    for name in METRICS.values():
        row.setdefault(name, math.nan)
    row["l3_access_per_iteration"] = float(row["l3_access"]) / iterations
    row["l3_miss_per_iteration"] = float(row["l3_miss"]) / iterations
    return row


def _median(rows: Iterable[Dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else math.nan


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
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

    reference_rows = [row for row in raw_rows if row["variant"] == "reference"]
    reference_misses = _median(reference_rows, "l3_miss_per_iteration")
    groups: Dict[tuple[str, int], List[Dict[str, object]]] = {}
    for row in raw_rows:
        key = (str(row["variant"]), int(row["split_k"]))
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (variant, split_k), rows in sorted(groups.items()):
        misses = [float(row["l3_miss_per_iteration"]) for row in rows]
        accesses = [float(row["l3_access_per_iteration"]) for row in rows]
        miss_median = statistics.median(misses)
        miss_mean = statistics.fmean(misses)
        miss_cv = (
            statistics.stdev(misses) / miss_mean * 100.0
            if len(misses) > 1 and miss_mean
            else math.nan
        )
        summary_rows.append(
            {
                "variant": variant,
                "split_k": split_k,
                "samples": len(rows),
                "iterations": int(rows[0]["iterations"]),
                "l3_access_per_iteration_median": statistics.median(accesses),
                "l3_access_per_iteration_min": min(accesses),
                "l3_access_per_iteration_max": max(accesses),
                "l3_miss_per_iteration_median": miss_median,
                "l3_miss_per_iteration_min": min(misses),
                "l3_miss_per_iteration_max": max(misses),
                "l3_miss_per_iteration_cv_pct": miss_cv,
                "reference_l3_miss_per_iteration_median": reference_misses,
                "l3_miss_reduction_vs_reference_pct": (1.0 - miss_median / reference_misses) * 100.0,
                "l3_miss_pct_median": _median(rows, "l3_miss_pct"),
                "l3_hit_pct_median": _median(rows, "l3_hit_pct"),
                "l3_miss_latency_ns_median": _median(rows, "l3_miss_latency_ns"),
                "memory_bw_gbps_median": _median(rows, "memory_bw_gbps"),
                "local_dram_read_gbps_median": _median(rows, "local_dram_read_gbps"),
                "local_dram_write_gbps_median": _median(rows, "local_dram_write_gbps"),
                "remote_dram_read_gbps_median": _median(rows, "remote_dram_read_gbps"),
                "remote_dram_write_gbps_median": _median(rows, "remote_dram_write_gbps"),
            }
        )

    _write_csv(args.directory / "raw_metrics.csv", RAW_FIELDS, raw_rows)
    _write_csv(args.directory / "summary.csv", SUMMARY_FIELDS, summary_rows)
    for row in summary_rows:
        print(
            "%-9s K=%-2d L3_miss/iter=%10.1f reduction=%7.3f%% "
            "miss_rate=%7.3f%% CV=%6.3f%% mem_bw=%7.3f GB/s"
            % (
                row["variant"],
                row["split_k"],
                row["l3_miss_per_iteration_median"],
                row["l3_miss_reduction_vs_reference_pct"],
                row["l3_miss_pct_median"],
                row["l3_miss_per_iteration_cv_pct"],
                row["memory_bw_gbps_median"],
            )
        )


if __name__ == "__main__":
    main()
