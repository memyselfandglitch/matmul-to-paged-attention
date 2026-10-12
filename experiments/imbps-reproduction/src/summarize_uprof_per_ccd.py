#!/usr/bin/env python3
"""Summarize CCD-granular L3 counters for threaded per-CCD IMBPS."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from .uprof_report import parse_uprof_component_report
except ImportError:  # Direct execution
    from uprof_report import parse_uprof_component_report


def component_index(name: str, fallback: int) -> int:
    matches = re.findall(r"\d+", name)
    return int(matches[-1]) if matches else fallback


def summarize(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no per-CCD uProf manifests below {root}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete profile: {manifest_path}")
        case_dir = manifest_path.parent
        benchmark = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
        record = benchmark["records"][0]
        variant = benchmark["variant"]
        invocations = len(record["durations_ms"])
        report = parse_uprof_component_report(case_dir / "uprof.csv")
        try:
            accesses = report["l3"]["L3 Access"]
            misses = report["l3"]["L3 Miss"]
        except KeyError as error:
            raise ValueError(f"missing CCD L3 metrics in {case_dir}") from error
        common_components = [name for name in accesses if name in misses]
        named_ccds = [
            name
            for name in common_components
            if "ccd" in name.lower() or "ccx" in name.lower()
        ]
        components = named_ccds or common_components
        if not components:
            raise ValueError(f"no common L3 components in {case_dir}")
        summed_access = 0.0
        summed_miss = 0.0
        component_rows: list[dict[str, Any]] = []
        core = report.get("core", {})
        ipc = core.get("IPC (Sys + User)", {})
        utilization = core.get("Utilization (%)", {})
        l2_access_pti = core.get("L2 Access (pti)", {})
        l2_miss_pti = core.get("L2 Miss (pti)", {})
        for fallback, component in enumerate(components):
            access = accesses[component]
            miss = misses[component]
            index = component_index(component, fallback)
            summed_access += access
            summed_miss += miss
            component_row = {
                    "claim": benchmark["claim"],
                    "active_rows": record["active_rows"],
                    "variant_id": variant["variant_id"],
                    "mode": variant["mode"],
                    "ccd_count": variant["ccd_count"],
                    "total_splits": variant["total_splits"],
                    "component": component,
                    "component_index": index,
                    "selected_by_variant": index < int(variant["ccd_count"]),
                    "invocations": invocations,
                    "l3_access_per_invocation": access / invocations,
                    "l3_miss_per_invocation": miss / invocations,
                    "l3_hit_percent": 100 * (access - miss) / access if access else None,
                    "ipc": ipc.get(component),
                    "utilization_percent": utilization.get(component),
                    "l2_access_pti": l2_access_pti.get(component),
                    "l2_miss_pti": l2_miss_pti.get(component),
                }
            long_rows.append(component_row)
            component_rows.append(component_row)
        miss_values = [float(row["l3_miss_per_invocation"]) for row in component_rows]
        utilization_values = [
            float(row["utilization_percent"])
            for row in component_rows
            if row["utilization_percent"] is not None
        ]
        ipc_values = [
            float(row["ipc"]) for row in component_rows if row["ipc"] is not None
        ]
        ccd0 = min(component_rows, key=lambda row: int(row["component_index"]))
        aggregate_rows.append(
            {
                "claim": benchmark["claim"],
                "active_rows": record["active_rows"],
                "variant_id": variant["variant_id"],
                "mode": variant["mode"],
                "ccd_count": variant["ccd_count"],
                "total_splits": variant["total_splits"],
                "reduction_strategy": variant.get("reduction_strategy"),
                "invocations": invocations,
                "median_ms": record["median_ms"],
                "median_compute_and_sync_ms": record.get(
                    "median_compute_and_sync_ms"
                ),
                "median_reduction_and_conversion_ms": record.get(
                    "median_reduction_and_conversion_ms"
                ),
                "l3_access_per_invocation": summed_access / invocations,
                "l3_miss_per_invocation": summed_miss / invocations,
                "l3_hit_percent": (
                    100 * (summed_access - summed_miss) / summed_access
                    if summed_access
                    else None
                ),
                "l3_miss_max_to_min_ratio": (
                    max(miss_values) / min(miss_values)
                    if miss_values and min(miss_values) > 0
                    else None
                ),
                "ccd0_l3_miss_per_invocation": ccd0["l3_miss_per_invocation"],
                "ccd0_l3_hit_percent": ccd0["l3_hit_percent"],
                "ccd0_utilization_percent": ccd0["utilization_percent"],
                "other_ccd_median_utilization_percent": (
                    statistics.median(
                        float(row["utilization_percent"])
                        for row in component_rows
                        if int(row["component_index"]) != 0
                        and row["utilization_percent"] is not None
                    )
                    if len(utilization_values) > 1
                    else None
                ),
                "utilization_max_to_min_ratio": (
                    max(utilization_values) / min(utilization_values)
                    if utilization_values and min(utilization_values) > 0
                    else None
                ),
                "ipc_median_across_ccds": (
                    statistics.median(ipc_values) if ipc_values else None
                ),
                "ipc_min_across_ccds": min(ipc_values) if ipc_values else None,
                "ipc_max_across_ccds": max(ipc_values) if ipc_values else None,
                "counter_scope": "uProf package=0 with -A ccd; summed here",
            }
        )
    return long_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    long_rows, aggregate_rows = summarize(args.result_root)
    output = args.result_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ccd-l3.csv", long_rows)
    write_csv(output / "summary.csv", aggregate_rows)
    (output / "summary.json").write_text(
        json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
