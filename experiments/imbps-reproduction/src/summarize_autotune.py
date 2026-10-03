#!/usr/bin/env python3
"""Combine isolated IMBPS autotuning matrices into comparable rankings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("results") / f"autotune-analysis-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    combined: list[dict[str, Any]] = []
    for result_dir in args.result_dirs:
        # Shell globs can also match the adjacent CPU-selection metadata files.
        # Only benchmark result directories contain manifests and summaries.
        if not result_dir.is_dir():
            continue
        manifest_path = result_dir / "manifest.json"
        summary_path = result_dir / "summary.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise ValueError(f"missing manifest or summary in {result_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        environment = manifest.get("environment", {})
        threads = manifest["runner"].get("threads")
        wait_policy = environment.get("omp_wait_policy")
        for source in json.loads(summary_path.read_text(encoding="utf-8")):
            combined.append(
                {
                    "result_dir": str(result_dir),
                    "claim": manifest["claim"],
                    "threads": threads,
                    "wait_policy": wait_policy,
                    "affinity_count": len(environment.get("process_affinity", [])),
                    "model": source["model"],
                    "batch": source["batch"],
                    "sequence": source["sequence"],
                    "rows": source["batch"] * source["sequence"],
                    "backend": source["backend"],
                    "split": source["split"],
                    "median_ms": source["median_ms"],
                    "paired_speedup_median": source["paired_speedup_median"],
                    "paired_speedup_ci95_low": source["paired_speedup_ci95_low"],
                    "paired_speedup_ci95_high": source["paired_speedup_ci95_high"],
                    "working_set_mib": source.get("equation12_working_set_mib"),
                    "working_set_fits": source.get("equation12_fits"),
                }
            )

    if not combined:
        raise ValueError("no benchmark result directories were found")

    combined.sort(
        key=lambda row: (
            row["claim"],
            row["rows"],
            row["batch"],
            row["threads"] or 0,
            row["wait_policy"] or "",
            row["backend"],
            row["split"],
        )
    )
    write_csv(output_dir / "combined.csv", combined)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        grouped[
            (
                row["claim"],
                row["threads"],
                row["wait_policy"],
                row["backend"],
                row["split"],
            )
        ].append(row)
    rankings: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        claim, threads, wait_policy, backend, split = key
        rankings.append(
            {
                "claim": claim,
                "threads": threads,
                "wait_policy": wait_policy,
                "backend": backend,
                "split": split,
                "shape_count": len(rows),
                "geomean_paired_speedup": geometric_mean(
                    [row["paired_speedup_median"] for row in rows]
                ),
                "median_case_cv_proxy": statistics.median(
                    [
                        (row["paired_speedup_ci95_high"] - row["paired_speedup_ci95_low"])
                        / row["paired_speedup_median"]
                        for row in rows
                    ]
                ),
            }
        )
    rankings.sort(
        key=lambda row: (
            row["claim"],
            -row["geomean_paired_speedup"],
            row["threads"] or 0,
            row["split"],
        )
    )
    write_csv(output_dir / "rankings.csv", rankings)

    by_shape: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        by_shape[
            (row["claim"], row["model"], row["batch"], row["sequence"])
        ].append(row)
    best_by_shape = []
    for key, rows in sorted(by_shape.items()):
        best = min(rows, key=lambda row: row["median_ms"])
        best_by_shape.append(
            {
                "claim": key[0],
                "model": key[1],
                "batch": key[2],
                "sequence": key[3],
                "rows": key[2] * key[3],
                "best_backend": best["backend"],
                "best_split": best["split"],
                "best_threads": best["threads"],
                "best_wait_policy": best["wait_policy"],
                "best_median_ms": best["median_ms"],
                "paired_speedup_at_best": best["paired_speedup_median"],
                "source_result_dir": best["result_dir"],
            }
        )
    (output_dir / "best-by-shape.json").write_text(
        json.dumps(best_by_shape, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Autotune analysis: {output_dir}")


if __name__ == "__main__":
    main()
