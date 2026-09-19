#!/usr/bin/env python3
"""Collect nested Hugging Face OPT layer row sweeps into one CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "rows",
    "adapter_k1_speedup",
    "best_split_k",
    "best_split_speedup",
    "best_split_reference_median_ms",
    "best_split_median_ms",
    "best_split_workspace_bytes",
    "best_split_logical_activation_bytes",
    "best_split_max_abs_error",
    "best_split_allclose",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output_rows = []
    for path in sorted(
        args.directory.glob("m*/summary.csv"),
        key=lambda item: int(item.parent.name.removeprefix("m")),
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["variant"] == "imbps"]
        k1 = next(row for row in rows if int(row["split_k"]) == 1)
        split = max(
            (row for row in rows if int(row["split_k"]) > 1),
            key=lambda row: float(row["speedup_vs_reference"]),
        )
        output_rows.append(
            {
                "rows": int(split["rows"]),
                "adapter_k1_speedup": float(k1["speedup_vs_reference"]),
                "best_split_k": int(split["split_k"]),
                "best_split_speedup": float(split["speedup_vs_reference"]),
                "best_split_reference_median_ms": float(split["reference_median_ms"]),
                "best_split_median_ms": float(split["latency_median_ms"]),
                "best_split_workspace_bytes": int(split["workspace_bytes"]),
                "best_split_logical_activation_bytes": int(split["logical_activation_bytes"]),
                "best_split_max_abs_error": float(split["max_abs_error"]),
                "best_split_allclose": split["allclose"],
            }
        )
    output = args.directory / "phase_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            "M=%-5d adapter-K1=%7.3fx best-split K=%-2d speedup=%7.3fx"
            % (
                row["rows"],
                row["adapter_k1_speedup"],
                row["best_split_k"],
                row["best_split_speedup"],
            )
        )


if __name__ == "__main__":
    main()
