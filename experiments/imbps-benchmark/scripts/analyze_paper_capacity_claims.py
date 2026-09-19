#!/usr/bin/env python3
"""Evaluate published IMBPS table configurations against cache equations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from imbps_bench.analytical import (
    paper_split_prediction,
    paper_working_set_bytes,
    resident_split_prediction,
    resident_working_set_bytes,
)


MIB = 1024 * 1024


@dataclass(frozen=True)
class Case:
    source: str
    model: str
    hidden: int
    batch: int
    sequence: int
    element_size: int
    reported_k: int | None
    cache_mib: float = 512.0


CASES = (
    *(Case("Table II", "OPT-30B", 7168, batch, 1920, 2, 4) for batch in (16, 32, 64)),
    Case("Table III", "OPT-6.7B", 4096, 32, 256, 4, 4),
    Case("Table III", "OPT-6.7B", 4096, 64, 256, 4, 4),
    Case("Table III", "OPT-6.7B", 4096, 128, 256, 4, 8),
    Case("Table III", "OPT-6.7B", 4096, 256, 256, 4, 7),
    Case("Table III", "OPT-30B", 7168, 32, 256, 4, 8),
    Case("Table III", "OPT-30B", 7168, 64, 256, 4, 4),
    Case("Table III", "OPT-30B", 7168, 128, 256, 4, 6),
    Case("Table III", "OPT-30B", 7168, 256, 256, 4, 8),
    *(Case("Table IV", model, hidden, batch, 256, 2, None)
      for model, hidden in (("OPT-13B", 5120), ("OPT-30B", 7168))
      for batch in (64, 128, 256, 512, 1024)),
    Case("Table VIII", "OPT-13B", 5120, 512, 256, 2, 4),
    Case("Table VIII", "OPT-30B", 7168, 512, 256, 2, 8),
)


FIELDS = (
    "source",
    "model",
    "batch",
    "sequence",
    "rows_m",
    "hidden_h",
    "intermediate_i",
    "element_size",
    "cache_mib",
    "reported_k",
    "paper_prediction_feasible",
    "paper_continuous_k",
    "paper_minimum_integer_k",
    "resident_prediction_feasible",
    "resident_continuous_k",
    "resident_minimum_integer_k",
    "paper_working_set_at_reported_k_mib",
    "resident_working_set_at_reported_k_mib",
    "resident_working_set_over_cache_at_reported_k",
)


def main() -> None:
    output = Path("results/paper-capacity-audit-20260919.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        tokens = case.batch * case.sequence
        intermediate = 4 * case.hidden
        cache_bytes = int(case.cache_mib * MIB)
        paper = paper_split_prediction(
            cache_bytes, tokens, case.hidden, intermediate, case.element_size
        )
        resident = resident_split_prediction(
            cache_bytes, tokens, case.hidden, intermediate, case.element_size
        )
        reported_paper = ""
        reported_resident = ""
        reported_ratio = ""
        if case.reported_k is not None:
            reported_paper = paper_working_set_bytes(
                tokens, case.hidden, intermediate, case.element_size, case.reported_k
            ) / MIB
            reported_resident = resident_working_set_bytes(
                tokens, case.hidden, intermediate, case.element_size, case.reported_k
            ) / MIB
            reported_ratio = reported_resident * MIB / cache_bytes
        rows.append(
            {
                "source": case.source,
                "model": case.model,
                "batch": case.batch,
                "sequence": case.sequence,
                "rows_m": tokens,
                "hidden_h": case.hidden,
                "intermediate_i": intermediate,
                "element_size": case.element_size,
                "cache_mib": case.cache_mib,
                "reported_k": "" if case.reported_k is None else case.reported_k,
                "paper_prediction_feasible": paper.feasible,
                "paper_continuous_k": "" if paper.continuous_k is None else paper.continuous_k,
                "paper_minimum_integer_k": "" if paper.ceiling_k is None else paper.ceiling_k,
                "resident_prediction_feasible": resident.feasible,
                "resident_continuous_k": "" if resident.continuous_k is None else resident.continuous_k,
                "resident_minimum_integer_k": "" if resident.ceiling_k is None else resident.ceiling_k,
                "paper_working_set_at_reported_k_mib": reported_paper,
                "resident_working_set_at_reported_k_mib": reported_resident,
                "resident_working_set_over_cache_at_reported_k": reported_ratio,
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["reported_k"] == "":
            continue
        print(
            "%s %-9s B=%-4s M=%-7s reported K=%-2s paper K=%-4s "
            "resident=%-10s resident/cache=%s"
            % (
                row["source"],
                row["model"],
                row["batch"],
                row["rows_m"],
                row["reported_k"],
                row["paper_minimum_integer_k"] or "infeasible",
                row["resident_minimum_integer_k"] or "infeasible",
                (
                    "%.2fx" % float(row["resident_working_set_over_cache_at_reported_k"])
                    if row["resident_working_set_over_cache_at_reported_k"] != ""
                    else ""
                ),
            )
        )


if __name__ == "__main__":
    main()
