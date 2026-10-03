#!/usr/bin/env python3
"""Parse cumulative metric tables emitted by AMDuProfPcm 5.3."""

from __future__ import annotations

import csv
from pathlib import Path


SECTION_HEADERS = {
    "CORE METRICS": "core",
    "L3 METRICS": "l3",
    "DF METRICS": "df",
}


def parse_uprof_report(path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line in SECTION_HEADERS:
            current = SECTION_HEADERS[line]
            sections.setdefault(current, {})
            continue
        if not line or current is None:
            continue
        fields = next(csv.reader([line]))
        if len(fields) != 2 or fields[0] == "Metric":
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        sections[current][fields[0].strip()] = value
    return sections


def require_metric(
    report: dict[str, dict[str, float]], section: str, metric: str
) -> float:
    try:
        return report[section][metric]
    except KeyError as error:
        raise ValueError(f"missing uProf metric {section}/{metric}") from error
