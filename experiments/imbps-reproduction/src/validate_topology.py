#!/usr/bin/env python3
"""Validate that a benchmark allocation is one physical CPU socket without SMT."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_lscpu_csv(text: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"unexpected lscpu row: {line!r}")
        cpu, core, socket, node = (int(field) for field in fields)
        rows.append({"cpu": cpu, "core": core, "socket": socket, "node": node})
    if not rows:
        raise ValueError("lscpu returned no topology rows")
    return rows


def evaluate(
    rows: list[dict[str, int]],
    affinity: set[int],
    expected_physical_cores: int | None,
    require_single_socket: bool,
    reject_smt: bool,
) -> dict[str, Any]:
    selected = [row for row in rows if row["cpu"] in affinity]
    mapped = {row["cpu"] for row in selected}
    errors: list[str] = []
    if mapped != affinity:
        errors.append(f"affinity CPUs absent from lscpu output: {sorted(affinity - mapped)}")
    sockets = sorted({row["socket"] for row in selected})
    nodes = sorted({row["node"] for row in selected})
    physical_cores = {(row["socket"], row["core"]) for row in selected}
    smt_in_affinity = len(selected) != len(physical_cores)
    if require_single_socket and len(sockets) != 1:
        errors.append(f"expected one socket, found {sockets}")
    if reject_smt and smt_in_affinity:
        errors.append(
            f"SMT siblings are present: {len(selected)} logical CPUs for "
            f"{len(physical_cores)} physical cores"
        )
    if expected_physical_cores is not None and len(physical_cores) != expected_physical_cores:
        errors.append(
            f"expected {expected_physical_cores} physical cores, found {len(physical_cores)}"
        )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "affinity_logical_cpus": sorted(affinity),
        "logical_cpu_count": len(selected),
        "physical_core_count": len(physical_cores),
        "sockets": sockets,
        "numa_nodes": nodes,
        "smt_in_affinity": smt_in_affinity,
        "requirements": {
            "expected_physical_cores": expected_physical_cores,
            "require_single_socket": require_single_socket,
            "reject_smt": reject_smt,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-physical-cores", type=int, default=None)
    parser.add_argument("--require-single-socket", action="store_true")
    parser.add_argument("--reject-smt", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        ["lscpu", "--parse=CPU,CORE,SOCKET,NODE"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    report = evaluate(
        parse_lscpu_csv(completed.stdout),
        set(os.sched_getaffinity(0)),
        args.expected_physical_cores,
        args.require_single_socket,
        args.reject_smt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if report["status"] != "pass":
        raise SystemExit("; ".join(report["errors"]))


if __name__ == "__main__":
    main()
