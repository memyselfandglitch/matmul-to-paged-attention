#!/usr/bin/env python3
"""Fail fast when allocator or transparent-huge-page settings drift."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from .system_info import cache_inventory
except ImportError:
    from system_info import cache_inventory


THP_ENABLED = Path("/sys/kernel/mm/transparent_hugepage/enabled")


def selected_thp_mode(text: str) -> str | None:
    for token in text.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return None


def mapped_allocator_libraries(maps_text: str) -> list[str]:
    paths = {
        line.rsplit(maxsplit=1)[-1]
        for line in maps_text.splitlines()
        if "/" in line
        and any(name in line.lower() for name in ("tcmalloc", "jemalloc"))
    }
    return sorted(paths)


def build_report(
    require_tcmalloc: bool,
    require_thp: str | None,
    expected_l2_mib: float | None = None,
    expected_l3_mib: float | None = None,
) -> dict:
    thp_text = THP_ENABLED.read_text(encoding="utf-8").strip()
    maps_text = Path("/proc/self/maps").read_text(encoding="utf-8")
    allocators = mapped_allocator_libraries(maps_text)
    errors = []
    if require_tcmalloc and not any(
        "tcmalloc" in path.lower() for path in allocators
    ):
        errors.append("tcmalloc is required but is not mapped in this process")
    active_thp = selected_thp_mode(thp_text)
    if require_thp is not None and active_thp != require_thp:
        errors.append(
            f"transparent huge pages must be {require_thp}, found {active_thp}"
        )
    affinity = set(os.sched_getaffinity(0))
    caches = cache_inventory(affinity)
    observed_l2_mib = caches["affinity_unique_by_level_mib"].get("2", 0.0)
    if expected_l2_mib is not None and observed_l2_mib != expected_l2_mib:
        errors.append(
            f"affinity-visible aggregate L2 must be {expected_l2_mib} MiB, "
            f"found {observed_l2_mib} MiB"
        )
    observed_l3_mib = caches["affinity_unique_by_level_mib"].get("3", 0.0)
    if expected_l3_mib is not None and observed_l3_mib != expected_l3_mib:
        errors.append(
            f"affinity-visible aggregate L3 must be {expected_l3_mib} MiB, "
            f"found {observed_l3_mib} MiB"
        )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "requirements": {
            "tcmalloc": require_tcmalloc,
            "transparent_hugepage": require_thp,
            "affinity_l2_mib": expected_l2_mib,
            "affinity_l3_mib": expected_l3_mib,
        },
        "observed": {
            "ld_preload": os.environ.get("LD_PRELOAD"),
            "tcmalloc_prefix": os.environ.get("TCMALLOC_PREFIX"),
            "allocator_mappings": allocators,
            "transparent_hugepage_raw": thp_text,
            "transparent_hugepage_selected": active_thp,
            "process_affinity": sorted(affinity),
            "affinity_unique_cache_by_level_mib": caches[
                "affinity_unique_by_level_mib"
            ],
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-tcmalloc", action="store_true")
    parser.add_argument("--require-thp", choices=("always", "madvise", "never"))
    parser.add_argument("--expected-l2-mib", type=float)
    parser.add_argument("--expected-l3-mib", type=float)
    args = parser.parse_args()
    report = build_report(
        args.require_tcmalloc,
        args.require_thp,
        args.expected_l2_mib,
        args.expected_l3_mib,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if report["errors"]:
        raise SystemExit("; ".join(report["errors"]))


if __name__ == "__main__":
    main()
