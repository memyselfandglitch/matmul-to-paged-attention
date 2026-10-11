#!/usr/bin/env python3
"""Select complete last-level-cache CPU groups from the current affinity.

Some AMD kernels expose a socket-wide ``shared_cpu_list`` for L3 even though
the processor contains several physical CCD/L3 instances.  For experiments
that need CCD boundaries, use the per-cache ID and CPU topology labels as
independent discovery sources and require an exact topology match.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SYS_CPU = Path("/sys/devices/system/cpu")


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        if "-" in field:
            start_text, end_text = field.split("-", 1)
            cpus.update(range(int(start_text), int(end_text) + 1))
        else:
            cpus.add(int(field))
    return cpus


def llc_group_for_cpu(cpu: int, allowed: set[int]) -> tuple[int, ...]:
    cache_root = SYS_CPU / f"cpu{cpu}" / "cache"
    for index_path in sorted(cache_root.glob("index*")):
        try:
            level = (index_path / "level").read_text(encoding="utf-8").strip()
            cache_type = (index_path / "type").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if level != "3" or cache_type not in {"Unified", "Data"}:
            continue
        shared = parse_cpu_list(
            (index_path / "shared_cpu_list").read_text(encoding="utf-8")
        )
        group = tuple(sorted(shared & allowed))
        if group:
            return group
    raise RuntimeError(f"no L3 cache-sharing group found for CPU {cpu}")


def l3_index_for_cpu(cpu: int) -> Path:
    cache_root = SYS_CPU / f"cpu{cpu}" / "cache"
    for index_path in sorted(cache_root.glob("index*")):
        try:
            level = (index_path / "level").read_text(encoding="utf-8").strip()
            cache_type = (index_path / "type").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if level == "3" and cache_type in {"Unified", "Data"}:
            return index_path
    raise RuntimeError(f"no L3 cache index found for CPU {cpu}")


def groups_from_labels(labels: dict[int, tuple[str, ...]]) -> list[tuple[int, ...]]:
    grouped: dict[tuple[str, ...], list[int]] = {}
    for cpu, label in labels.items():
        grouped.setdefault(label, []).append(cpu)
    return sorted(
        (tuple(sorted(cpus)) for cpus in grouped.values()),
        key=lambda group: (min(group), group),
    )


def candidate_groupings(allowed: set[int]) -> dict[str, list[tuple[int, ...]]]:
    """Return independent hardware-topology interpretations for ``allowed``."""

    candidates: dict[str, list[tuple[int, ...]]] = {}

    try:
        shared_groups = {llc_group_for_cpu(cpu, allowed) for cpu in sorted(allowed)}
        candidates["l3_shared_cpu_list"] = sorted(
            shared_groups, key=lambda group: (min(group), group)
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    label_specs = {
        "l3_cache_id": ("cache_id",),
        "die_id": ("die_id",),
        "cluster_id": ("die_id", "cluster_id"),
    }
    for source, fields in label_specs.items():
        labels: dict[int, tuple[str, ...]] = {}
        try:
            for cpu in sorted(allowed):
                package = (
                    SYS_CPU
                    / f"cpu{cpu}"
                    / "topology"
                    / "physical_package_id"
                ).read_text(encoding="utf-8").strip()
                values = [package]
                for field in fields:
                    if field == "cache_id":
                        path = l3_index_for_cpu(cpu) / "id"
                    else:
                        path = SYS_CPU / f"cpu{cpu}" / "topology" / field
                    values.append(path.read_text(encoding="utf-8").strip())
                labels[cpu] = tuple(values)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        candidates[source] = groups_from_labels(labels)

    return candidates


def grouping_diagnostics(
    candidates: dict[str, list[tuple[int, ...]]],
) -> dict[str, dict[str, object]]:
    return {
        source: {
            "group_count": len(groups),
            "group_sizes": [len(group) for group in groups],
            "groups": [list(group) for group in groups],
        }
        for source, groups in candidates.items()
    }


def select_grouping(
    candidates: dict[str, list[tuple[int, ...]]],
    expected_groups: int | None = None,
    expected_group_size: int | None = None,
) -> tuple[list[tuple[int, ...]], str]:
    if not candidates:
        raise RuntimeError("no CPU cache/topology grouping source is available")

    preferred_sources = (
        "l3_cache_id",
        "l3_shared_cpu_list",
        "die_id",
        "cluster_id",
    )
    for source in preferred_sources:
        groups = candidates.get(source)
        if groups is None:
            continue
        if expected_groups is not None and len(groups) != expected_groups:
            continue
        if expected_group_size is not None and any(
            len(group) != expected_group_size for group in groups
        ):
            continue
        return groups, source

    expected = []
    if expected_groups is not None:
        expected.append(f"{expected_groups} groups")
    if expected_group_size is not None:
        expected.append(f"{expected_group_size} CPUs/group")
    detail = ", ".join(
        f"{source}={len(groups)} groups with sizes "
        f"{[len(group) for group in groups]}"
        for source, groups in candidates.items()
    )
    raise RuntimeError(
        f"no topology source matched {' and '.join(expected) or 'the request'}; "
        f"candidates: {detail}"
    )


def discover_groups_with_source(
    allowed: set[int],
    expected_groups: int | None = None,
    expected_group_size: int | None = None,
) -> tuple[list[tuple[int, ...]], str, dict[str, dict[str, object]]]:
    candidates = candidate_groupings(allowed)
    groups, source = select_grouping(
        candidates,
        expected_groups=expected_groups,
        expected_group_size=expected_group_size,
    )
    return groups, source, grouping_diagnostics(candidates)


def discover_groups(allowed: set[int]) -> list[tuple[int, ...]]:
    groups, _, _ = discover_groups_with_source(allowed)
    return groups


def choose_groups(groups: list[tuple[int, ...]], threads: int) -> list[tuple[int, ...]]:
    if threads <= 0:
        raise ValueError("threads must be positive")
    selected: list[tuple[int, ...]] = []
    selected_count = 0
    for group in groups:
        if selected_count == threads:
            break
        if selected_count + len(group) > threads:
            raise ValueError(
                f"threads={threads} would split LLC group {list(group)}; "
                "use a complete-CCD thread count"
            )
        selected.append(group)
        selected_count += len(group)
    if selected_count != threads:
        raise ValueError(
            f"requested {threads} CPUs but only {selected_count} CPUs were "
            "available in complete LLC groups"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--expected-groups", type=int)
    parser.add_argument("--expected-group-size", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    allowed = set(os.sched_getaffinity(0))
    groups, source, diagnostics = discover_groups_with_source(
        allowed,
        expected_groups=args.expected_groups,
        expected_group_size=args.expected_group_size,
    )
    selected_groups = choose_groups(groups, args.threads)
    selected = [cpu for group in selected_groups for cpu in group]
    if args.json:
        print(
            json.dumps(
                {
                    "allowed_cpus": sorted(allowed),
                    "selected_topology_source": source,
                    "topology_candidates": diagnostics,
                    "llc_groups": [list(group) for group in groups],
                    "selected_groups": [list(group) for group in selected_groups],
                    "selected_cpus": selected,
                },
                indent=2,
            )
        )
    else:
        print(",".join(str(cpu) for cpu in selected))


if __name__ == "__main__":
    main()
