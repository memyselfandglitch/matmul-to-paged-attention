#!/usr/bin/env python3
"""Select complete last-level-cache CPU groups from the current affinity."""

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


def discover_groups(allowed: set[int]) -> list[tuple[int, ...]]:
    groups = {llc_group_for_cpu(cpu, allowed) for cpu in sorted(allowed)}
    return sorted(groups, key=lambda group: (min(group), group))


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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    allowed = set(os.sched_getaffinity(0))
    groups = discover_groups(allowed)
    selected_groups = choose_groups(groups, args.threads)
    selected = [cpu for group in selected_groups for cpu in group]
    if args.json:
        print(
            json.dumps(
                {
                    "allowed_cpus": sorted(allowed),
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
