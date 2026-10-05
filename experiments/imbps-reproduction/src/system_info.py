#!/usr/bin/env python3
"""Capture the hardware and software facts needed to interpret an IMBPS run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "IMBPS_BLOCK_SIZE",
    "LIBXSMM_BLOCK_SIZE",
    "DNNL_MAX_CPU_ISA",
    "DNNL_VERBOSE",
    "MALLOC_CONF",
    "LD_PRELOAD",
    "TCMALLOC_PREFIX",
    "REQUIRE_TCMALLOC",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_CPU_BIND",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def parse_cache_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMG])", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported sysfs cache size: {value!r}")
    number = int(match.group(1))
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return number * multiplier


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            cpus.update(range(int(start_text), int(end_text) + 1))
        else:
            cpus.add(int(item))
    return cpus


def cache_inventory(affinity: set[int] | None = None) -> dict[str, Any]:
    entries: dict[tuple[str, ...], dict[str, Any]] = {}
    for index_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index*")):
        try:
            level = (index_dir / "level").read_text().strip()
            cache_type = (index_dir / "type").read_text().strip()
            size_text = (index_dir / "size").read_text().strip()
            shared = (index_dir / "shared_cpu_list").read_text().strip()
            cache_id_path = index_dir / "id"
            cache_id = cache_id_path.read_text().strip() if cache_id_path.exists() else "unknown"
            key = (level, cache_type, cache_id, shared, size_text)
            entries[key] = {
                "level": int(level),
                "type": cache_type,
                "id": cache_id,
                "shared_cpu_list": shared,
                "size_text": size_text,
                "size_bytes": parse_cache_size(size_text),
            }
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    unique = sorted(entries.values(), key=lambda item: (item["level"], item["type"], item["id"], item["shared_cpu_list"]))
    l3 = [item for item in unique if item["level"] == 3]
    affinity_l3 = (
        [item for item in l3 if parse_cpu_list(item["shared_cpu_list"]) & affinity]
        if affinity is not None
        else []
    )
    levels = sorted({item["level"] for item in unique})
    aggregate_by_level_mib = {
        str(level): sum(
            item["size_bytes"] for item in unique if item["level"] == level
        )
        / 1024**2
        for level in levels
    }
    affinity_by_level_mib = (
        {
            str(level): sum(
                item["size_bytes"]
                for item in unique
                if item["level"] == level
                and parse_cpu_list(item["shared_cpu_list"]) & affinity
            )
            / 1024**2
            for level in levels
        }
        if affinity is not None
        else None
    )
    return {
        "unique_entries": unique,
        "aggregate_unique_by_level_mib": aggregate_by_level_mib,
        "affinity_unique_by_level_mib": affinity_by_level_mib,
        "unique_l3_count": len(l3),
        "aggregate_unique_l3_bytes": sum(item["size_bytes"] for item in l3),
        "aggregate_unique_l3_mib": sum(item["size_bytes"] for item in l3) / 1024**2,
        "affinity_unique_l3_count": len(affinity_l3) if affinity is not None else None,
        "affinity_unique_l3_bytes": (
            sum(item["size_bytes"] for item in affinity_l3)
            if affinity is not None
            else None
        ),
        "affinity_unique_l3_mib": (
            sum(item["size_bytes"] for item in affinity_l3) / 1024**2
            if affinity is not None
            else None
        ),
    }


def read_if_present(path: str) -> str | None:
    candidate = Path(path)
    try:
        return candidate.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "transformers", "pace", "lm_eval", "datasets", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def build_report() -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    affinity_set = set(affinity) if affinity is not None else None
    pace_root = os.environ.get("PACE_ROOT")
    pace_git = None
    if pace_root:
        pace_git = {
            "commit": command_output(["git", "-C", pace_root, "rev-parse", "HEAD"]),
            "tracked_status": command_output(
                ["git", "-C", pace_root, "status", "--short", "--untracked-files=no"]
            ),
        }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": os.path.realpath(os.sys.executable),
        "cpu_affinity": affinity,
        "environment": {key: os.environ.get(key) for key in SAFE_ENV_KEYS},
        "sysfs": {
            "perf_event_paranoid": read_if_present("/proc/sys/kernel/perf_event_paranoid"),
            "thp_enabled": read_if_present("/sys/kernel/mm/transparent_hugepage/enabled"),
            "thp_defrag": read_if_present("/sys/kernel/mm/transparent_hugepage/defrag"),
            "scaling_governor_cpu0": read_if_present("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "smt_active": read_if_present("/sys/devices/system/cpu/smt/active"),
        },
        "cache": cache_inventory(affinity_set),
        "packages": package_versions(),
        "pace_git": pace_git,
        "commands": {
            "lscpu": command_output(["lscpu", "-e=CPU,NODE,SOCKET,CORE,ONLINE,MAXMHZ,MINMHZ"]),
            "lscpu_summary": command_output(["lscpu"]),
            "numactl": command_output(["numactl", "--hardware"]),
            "uname": command_output(["uname", "-a"]),
            "os_release": command_output(["sh", "-c", "cat /etc/os-release"]),
            "free": command_output(["free", "-h"]),
            "gcc": command_output(["gcc", "--version"]),
            "perf": command_output(["perf", "--version"]),
            "slurm": command_output(["scontrol", "show", "job", os.environ.get("SLURM_JOB_ID", "")]) if os.environ.get("SLURM_JOB_ID") else {"available": False, "error": "not in a Slurm job"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_report(), indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
