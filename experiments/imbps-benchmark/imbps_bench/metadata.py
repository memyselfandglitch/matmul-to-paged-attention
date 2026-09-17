"""System and software metadata capture for benchmark provenance."""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "DNNL_VERBOSE",
    "ONEDNN_VERBOSE",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "GOMP_CPU_AFFINITY",
    "MALLOC_CONF",
    "LD_PRELOAD",
    "PYTHONHASHSEED",
)


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _command_output(command: List[str]) -> Optional[str]:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _cpu_affinity() -> Optional[List[int]]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _cache_domains() -> List[Dict[str, Any]]:
    domains: Dict[str, Dict[str, Any]] = {}
    cpu_root = Path("/sys/devices/system/cpu")
    for cpu_path in sorted(cpu_root.glob("cpu[0-9]*")):
        for index_path in sorted((cpu_path / "cache").glob("index*")):
            level = _read_text(index_path / "level")
            cache_type = _read_text(index_path / "type")
            shared = _read_text(index_path / "shared_cpu_list")
            if level is None or cache_type is None or shared is None:
                continue
            key = "%s:%s:%s" % (level, cache_type, shared)
            if key in domains:
                continue
            domains[key] = {
                "level": int(level),
                "type": cache_type,
                "size": _read_text(index_path / "size"),
                "line_size": _read_text(index_path / "coherency_line_size"),
                "ways": _read_text(index_path / "ways_of_associativity"),
                "sets": _read_text(index_path / "number_of_sets"),
                "shared_cpu_list": shared,
            }
    return sorted(
        domains.values(),
        key=lambda item: (item["level"], item["type"], item["shared_cpu_list"]),
    )


def _unique_sysfs_values(pattern: str) -> List[str]:
    values = set()
    for path in Path("/").glob(pattern.lstrip("/")):
        value = _read_text(path)
        if value is not None:
            values.add(value)
    return sorted(values)


def collect_metadata(output_path: Optional[Path] = None, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        torch_config = torch.__config__.show()
    except (AttributeError, RuntimeError):
        torch_config = None
    try:
        parallel_info = torch.__config__.parallel_info()
    except (AttributeError, RuntimeError):
        parallel_info = None

    metadata: Dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "torch": {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "config": torch_config,
            "parallel_info": parallel_info,
        },
        "process": {
            "pid": os.getpid(),
            "cpu_affinity": _cpu_affinity(),
            "cwd": os.getcwd(),
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS if key in os.environ},
        "commands": {
            "lscpu": _command_output(["lscpu"]),
            "lscpu_cache": _command_output(["lscpu", "-C"]),
            "numactl_hardware": _command_output(["numactl", "--hardware"]),
            "free": _command_output(["free", "-h"]),
            "perf_version": _command_output(["perf", "--version"]),
            "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        },
        "tools": {
            name: shutil.which(name)
            for name in ("perf", "numactl", "taskset", "AMDuProfCLI", "AMDuProfPcm")
        },
        "sysfs": {
            "cache_domains": _cache_domains(),
            "smt_active": _read_text(Path("/sys/devices/system/cpu/smt/active")),
            "governors": _unique_sysfs_values(
                "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"
            ),
            "transparent_hugepage_enabled": _read_text(
                Path("/sys/kernel/mm/transparent_hugepage/enabled")
            ),
            "perf_event_paranoid": _read_text(Path("/proc/sys/kernel/perf_event_paranoid")),
        },
        "arguments": arguments or {},
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata

