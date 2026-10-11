#!/usr/bin/env python3
"""Record the exact PACE v1.0 IMBPS ownership facts visible in source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "cfbe8b551cca18c686b771144c18129242796ea1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sources(python_text: str, cpp_text: str) -> dict[str, Any]:
    facts = {
        "up_projection_split_contiguously": (
            "layer_up_proj_weight.chunk(chunks=chunks)" in python_text
        ),
        "down_projection_split_contiguously": (
            "layer.down_proj.weight.chunk(chunks=chunks, dim=1)" in python_text
        ),
        "split_loop_is_sequential": "for (int i = 0; i < iterations; ++i)" in cpp_text,
        "parallel_split_loop_is_todo": (
            "TODO: Parallelize this loop(IMBPS_PARALLEL_FLOW)" in cpp_text
        ),
        "each_block_runs_up_then_down": (
            "prim_vec.at(i).execute" in cpp_text
            and "prim_vec.at(i + 1).execute" in cpp_text
        ),
    }
    if not all(facts.values()):
        missing = sorted(key for key, value in facts.items() if not value)
        raise RuntimeError(f"PACE IMBPS source shape changed; missing facts: {missing}")
    return {
        **facts,
        "explicit_block_to_core_mapping_in_pace": False,
        "worker_tiling_owner": "oneDNN/ZenDNN matmul runtime",
        "conclusion": (
            "PACE v1.0 owns K at the sequential outer loop. It does not bind a "
            "split block to a core or private L2. The 96 MiB aggregate-private-"
            "L2 equation is therefore an optimistic placement hypothesis, not "
            "an implementation guarantee."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.pace_root.resolve()
    python_path = root / "pace/ops/backends/imbps.py"
    cpp_path = root / "csrc/ops/kernels/mlp_kernel.cpp"
    if not python_path.is_file() or not cpp_path.is_file():
        raise FileNotFoundError("PACE IMBPS source files are missing")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"expected PACE {EXPECTED_COMMIT}, found {commit}")

    payload = {
        "schema_version": 1,
        "status": "complete",
        "pace_commit": commit,
        "sources": {
            str(python_path.relative_to(root)): sha256(python_path),
            str(cpp_path.relative_to(root)): sha256(cpp_path),
        },
        "facts": inspect_sources(
            python_path.read_text(encoding="utf-8"),
            cpp_path.read_text(encoding="utf-8"),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
