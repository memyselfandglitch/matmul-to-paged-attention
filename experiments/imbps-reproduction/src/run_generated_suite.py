#!/usr/bin/env python3
"""Execute generated PACE configs sequentially with no automatic retry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_COMMIT = "cfbe8b551cca18c686b771144c18129242796ea1"
RUNNER = Path(__file__).resolve().parent / "run_pace_entrypoint.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pace-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=86400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pace_root = args.pace_root.resolve()
    resolved = subprocess.run(
        ["git", "-C", str(pace_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if resolved != EXPECTED_COMMIT:
        raise SystemExit(f"PACE commit mismatch: expected {EXPECTED_COMMIT}, found {resolved}")
    for diff_args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        tracked_diff = subprocess.run(
            ["git", "-C", str(pace_root), *diff_args], check=False
        )
        if tracked_diff.returncode != 0:
            raise SystemExit("PACE checkout has tracked changes; refusing mixed provenance")

    for case in manifest["cases"]:
        if case["kind"] == "performance":
            entrypoint = pace_root / "benchmarks" / "llm" / "performance" / "benchmark_llm_offline.py"
            config_flag = "--config"
        else:
            entrypoint = pace_root / "benchmarks" / "llm" / "accuracy" / "evaluation.py"
            config_flag = "--config"
        log_path = Path(case["config"]).parent / "run.log"
        environment = os.environ.copy()
        environment["IMBPS_BLOCK_SIZE"] = str(case["split"])
        seed = manifest.get("assumptions", {}).get("input_seed", 0)
        environment["PYTHONHASHSEED"] = str(seed)
        command = [
            sys.executable,
            str(RUNNER),
            "--entrypoint",
            str(entrypoint),
            config_flag,
            case["config"],
            "--seed",
            str(seed),
        ]
        print(f"running {case['case_id']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=str(entrypoint.parent),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{case['case_id']} failed with exit code {completed.returncode}; "
                f"no retry was attempted; see {log_path}"
            )
    print("suite complete")


if __name__ == "__main__":
    main()
