#!/usr/bin/env python3
"""Compare generated PACE E2E results with Tables III and VI."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((PROJECT_ROOT / "configs" / "claims.json").read_text(encoding="utf-8"))
TABLE_III_PATTERN = re.compile(r"^(opt6\.7b|opt30b)-b([0-9]+)-(native|imbps)-k([0-9]+)$")
TABLE_VI_PATTERN = re.compile(r"^llama3\.1-8b-b([0-9]+)-(vllm|imbps)-k([0-9]+)$")


def one_result_file(result_dir: Path) -> Path:
    files = sorted(result_dir.glob("*_results.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one result JSON in {result_dir}, found {len(files)}")
    return files[0]


def framework_metrics(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["framework"]: row["metrics"]
        for row in payload.get("benchmark_results", [])
    }


def ttft(metrics: dict[str, Any]) -> float:
    value = metrics.get("average_ttft")
    if value is None:
        raise ValueError("average_ttft is missing; token_metrics.time_to_first_token must be true")
    return float(value)


def summarize_table_iii(suite_dir: Path) -> list[dict[str, Any]]:
    measured: dict[tuple[str, int, str], tuple[int, float, Path]] = {}
    for case_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        match = TABLE_III_PATTERN.match(case_dir.name)
        if not match:
            continue
        model, batch_text, backend, split_text = match.groups()
        result = one_result_file(case_dir / "results")
        metrics = framework_metrics(result)
        measured[(model, int(batch_text), backend)] = (
            int(split_text),
            ttft(metrics["pace"]),
            result,
        )

    rows = []
    targets = CLAIMS["e2e"]["table_iii"]["models"]
    for model, batches in targets.items():
        for batch_text, (target_improvement, target_split) in batches.items():
            batch = int(batch_text)
            _, baseline, baseline_file = measured[(model, batch, "native")]
            split, candidate, candidate_file = measured[(model, batch, "imbps")]
            improvement = (baseline - candidate) / baseline * 100
            rows.append(
                {
                    "suite": "table_iii",
                    "model": model,
                    "batch": batch,
                    "split": split,
                    "baseline": "PACE v1.0 NATIVE FP32",
                    "baseline_ttft_seconds": baseline,
                    "candidate_ttft_seconds": candidate,
                    "measured_improvement_percent": improvement,
                    "paper_improvement_percent": target_improvement,
                    "paper_split": target_split,
                    "difference_percentage_points": improvement - target_improvement,
                    "baseline_result": str(baseline_file.resolve()),
                    "candidate_result": str(candidate_file.resolve()),
                }
            )
    return rows


def summarize_table_vi(suite_dir: Path) -> list[dict[str, Any]]:
    measured: dict[tuple[int, str], tuple[int, float, Path]] = {}
    targets = CLAIMS["e2e"]["table_vi"]["targets"]
    for case_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        match = TABLE_VI_PATTERN.match(case_dir.name)
        if not match:
            continue
        batch_text, backend, split_text = match.groups()
        batch = int(batch_text)
        result = one_result_file(case_dir / "results")
        metrics = framework_metrics(result)
        framework = "vllm" if backend == "vllm" else "pace"
        measured[(batch, backend)] = (int(split_text), ttft(metrics[framework]), result)

    rows = []
    for batch_text, target in targets.items():
        batch = int(batch_text)
        _, baseline, baseline_file = measured[(batch, "vllm")]
        split, candidate, candidate_file = measured[(batch, "imbps")]
        improvement = (baseline - candidate) / baseline * 100
        rows.append(
            {
                "suite": "table_vi",
                "model": "llama3.1-8b",
                "batch": batch,
                "split": split,
                "baseline": "vLLM 0.8.4",
                "baseline_ttft_seconds": baseline,
                "candidate_ttft_seconds": candidate,
                "measured_improvement_percent": improvement,
                "paper_improvement_percent": target,
                "paper_split": 2,
                "difference_percentage_points": improvement - target,
                "baseline_result": str(baseline_file.resolve()),
                "candidate_result": str(candidate_file.resolve()),
            }
        )
    if len(rows) != len(targets):
        raise RuntimeError(f"expected {len(targets)} Table VI rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("table_iii", "table_vi"), required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = (
        summarize_table_iii(args.suite_dir)
        if args.suite == "table_iii"
        else summarize_table_vi(args.suite_dir)
    )
    if not rows:
        raise SystemExit("no results found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
