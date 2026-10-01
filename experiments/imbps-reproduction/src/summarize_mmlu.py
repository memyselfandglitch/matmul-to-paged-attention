#!/usr/bin/env python3
"""Extract average MMLU accuracy from generated PACE result directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((PROJECT_ROOT / "configs" / "claims.json").read_text(encoding="utf-8"))
CASE_PATTERN = re.compile(r"^(opt125m|opt30b)-(?:tpp|imbps)-k([0-9]+)$")


def find_accuracy(results: dict[str, Any]) -> float:
    task_results = results.get("results", {})
    preferred = task_results.get("mmlu")
    if isinstance(preferred, dict):
        for key in ("acc,none", "acc_norm,none", "acc"):
            if key in preferred:
                return float(preferred[key])
    candidates: list[float] = []
    for value in task_results.values():
        if not isinstance(value, dict):
            continue
        for key in ("acc,none", "acc_norm,none", "acc"):
            if key in value:
                candidates.append(float(value[key]))
                break
    if candidates:
        return sum(candidates) / len(candidates)
    raise ValueError("could not find MMLU accuracy in lm-eval JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case_dir in sorted(path for path in args.suite_dir.iterdir() if path.is_dir()):
        match = CASE_PATTERN.match(case_dir.name)
        if not match:
            continue
        model, split_text = match.groups()
        split = int(split_text)
        result_files = sorted((case_dir / "results").glob("*_eval_results.json"))
        if len(result_files) != 1:
            raise RuntimeError(f"expected one result for {case_dir.name}, found {len(result_files)}")
        payload = json.loads(result_files[0].read_text(encoding="utf-8"))
        measured = find_accuracy(payload)
        target = CLAIMS["mmlu"]["targets"][model][str(split)]
        rows.append(
            {
                "model": model,
                "split": split,
                "measured_accuracy": measured,
                "paper_accuracy": target,
                "absolute_difference": measured - target,
                "result_file": str(result_files[0].resolve()),
            }
        )
    if not rows:
        raise SystemExit("no generated MMLU result directories found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()

