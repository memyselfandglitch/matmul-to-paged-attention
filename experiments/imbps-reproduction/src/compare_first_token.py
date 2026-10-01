#!/usr/bin/env python3
"""Compare baseline and IMBPS first-token log-probability dumps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_metadata(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata sidecar: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    baseline = np.load(args.baseline)
    candidate = np.load(args.candidate)
    baseline_metadata = load_metadata(args.baseline)
    candidate_metadata = load_metadata(args.candidate)
    for key in ("model", "dtype", "samples", "sequence", "batch_size", "seed"):
        if baseline_metadata.get(key) != candidate_metadata.get(key):
            raise ValueError(f"baseline and candidate metadata differ for {key}")
    baseline_revision = baseline_metadata.get("model_revision")
    candidate_revision = candidate_metadata.get("model_revision")
    if baseline_revision != candidate_revision:
        raise ValueError("baseline and candidate model revisions differ")
    if not np.array_equal(baseline["input_ids"], candidate["input_ids"]):
        raise ValueError("baseline and candidate input IDs differ")
    base_logits = baseline["last_logprobs"].astype(np.float32, copy=False)
    candidate_logits = candidate["last_logprobs"].astype(np.float32, copy=False)
    if base_logits.shape != candidate_logits.shape:
        raise ValueError("baseline and candidate log-probability shapes differ")

    absolute = np.abs(candidate_logits - base_logits)
    baseline_top1 = base_logits.argmax(axis=1)
    candidate_top1 = candidate_logits.argmax(axis=1)
    candidate_top5 = np.argpartition(candidate_logits, -5, axis=1)[:, -5:]
    baseline_in_top5 = (candidate_top5 == baseline_top1[:, None]).any(axis=1)
    generated_equal = baseline["generated_tokens"] == candidate["generated_tokens"]
    report = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "model": baseline_metadata["model"],
        "model_revision": baseline_revision,
        "model_revision_verified": baseline_revision is not None,
        "samples": int(base_logits.shape[0]),
        "vocabulary": int(base_logits.shape[1]),
        "max_abs_logprob_difference": float(absolute.max()),
        "mean_abs_logprob_difference": float(absolute.mean()),
        "q50_abs_logprob_difference": float(np.quantile(absolute, 0.50)),
        "q90_abs_logprob_difference": float(np.quantile(absolute, 0.90)),
        "q99_abs_logprob_difference": float(np.quantile(absolute, 0.99)),
        "q999_abs_logprob_difference": float(np.quantile(absolute, 0.999)),
        "fraction_abs_le_1e_4": float((absolute <= 1e-4).mean()),
        "fraction_abs_le_1e_2": float((absolute <= 1e-2).mean()),
        "top1_agreement": float((baseline_top1 == candidate_top1).mean()),
        "baseline_top1_in_candidate_top5": float(baseline_in_top5.mean()),
        "generated_token_agreement": float(generated_equal.mean()),
        "changed_generated_tokens": int((~generated_equal).sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
