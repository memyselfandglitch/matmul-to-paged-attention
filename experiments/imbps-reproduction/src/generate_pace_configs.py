#!/usr/bin/env python3
"""Generate explicit AMD-PACE v1.0 configs for paper E2E and MMLU claims."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((PROJECT_ROOT / "configs" / "claims.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("table_iii", "table_vi", "mmlu"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--mmlu-fewshot", type=int, default=5)
    parser.add_argument("--input-seed", type=int, default=0)
    return parser.parse_args()


def operators(mlp_backend: str) -> dict[str, str]:
    return {
        "Norm": "NATIVE",
        "QKVProjection": "TPP",
        "Attention": "JIT",
        "OutProjection": "TPP",
        "MLP": mlp_backend,
        "LMHead": "TPP",
    }


def performance_config(
    model_id: str,
    dtype: str,
    batch: int,
    sequence: int,
    mlp_backend: str,
    output_dir: Path,
    output_tokens: int,
    warmup_runs: int,
    num_runs: int,
    use_real_data: bool,
    input_seed: int,
    frameworks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "frameworks": frameworks or ["pace"],
        "model_args": {
            "model_name": model_id,
            "dtype": dtype,
            "llm_operators": operators(mlp_backend),
            "spec_config": None,
        },
        "use_real_data": use_real_data,
        "generation_args": {
            "input_tokens": sequence,
            "output_tokens": output_tokens,
            "batch_size": batch,
            "num_beams": 1,
            "kv_cache_type": "BMC",
            "do_sample": False,
            "manual_seed": input_seed,
        },
        "warmup_runs": warmup_runs,
        "num_runs": num_runs,
        "visualize": False,
        "verbose": True,
        "output_dir": str(output_dir),
        "token_metrics": {"time_to_first_token": True, "time_per_tokens": False},
        "system_metrics": False,
    }


def write_case(
    root: Path,
    case_id: str,
    config: dict[str, Any],
    split: int,
    kind: str,
    cases: list[dict[str, Any]],
) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    config_path = case_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    cases.append(
        {
            "case_id": case_id,
            "kind": kind,
            "config": str(config_path.resolve()),
            "split": split,
            "result_dir": config["output_dir"],
        }
    )


def generate_table_iii(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[str]:
    claim = CLAIMS["e2e"]["table_iii"]
    caveats = [
        "PACE v1.0 has no TPP FP32 registration. Generated K=1 baselines use NATIVE, not the TPP baseline named in the author email.",
        "The paper does not state output-token count or dataset; defaults are explicit assumptions.",
    ]
    for model_name, batches in claim["models"].items():
        model_id = CLAIMS["models"][model_name]["hf_id"]
        for batch_text, (_, split) in batches.items():
            batch = int(batch_text)
            baseline_id = f"{model_name}-b{batch}-native-k1"
            baseline_results = (args.output_dir / baseline_id / "results").resolve()
            write_case(
                args.output_dir,
                baseline_id,
                performance_config(
                    model_id,
                    claim["dtype"],
                    batch,
                    claim["sequence"],
                    "NATIVE",
                    baseline_results,
                    args.output_tokens,
                    args.warmup_runs,
                    args.num_runs,
                    args.use_real_data,
                    args.input_seed,
                ),
                1,
                "performance",
                cases,
            )
            imbps_id = f"{model_name}-b{batch}-imbps-k{split}"
            imbps_results = (args.output_dir / imbps_id / "results").resolve()
            write_case(
                args.output_dir,
                imbps_id,
                performance_config(
                    model_id,
                    claim["dtype"],
                    batch,
                    claim["sequence"],
                    "IMBPS",
                    imbps_results,
                    args.output_tokens,
                    args.warmup_runs,
                    args.num_runs,
                    args.use_real_data,
                    args.input_seed,
                ),
                split,
                "performance",
                cases,
            )
    return caveats


def generate_table_vi(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[str]:
    claim = CLAIMS["e2e"]["table_vi"]
    model_id = CLAIMS["models"][claim["model"]]["hf_id"]
    for batch_text in claim["targets"]:
        batch = int(batch_text)
        baseline_id = f"llama3.1-8b-b{batch}-vllm-k1"
        baseline_results = (args.output_dir / baseline_id / "results").resolve()
        write_case(
            args.output_dir,
            baseline_id,
            performance_config(
                model_id,
                claim["dtype"],
                batch,
                claim["sequence"],
                "TPP",
                baseline_results,
                args.output_tokens,
                args.warmup_runs,
                args.num_runs,
                args.use_real_data,
                args.input_seed,
                frameworks=["vllm"],
            ),
            1,
            "performance",
            cases,
        )
        candidate_id = f"llama3.1-8b-b{batch}-imbps-k{claim['split']}"
        candidate_results = (args.output_dir / candidate_id / "results").resolve()
        write_case(
            args.output_dir,
            candidate_id,
            performance_config(
                model_id,
                claim["dtype"],
                batch,
                claim["sequence"],
                "IMBPS",
                candidate_results,
                args.output_tokens,
                args.warmup_runs,
                args.num_runs,
                args.use_real_data,
                args.input_seed,
                frameworks=["pace"],
            ),
            claim["split"],
            "performance",
            cases,
        )
    return [
        "Requires vLLM 0.8.4 CPU support in the same environment.",
        "Baseline and candidate run in separate seeded processes so PACE v1.0's Python-random input generator produces the same token IDs.",
    ]


def generate_mmlu(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[str]:
    claim = CLAIMS["mmlu"]
    for model_name in claim["targets"]:
        model_id = CLAIMS["models"][model_name]["hf_id"]
        for split in claim["splits"]:
            backend = "TPP" if split == 1 else "IMBPS"
            case_id = f"{model_name}-{backend.lower()}-k{split}"
            result_dir = (args.output_dir / case_id / "results").resolve()
            config = {
                "model_args": {
                    "model_name": model_id,
                    "tokenizer_name": model_id,
                    "dtype": "bf16",
                    "llm_operators": operators(backend),
                    "spec_config": None,
                },
                "generation_args": {
                    "batch_size": 1,
                    "num_beams": 1,
                    "kv_cache_type": "BMC",
                },
                "tasks": [
                    {
                        "task_name": "mmlu",
                        "num_fewshot": args.mmlu_fewshot,
                        "limit": None,
                    }
                ],
                "verbose": True,
                "output_dir": str(result_dir),
            }
            write_case(args.output_dir, case_id, config, split, "accuracy", cases)
    return [
        "The paper does not state MMLU few-shot count. The generated value is an assumption.",
        "Run every K with the same lm-eval 0.4.7 task data and model revision cache.",
    ]


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases: list[dict[str, Any]] = []
    if args.suite == "table_iii":
        caveats = generate_table_iii(args, cases)
    elif args.suite == "table_vi":
        caveats = generate_table_vi(args, cases)
    else:
        caveats = generate_mmlu(args, cases)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "paper": CLAIMS["paper"],
        "pace": CLAIMS["pace"],
        "assumptions": {
            "output_tokens": args.output_tokens if args.suite != "mmlu" else None,
            "warmup_runs": args.warmup_runs if args.suite != "mmlu" else None,
            "num_runs": args.num_runs if args.suite != "mmlu" else None,
            "use_real_data": args.use_real_data if args.suite != "mmlu" else None,
            "input_seed": args.input_seed,
            "mmlu_num_fewshot": args.mmlu_fewshot if args.suite == "mmlu" else None,
        },
        "caveats": caveats,
        "cases": cases,
    }
    (args.output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "run-manifest.json")


if __name__ == "__main__":
    main()
