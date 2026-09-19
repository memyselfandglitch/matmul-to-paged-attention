#!/usr/bin/env python3
"""Audit IMBPS numerical drift across OPT layers and deterministic inputs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from imbps_bench.hf_opt import HFOPTReferenceMLP, OPTModelIMBPSPatcher, find_opt_decoder_layers
from imbps_bench.runner import DTYPES, error_metrics, tolerances


MODEL_FIELDS = (
    "seed",
    "split_k",
    "dtype",
    "accumulation_dtype",
    "max_abs_logits",
    "mean_abs_logits",
    "p99_abs_logits",
    "max_rel_logits",
    "logits_allclose",
    "first_token_match",
    "reference_first_token_id",
    "imbps_first_token_id",
    "reference_top1_margin",
    "imbps_top1_margin",
)
LAYER_FIELDS = (
    "seed",
    "layer_index",
    "split_k",
    "dtype",
    "accumulation_dtype",
    "max_abs_output",
    "mean_abs_output",
    "p99_abs_output",
    "max_rel_output",
    "output_allclose",
)
SUMMARY_FIELDS = (
    "split_k",
    "dtype",
    "accumulation_dtype",
    "seeds",
    "model_max_abs_median",
    "model_max_abs_max",
    "model_p99_abs_median",
    "model_allclose_count",
    "first_token_match_count",
    "layer_checks",
    "layer_max_abs_median",
    "layer_max_abs_max",
    "layer_allclose_count",
    "worst_layer_index",
)


def _write(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _abs_statistics(actual: torch.Tensor, expected: torch.Tensor):
    difference = (actual.float() - expected.float()).abs().reshape(-1)
    return (
        float(difference.mean().item()),
        float(torch.quantile(difference, 0.99).item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument(
        "--accumulation-dtype", choices=("input", "fp32", "fp32_sum"), default="input"
    )
    parser.add_argument("--splits", default="1,2,4,8,16")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-layer-audit",
        action="store_true",
        help="only compare final logits/tokens, allowing many more input seeds",
    )
    args = parser.parse_args()
    splits = [int(value) for value in args.splits.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    dtype = DTYPES[args.dtype]
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="sdpa",
    ).eval()
    layers = find_opt_decoder_layers(model)
    activation_name = str(model.config.activation_function)
    rtol, atol = tolerances(dtype)
    fixtures = []
    with torch.inference_mode():
        for seed in seeds:
            captured = [None] * len(layers) if not args.skip_layer_audit else []
            handles = []
            for layer_index, layer in enumerate(layers if not args.skip_layer_audit else []):
                def hook(module, inputs, index=layer_index):
                    del module
                    captured[index] = inputs[0].detach().clone()

                handles.append(layer.fc1.register_forward_pre_hook(hook))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            input_ids = torch.randint(
                0,
                int(model.config.vocab_size),
                (args.batch_size, args.sequence_length),
                generator=generator,
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for handle in handles:
                handle.remove()
            if not args.skip_layer_audit and any(value is None for value in captured):
                raise RuntimeError("failed to capture every OPT MLP input")
            fixtures.append(
                {
                    "seed": seed,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "logits": result.logits[:, -1, :].detach().clone(),
                    "captured": captured,
                }
            )

    model_rows = []
    layer_rows = []
    with torch.inference_mode():
        for split_k in splits:
            patcher = OPTModelIMBPSPatcher(
                model,
                split_k,
                "prepacked",
                activation_name,
                accumulation_dtype=args.accumulation_dtype,
            )
            for fixture in fixtures:
                patcher.enable_imbps()
                result = model(
                    input_ids=fixture["input_ids"],
                    attention_mask=fixture["attention_mask"],
                    use_cache=False,
                )
                actual_logits = result.logits[:, -1, :]
                expected_logits = fixture["logits"]
                max_abs, max_rel = error_metrics(actual_logits, expected_logits)
                mean_abs, p99_abs = _abs_statistics(actual_logits, expected_logits)
                reference_top2 = torch.topk(expected_logits.float(), k=2, dim=-1)
                actual_top2 = torch.topk(actual_logits.float(), k=2, dim=-1)
                reference_token = expected_logits.argmax(dim=-1)
                actual_token = actual_logits.argmax(dim=-1)
                model_rows.append(
                    {
                        "seed": fixture["seed"],
                        "split_k": split_k,
                        "dtype": args.dtype,
                        "accumulation_dtype": args.accumulation_dtype,
                        "max_abs_logits": max_abs,
                        "mean_abs_logits": mean_abs,
                        "p99_abs_logits": p99_abs,
                        "max_rel_logits": max_rel,
                        "logits_allclose": bool(
                            torch.allclose(actual_logits, expected_logits, rtol=rtol, atol=atol)
                        ),
                        "first_token_match": bool(torch.equal(actual_token, reference_token)),
                        "reference_first_token_id": int(reference_token[0].item()),
                        "imbps_first_token_id": int(actual_token[0].item()),
                        "reference_top1_margin": float(
                            (reference_top2.values[0, 0] - reference_top2.values[0, 1]).item()
                        ),
                        "imbps_top1_margin": float(
                            (actual_top2.values[0, 0] - actual_top2.values[0, 1]).item()
                        ),
                    }
                )
                for layer_index, (patch, hidden_states) in enumerate(
                    zip(patcher.patches, fixture["captured"])
                ):
                    reference = HFOPTReferenceMLP(
                        patch.fc1, patch.activation_fn, patch.fc2
                    ).eval()
                    expected = reference(hidden_states).clone()
                    actual = patch.split_mlp(hidden_states).clone()
                    max_abs, max_rel = error_metrics(actual, expected)
                    mean_abs, p99_abs = _abs_statistics(actual, expected)
                    layer_rows.append(
                        {
                            "seed": fixture["seed"],
                            "layer_index": layer_index,
                            "split_k": split_k,
                            "dtype": args.dtype,
                            "accumulation_dtype": args.accumulation_dtype,
                            "max_abs_output": max_abs,
                            "mean_abs_output": mean_abs,
                            "p99_abs_output": p99_abs,
                            "max_rel_output": max_rel,
                            "output_allclose": bool(
                                torch.allclose(actual, expected, rtol=rtol, atol=atol)
                            ),
                        }
                    )
            patcher.close()

    summaries = []
    for split_k in splits:
        model_group = [row for row in model_rows if row["split_k"] == split_k]
        layer_group = [row for row in layer_rows if row["split_k"] == split_k]
        worst = (
            max(layer_group, key=lambda row: float(row["max_abs_output"]))
            if layer_group
            else None
        )
        summaries.append(
            {
                "split_k": split_k,
                "dtype": args.dtype,
                "accumulation_dtype": args.accumulation_dtype,
                "seeds": len(model_group),
                "model_max_abs_median": statistics.median(
                    float(row["max_abs_logits"]) for row in model_group
                ),
                "model_max_abs_max": max(float(row["max_abs_logits"]) for row in model_group),
                "model_p99_abs_median": statistics.median(
                    float(row["p99_abs_logits"]) for row in model_group
                ),
                "model_allclose_count": sum(row["logits_allclose"] for row in model_group),
                "first_token_match_count": sum(row["first_token_match"] for row in model_group),
                "layer_checks": len(layer_group),
                "layer_max_abs_median": (
                    statistics.median(float(row["max_abs_output"]) for row in layer_group)
                    if layer_group
                    else ""
                ),
                "layer_max_abs_max": "" if worst is None else float(worst["max_abs_output"]),
                "layer_allclose_count": sum(row["output_allclose"] for row in layer_group),
                "worst_layer_index": "" if worst is None else int(worst["layer_index"]),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "model_raw.csv", MODEL_FIELDS, model_rows)
    _write(args.output_dir / "layer_raw.csv", LAYER_FIELDS, layer_rows)
    _write(args.output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    for row in summaries:
        print(
            "K=%-2d model max median=%g max=%g allclose=%d/%d token=%d/%d "
            "layer max=%s allclose=%d/%d worst_layer=%s"
            % (
                row["split_k"],
                row["model_max_abs_median"],
                row["model_max_abs_max"],
                row["model_allclose_count"],
                row["seeds"],
                row["first_token_match_count"],
                row["seeds"],
                row["layer_max_abs_max"],
                row["layer_allclose_count"],
                row["layer_checks"],
                row["worst_layer_index"],
            )
        )


if __name__ == "__main__":
    main()
