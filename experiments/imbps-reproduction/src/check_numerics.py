#!/usr/bin/env python3
"""Compare PACE's TPP baseline and IMBPS outputs across split counts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def comma_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--splits", type=comma_ints, default=comma_ints("1,2,4,8,16"))
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--activation", choices=("gelu", "relu"), default="relu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.hidden, args.intermediate, args.rows) <= 0:
        parser.error("dimensions must be positive")
    for split in args.splits:
        if args.intermediate % split != 0:
            parser.error(f"intermediate={args.intermediate} is not divisible by K={split}")
    return args


def main() -> None:
    args = parse_args()

    import torch
    from pace.ops.base import BackendBase
    from pace.ops.enum import BackendType
    from pace.ops.mlp import MergedMLP

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    affinity = set(os.sched_getaffinity(0))
    if len(affinity) < torch.get_num_threads():
        raise RuntimeError(
            "OpenMP oversubscription: "
            f"torch has {torch.get_num_threads()} threads but process affinity "
            f"contains only {len(affinity)} CPUs ({sorted(affinity)})"
        )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(args.seed)

    raw = {
        "up_weight": torch.empty(args.intermediate, args.hidden, dtype=dtype).uniform_(-0.02, 0.02),
        "up_bias": torch.empty(args.intermediate, dtype=dtype).uniform_(-0.02, 0.02),
        "down_weight": torch.empty(args.hidden, args.intermediate, dtype=dtype).uniform_(-0.02, 0.02),
        "down_bias": torch.empty(args.hidden, dtype=dtype).uniform_(-0.02, 0.02),
    }
    source = torch.empty(args.rows, args.hidden, dtype=dtype).uniform_(-0.02, 0.02)

    def make_model(backend: BackendType, split: int) -> MergedMLP:
        os.environ["IMBPS_BLOCK_SIZE"] = str(split)
        model = MergedMLP(
            in_features=args.hidden,
            out_features=args.intermediate,
            bias=True,
            activation=args.activation,
            gate=False,
            dtype=dtype,
            backend_impl=backend,
        )
        model.eval()
        with torch.no_grad():
            model.up_proj.linear.weight.copy_(raw["up_weight"])
            model.up_proj.linear.bias.copy_(raw["up_bias"])
            model.down_proj.weight.copy_(raw["down_weight"])
            model.down_proj.bias.copy_(raw["down_bias"])
        for module in model.modules():
            module_backend = getattr(module, "backend", None)
            if isinstance(module_backend, BackendBase):
                module_backend.preprocess(module)
        return model

    baseline_model = make_model(BackendType.TPP, 1)
    with torch.inference_mode():
        baseline = baseline_model(source).float()
    del baseline_model

    baseline_argmax = baseline.argmax(dim=-1)
    rows = []
    for split in args.splits:
        candidate_model = make_model(BackendType.IMBPS, split)
        with torch.inference_mode():
            candidate = candidate_model(source).float()
        absolute = (candidate - baseline).abs().flatten()
        candidate_argmax = candidate.argmax(dim=-1)
        candidate_top5 = candidate.topk(k=min(5, candidate.shape[-1]), dim=-1).indices
        baseline_in_top5 = (candidate_top5 == baseline_argmax.unsqueeze(-1)).any(dim=-1)
        row = {
            "splits": split,
            "elements": absolute.numel(),
            "max_abs": float(absolute.max().item()),
            "mean_abs": float(absolute.mean().item()),
            "q50_abs": float(torch.quantile(absolute, 0.50).item()),
            "q90_abs": float(torch.quantile(absolute, 0.90).item()),
            "q99_abs": float(torch.quantile(absolute, 0.99).item()),
            "q999_abs": float(torch.quantile(absolute, 0.999).item()),
            "exact_element_fraction": float((absolute == 0).float().mean().item()),
            "fraction_abs_le_1e_4": float((absolute <= 1e-4).float().mean().item()),
            "fraction_abs_le_1e_2": float((absolute <= 1e-2).float().mean().item()),
            "output_coordinate_argmax_agreement": float((candidate_argmax == baseline_argmax).float().mean().item()),
            "baseline_argmax_in_candidate_top5": float(baseline_in_top5.float().mean().item()),
        }
        rows.append(row)
        del candidate_model, candidate, absolute

    result = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "standalone MLP output; coordinate argmax is a proxy, not a vocabulary-logit claim",
        "configuration": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "rows": args.rows,
            "dtype": args.dtype,
            "activation": args.activation,
            "seed": args.seed,
            "threads": torch.get_num_threads(),
        },
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
