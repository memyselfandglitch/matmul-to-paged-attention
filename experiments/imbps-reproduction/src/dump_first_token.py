#!/usr/bin/env python3
"""Dump deterministic first-token log probabilities from one PACE backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--backend", choices=("tpp", "imbps"), required=True)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    args = parser.parse_args()
    if min(args.splits, args.samples, args.sequence, args.batch_size) <= 0:
        parser.error("splits and dimensions must be positive")
    if args.samples % args.batch_size != 0:
        parser.error("samples must be divisible by batch-size")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    os.environ["IMBPS_BLOCK_SIZE"] = str(args.splits)

    import numpy as np
    import torch
    from transformers import BatchEncoding
    from pace.llm import (
        KVCacheType,
        LLMBackendType,
        LLMModel,
        LLMOperatorType,
        OperatorConfig,
        SamplingConfig,
    )

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    backend = LLMBackendType.IMBPS if args.backend == "imbps" else LLMBackendType.TPP
    opconfig = OperatorConfig(
        **{
            LLMOperatorType.Norm: LLMBackendType.NATIVE,
            LLMOperatorType.QKVProjection: LLMBackendType.TPP,
            LLMOperatorType.Attention: LLMBackendType.JIT,
            LLMOperatorType.OutProjection: LLMBackendType.TPP,
            LLMOperatorType.MLP: backend,
            LLMOperatorType.LMHead: LLMBackendType.TPP,
        }
    )
    model = LLMModel(
        args.model,
        dtype=torch.bfloat16,
        kv_cache_type=KVCacheType.DYNAMIC,
        opconfig=opconfig,
    )
    model_config = model.get_config()
    tokenizer = model.get_tokenizer()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(args.samples, args.sequence),
        generator=generator,
        dtype=torch.long,
    )
    sampling = SamplingConfig(
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        temperature=0,
        seed=args.seed,
        return_input_logprobs=True,
    )

    last_logprobs = []
    generated_tokens = []
    for start in range(0, args.samples, args.batch_size):
        batch_ids = input_ids[start : start + args.batch_size]
        encoded = BatchEncoding(
            {"input_ids": batch_ids, "attention_mask": torch.ones_like(batch_ids)}
        )
        output = model.generate(encoded, sampling)
        if output.input_logprobs is None:
            raise RuntimeError("PACE did not return input log probabilities")
        last_logprobs.append(output.input_logprobs[:, -1, :].float().cpu())
        generated_tokens.append(output.output_token_ids[:, -1].cpu())

    logprobs = torch.cat(last_logprobs, dim=0).numpy()
    generated = torch.cat(generated_tokens, dim=0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        input_ids=input_ids.numpy(),
        last_logprobs=logprobs,
        generated_tokens=generated,
    )
    metadata = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": getattr(model_config, "_commit_hash", None),
        "backend": args.backend,
        "splits": args.splits,
        "dtype": "bf16",
        "samples": args.samples,
        "sequence": args.sequence,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "output": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "shape": list(logprobs.shape),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
