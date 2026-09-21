#!/usr/bin/env python3
"""Run one PACE fused MLP configuration for external PMU collection."""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--split-k", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--measurement-delay", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--data-scale", type=float, default=0.01)
    args = parser.parse_args()
    if min(
        args.tokens,
        args.hidden_size,
        args.intermediate_size,
        args.split_k,
        args.threads,
        args.iterations,
    ) <= 0:
        raise ValueError("dimensions, K, threads, and iterations must be positive")

    import pace  # noqa: F401

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    x = torch.rand(
        args.tokens, args.hidden_size, generator=generator, dtype=dtype
    ) * args.data_scale
    up_weight = torch.rand(
        args.intermediate_size, args.hidden_size, generator=generator, dtype=dtype
    ) * args.data_scale
    up_bias = torch.rand(
        args.intermediate_size, generator=generator, dtype=dtype
    ) * args.data_scale
    down_weight = torch.rand(
        args.hidden_size, args.intermediate_size, generator=generator, dtype=dtype
    ) * args.data_scale
    down_bias = torch.rand(
        args.hidden_size, generator=generator, dtype=dtype
    ) * args.data_scale
    up_weights = [part.contiguous() for part in up_weight.chunk(args.split_k, dim=0)]
    up_biases = [part.contiguous() for part in up_bias.chunk(args.split_k, dim=0)]
    down_weights = [part.contiguous() for part in down_weight.chunk(args.split_k, dim=1)]

    def invoke() -> torch.Tensor:
        return torch.ops.pace.mlp_mlp_fusion(
            x,
            up_weights,
            up_biases,
            down_weights,
            down_bias,
            args.activation.capitalize(),
            None,
            None,
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = invoke()
        print(
            f"ready split_k={args.split_k} delay_s={args.measurement_delay}",
            flush=True,
        )
        time.sleep(args.measurement_delay)
        start = time.perf_counter_ns()
        for _ in range(args.iterations):
            output = invoke()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        checksum = float(output.float().sum().item())
    print(
        f"split_k={args.split_k} iterations={args.iterations} "
        f"elapsed_ms={elapsed_ms:.6f} ms_per_iteration={elapsed_ms / args.iterations:.6f} "
        f"checksum={checksum:.9g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
