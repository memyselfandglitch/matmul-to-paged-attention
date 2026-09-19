#!/usr/bin/env python3
"""Long-running single-variant MLP loop for external hardware counters."""

from __future__ import annotations

import argparse
import time

import torch

from imbps_bench.kernels import (
    imbps_forward_into,
    make_input,
    make_reference_workspace,
    make_split_workspace,
    make_weights,
    pack_weights,
    reference_forward_into,
)
from imbps_bench.runner import DTYPES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("reference", "imbps"), required=True)
    parser.add_argument("--mlp-kind", choices=("opt", "swiglu"), default="opt")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--weight-layout", choices=("prepacked", "views"), default="prepacked")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--measurement-delay", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20250917)
    args = parser.parse_args()
    if min(
        args.tokens,
        args.hidden_size,
        args.intermediate_size,
        args.split_k,
        args.threads,
        args.iterations,
    ) <= 0:
        raise ValueError("sizes, splits, threads, and iterations must be positive")
    if args.warmup < 0 or args.measurement_delay < 0:
        raise ValueError("warmup and delay must be non-negative")

    started_setup = time.perf_counter_ns()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dtype = DTYPES[args.dtype]
    weights = make_weights(
        args.mlp_kind,
        args.hidden_size,
        args.intermediate_size,
        dtype,
        args.seed,
    )
    source = make_input(args.tokens, args.hidden_size, dtype, args.seed + 1)
    if args.variant == "reference":
        workspace = make_reference_workspace(
            args.mlp_kind,
            args.tokens,
            args.hidden_size,
            args.intermediate_size,
            dtype,
        )

        def invoke():
            return reference_forward_into(source, weights, workspace)

    else:
        packed = pack_weights(weights, args.split_k, args.weight_layout)
        workspace = make_split_workspace(
            args.mlp_kind,
            args.tokens,
            args.hidden_size,
            packed.max_width,
            dtype,
        )

        def invoke():
            return imbps_forward_into(source, packed, workspace)

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = invoke()
        setup_ms = (time.perf_counter_ns() - started_setup) / 1_000_000.0
        print(
            "ready variant=%s split_k=%d setup_ms=%.6f delay_s=%.3f"
            % (args.variant, args.split_k, setup_ms, args.measurement_delay),
            flush=True,
        )
        time.sleep(args.measurement_delay)
        started = time.perf_counter_ns()
        for _ in range(args.iterations):
            output = invoke()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        checksum = float(output.float().sum().item())
    print(
        "variant=%s split_k=%d iterations=%d elapsed_ms=%.6f ms_per_iteration=%.6f checksum=%.9g"
        % (
            args.variant,
            args.split_k,
            args.iterations,
            elapsed_ms,
            elapsed_ms / args.iterations,
            checksum,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
