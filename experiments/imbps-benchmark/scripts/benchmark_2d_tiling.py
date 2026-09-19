#!/usr/bin/env python3
"""Prototype row-by-intermediate 2D blocking against the native MLP."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from imbps_bench.kernels import (
    make_input,
    make_reference_workspace,
    make_weights,
    pack_weights,
    reference_forward_into,
)
from imbps_bench.runner import DTYPES, error_metrics, tolerances


FIELDS = (
    "row_tile",
    "split_k",
    "samples",
    "reference_median_ms",
    "tiled_median_ms",
    "speedup_vs_reference",
    "max_abs_error",
    "max_rel_error",
    "allclose",
    "up_workspace_bytes",
    "output_bytes",
)


def _gelu_in_place(tensor: torch.Tensor) -> None:
    try:
        torch.ops.aten.gelu_.default(tensor, approximate="none")
    except (AttributeError, RuntimeError, TypeError):
        tensor.copy_(F.gelu(tensor))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-kind", choices=("opt", "swiglu"), default="opt")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--split-k", type=int, required=True)
    parser.add_argument("--row-tiles", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250917)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row_tiles = [int(value) for value in args.row_tiles.split(",")]
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
    reference_workspace = make_reference_workspace(
        args.mlp_kind,
        args.tokens,
        args.hidden_size,
        args.intermediate_size,
        dtype,
    )
    packed = pack_weights(weights, args.split_k, "prepacked")
    output = torch.empty((args.tokens, args.hidden_size), dtype=dtype)
    rtol, atol = tolerances(dtype)
    rng = random.Random(args.seed)

    with torch.inference_mode():
        expected = reference_forward_into(source, weights, reference_workspace).clone()
        rows = []
        for requested_tile in row_tiles:
            row_tile = min(requested_tile, args.tokens)
            up_workspace = torch.empty((row_tile, packed.max_width), dtype=dtype)
            gate_workspace = (
                torch.empty((row_tile, packed.max_width), dtype=dtype)
                if args.mlp_kind == "swiglu"
                else None
            )

            def tiled_forward():
                for start in range(0, args.tokens, row_tile):
                    end = min(start + row_tile, args.tokens)
                    active = end - start
                    x_tile = source[start:end]
                    output_tile = output[start:end]
                    output_tile.zero_()
                    for block in packed.blocks:
                        up = up_workspace[:active, : block.width]
                        torch.mm(x_tile, block.up, out=up)
                        if args.mlp_kind == "opt":
                            _gelu_in_place(up)
                            output_tile.addmm_(up, block.down)
                            continue
                        if gate_workspace is None or block.gate is None:
                            raise RuntimeError("missing SwiGLU gate workspace")
                        gate = gate_workspace[:active, : block.width]
                        torch.mm(x_tile, block.gate, out=gate)
                        F.silu(gate, inplace=True)
                        gate.mul_(up)
                        output_tile.addmm_(gate, block.down)
                return output

            actual = tiled_forward().clone()
            max_abs, max_rel = error_metrics(actual, expected)
            allclose = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
            for _ in range(args.warmup):
                reference_forward_into(source, weights, reference_workspace)
                tiled_forward()
            reference_times = []
            tiled_times = []
            for _ in range(args.repeats):
                order = ["reference", "tiled"]
                rng.shuffle(order)
                for variant in order:
                    started = time.perf_counter_ns()
                    if variant == "reference":
                        reference_forward_into(source, weights, reference_workspace)
                        reference_times.append((time.perf_counter_ns() - started) / 1e6)
                    else:
                        tiled_forward()
                        tiled_times.append((time.perf_counter_ns() - started) / 1e6)
            ref_median = statistics.median(reference_times)
            tiled_median = statistics.median(tiled_times)
            rows.append(
                {
                    "row_tile": row_tile,
                    "split_k": args.split_k,
                    "samples": args.repeats,
                    "reference_median_ms": ref_median,
                    "tiled_median_ms": tiled_median,
                    "speedup_vs_reference": ref_median / tiled_median,
                    "max_abs_error": max_abs,
                    "max_rel_error": max_rel,
                    "allclose": allclose,
                    "up_workspace_bytes": up_workspace.numel() * up_workspace.element_size(),
                    "output_bytes": output.numel() * output.element_size(),
                }
            )
            print(
                "row_tile=%-6d K=%-2d reference=%9.3f ms tiled=%9.3f ms speedup=%7.3fx"
                % (row_tile, args.split_k, ref_median, tiled_median, ref_median / tiled_median),
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
