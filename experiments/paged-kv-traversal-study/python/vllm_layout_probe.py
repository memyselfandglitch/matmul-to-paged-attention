#!/usr/bin/env python3
"""Report the physical per-layer shapes implied by vLLM's layout enum."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="optional path to a vLLM source checkout",
    )
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--content", type=int, default=128)
    return parser.parse_args()


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = stride
        stride *= shape[index]
    return tuple(strides)


def main() -> None:
    args = parse_args()
    if args.vllm_source:
        module_path = (
            args.vllm_source.resolve() / "vllm" / "v1" / "kv_cache_layout.py"
        )
        spec = importlib.util.spec_from_file_location("vllm_kv_cache_layout", module_path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        KVCacheLayout = module.KVCacheLayout
    else:
        try:
            from vllm.v1.kv_cache_layout import KVCacheLayout
        except ImportError as error:
            raise SystemExit(
                "Could not import vLLM. Activate the vLLM environment or pass "
                "--vllm-source /path/to/vllm."
            ) from error

    # vLLM's logical per-layer order is B,H,N,C. B is a physical cache block.
    logical_shape = (args.blocks, args.heads, args.block_size, args.content)
    print("Logical per-layer axes: B,H,N,C")
    print(f"Logical shape: {logical_shape}")
    print("Legacy aliases: HND=LBHNC, NHD=LBNHC")
    print()

    for layout in KVCacheLayout:
        order = layout.layer_view_order
        physical_shape = tuple(logical_shape[index] for index in order)
        physical_strides = contiguous_strides(physical_shape)
        logical_strides = tuple(physical_strides[order.index(i)] for i in range(4))
        print(
            f"{layout.name:5} per-layer order={order} "
            f"physical_shape={physical_shape} logical_strides={logical_strides}"
        )


if __name__ == "__main__":
    main()
