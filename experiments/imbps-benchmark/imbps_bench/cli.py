"""Command-line interface for IMBPS benchmarks."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from imbps_bench.analytical import paper_split_prediction, paper_working_set_bytes
from imbps_bench.kernels import (
    MLP_KINDS,
    imbps_forward_into,
    make_input,
    make_reference_workspace,
    make_split_workspace,
    make_weights,
    pack_weights,
    reference_forward_into,
)
from imbps_bench.metadata import collect_metadata
from imbps_bench.runner import DTYPES, RunConfig, error_metrics, run_sweep, tolerances


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _splits(value: str) -> List[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("splits must be comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("splits must contain positive integers")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("splits must not contain duplicates")
    return result


def _default_output(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / ("%s-%s" % (stamp, kind))


def _correctness(args: argparse.Namespace) -> int:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    records = []
    failed = False
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        rtol, atol = tolerances(dtype)
        for kind in MLP_KINDS:
            weights = make_weights(
                kind,
                args.hidden_size,
                args.intermediate_size,
                dtype,
                args.seed,
            )
            x = make_input(args.tokens, args.hidden_size, dtype, args.seed + 1)
            reference_workspace = make_reference_workspace(
                kind,
                args.tokens,
                args.hidden_size,
                args.intermediate_size,
                dtype,
            )
            with torch.inference_mode():
                expected = reference_forward_into(
                    x, weights, reference_workspace, args.gelu_approximate
                ).clone()
                for layout in args.layouts:
                    for split_k in args.splits:
                        packed = pack_weights(weights, split_k, layout)
                        workspace = make_split_workspace(
                            kind,
                            args.tokens,
                            args.hidden_size,
                            packed.max_width,
                            dtype,
                        )
                        actual = imbps_forward_into(
                            x, packed, workspace, args.gelu_approximate
                        )
                        max_abs, max_rel = error_metrics(actual, expected)
                        allclose = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
                        failed = failed or not allclose
                        records.append(
                            {
                                "kind": kind,
                                "dtype": dtype_name,
                                "layout": layout,
                                "split_k": split_k,
                                "max_abs_error": max_abs,
                                "max_rel_error": max_rel,
                                "rtol": rtol,
                                "atol": atol,
                                "allclose": allclose,
                            }
                        )
    print(json.dumps({"passed": not failed, "checks": records}, indent=2))
    return 1 if failed else 0


def _sweep(args: argparse.Namespace) -> int:
    intermediate_size = args.intermediate_size
    if intermediate_size is None:
        intermediate_size = int(round(args.hidden_size * args.expansion_factor))
    output_dir = args.output_dir or _default_output(args.mlp_kind)
    config = RunConfig(
        mlp_kind=args.mlp_kind,
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        intermediate_size=intermediate_size,
        dtype_name=args.dtype,
        splits=args.splits,
        threads=args.threads,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        weight_layout=args.weight_layout,
        gelu_approximate=args.gelu_approximate,
        output_dir=output_dir,
        max_allocation_gib=args.max_allocation_gib,
        skip_correctness=args.skip_correctness,
        allow_correctness_failure=args.allow_correctness_failure,
    )
    result = run_sweep(config)
    print(json.dumps(result, indent=2, default=str))
    return 0


def _metadata(args: argparse.Namespace) -> int:
    metadata = collect_metadata(args.output)
    if args.output is None:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        print(str(args.output))
    return 0


def _predict_k(args: argparse.Namespace) -> int:
    dtype = DTYPES[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    cache_bytes = int(args.cache_mib * 1024 * 1024)
    prediction = paper_split_prediction(
        cache_bytes=cache_bytes,
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        element_size=element_size,
        cache_fraction=args.cache_fraction,
    )
    working_sets = {
        str(split_k): int(
            paper_working_set_bytes(
                args.tokens,
                args.hidden_size,
                args.intermediate_size,
                element_size,
                split_k,
            )
        )
        for split_k in args.splits
    }
    print(
        json.dumps(
            {
                "model": "IMBPS paper equations 12-13",
                "tokens_M": args.tokens,
                "hidden_size_H": args.hidden_size,
                "intermediate_size_fH": args.intermediate_size,
                "dtype": args.dtype,
                "element_size_bytes": element_size,
                "cache_fraction": args.cache_fraction,
                "prediction": {
                    "cache_bytes": prediction.cache_bytes,
                    "usable_cache_bytes": prediction.usable_cache_bytes,
                    "input_bytes": prediction.input_bytes,
                    "denominator_bytes": prediction.denominator_bytes,
                    "continuous_k": prediction.continuous_k,
                    "ceiling_k": prediction.ceiling_k,
                    "feasible": prediction.feasible,
                },
                "working_set_bytes_by_k": working_sets,
                "warning": (
                    "K-only capacity model is infeasible; tile M as well"
                    if not prediction.feasible
                    else "Prediction is a capacity hypothesis; benchmark nearby K values"
                ),
            },
            indent=2,
        )
    )
    return 0


def _hf_opt_layer(args: argparse.Namespace) -> int:
    from imbps_bench.hf_runner import HFOPTLayerConfig, run_hf_opt_layer

    output_dir = args.output_dir or _default_output("hf-opt-layer")
    result = run_hf_opt_layer(
        HFOPTLayerConfig(
            model_name_or_path=args.model_name_or_path,
            model_mode=args.model_mode,
            layer_index=args.layer_index,
            input_source=args.input_source,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            dtype_name=args.dtype,
            splits=args.splits,
            threads=args.threads,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            weight_layout=args.weight_layout,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            output_dir=output_dir,
        )
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _hf_opt_e2e(args: argparse.Namespace) -> int:
    from imbps_bench.hf_runner import HFOPTE2EConfig, run_hf_opt_e2e

    output_dir = args.output_dir or _default_output("hf-opt-e2e")
    result = run_hf_opt_e2e(
        HFOPTE2EConfig(
            model_name_or_path=args.model_name_or_path,
            batch_size=args.batch_size,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            prompt=args.prompt,
            dtype_name=args.dtype,
            splits=args.splits,
            threads=args.threads,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            weight_layout=args.weight_layout,
            attn_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            output_dir=output_dir,
        )
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imbps-bench",
        description="Reference and IMBPS CPU MLP benchmark harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    correctness = subparsers.add_parser("correctness", help="check split kernels against references")
    correctness.add_argument("--tokens", type=_positive_int, default=17)
    correctness.add_argument("--hidden-size", type=_positive_int, default=32)
    correctness.add_argument("--intermediate-size", type=_positive_int, default=93)
    correctness.add_argument("--splits", type=_splits, default=[1, 2, 3, 4, 7])
    correctness.add_argument(
        "--dtypes",
        type=lambda value: [item.strip() for item in value.split(",")],
        default=["fp32", "bf16"],
        choices=None,
        help="comma-separated subset of fp32,bf16",
    )
    correctness.add_argument(
        "--layouts",
        type=lambda value: [item.strip() for item in value.split(",")],
        default=["views", "prepacked"],
        help="comma-separated subset of views,prepacked",
    )
    correctness.add_argument("--threads", type=_positive_int, default=1)
    correctness.add_argument("--seed", type=int, default=20250917)
    correctness.add_argument("--gelu-approximate", choices=("none", "tanh"), default="none")
    correctness.set_defaults(function=_correctness)

    sweep = subparsers.add_parser("sweep", help="benchmark reference and split kernels")
    sweep.add_argument("--mlp-kind", choices=MLP_KINDS, required=True)
    sweep.add_argument("--tokens", type=_positive_int, required=True, help="active MLP rows (M)")
    sweep.add_argument("--hidden-size", type=_positive_int, required=True)
    size_group = sweep.add_mutually_exclusive_group()
    size_group.add_argument("--intermediate-size", type=_positive_int)
    size_group.add_argument("--expansion-factor", type=float, default=4.0)
    sweep.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    sweep.add_argument("--splits", type=_splits, default=[1, 2, 4, 8, 16, 32, 64])
    sweep.add_argument("--threads", type=_positive_int, default=1)
    sweep.add_argument("--warmup", type=_nonnegative_int, default=3)
    sweep.add_argument("--repeats", type=_positive_int, default=10)
    sweep.add_argument("--seed", type=int, default=20250917)
    sweep.add_argument("--weight-layout", choices=("prepacked", "views"), default="prepacked")
    sweep.add_argument("--gelu-approximate", choices=("none", "tanh"), default="none")
    sweep.add_argument("--output-dir", type=Path)
    sweep.add_argument("--max-allocation-gib", type=float, default=16.0)
    sweep.add_argument("--skip-correctness", action="store_true")
    sweep.add_argument("--allow-correctness-failure", action="store_true")
    sweep.set_defaults(function=_sweep)

    metadata = subparsers.add_parser("metadata", help="capture machine and software metadata")
    metadata.add_argument("--output", type=Path)
    metadata.set_defaults(function=_metadata)

    predict = subparsers.add_parser("predict-k", help="evaluate the paper's cache-capacity K model")
    predict.add_argument("--tokens", type=_positive_int, required=True)
    predict.add_argument("--hidden-size", type=_positive_int, required=True)
    predict.add_argument("--intermediate-size", type=_positive_int, required=True)
    predict.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    predict.add_argument("--cache-mib", type=float, required=True)
    predict.add_argument("--cache-fraction", type=float, default=1.0)
    predict.add_argument("--splits", type=_splits, default=[1, 2, 4, 8, 16, 32, 64])
    predict.set_defaults(function=_predict_k)

    hf_layer = subparsers.add_parser(
        "hf-opt-layer",
        help="benchmark an actual Hugging Face OPT fc1/activation/fc2 module",
    )
    hf_layer.add_argument("--model-name-or-path", required=True)
    hf_layer.add_argument(
        "--model-mode",
        choices=("pretrained", "random-config"),
        default="pretrained",
        help="load pretrained weights or instantiate the real layer class from config",
    )
    hf_layer.add_argument("--layer-index", type=_nonnegative_int, default=0)
    hf_layer.add_argument(
        "--input-source",
        choices=("captured", "random"),
        default="captured",
        help="capture the real fc1 input or use a deterministic synthetic activation",
    )
    hf_layer.add_argument("--batch-size", type=_positive_int, default=1)
    hf_layer.add_argument("--sequence-length", type=_positive_int, default=256)
    hf_layer.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    hf_layer.add_argument("--splits", type=_splits, default=[1, 2, 4, 5, 6, 7, 8, 16])
    hf_layer.add_argument("--threads", type=_positive_int, default=8)
    hf_layer.add_argument("--warmup", type=_nonnegative_int, default=3)
    hf_layer.add_argument("--repeats", type=_positive_int, default=10)
    hf_layer.add_argument("--seed", type=int, default=20250917)
    hf_layer.add_argument("--weight-layout", choices=("prepacked", "views"), default="prepacked")
    hf_layer.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    hf_layer.add_argument("--local-files-only", action="store_true")
    hf_layer.add_argument("--output-dir", type=Path)
    hf_layer.set_defaults(function=_hf_opt_layer)

    hf_e2e = subparsers.add_parser(
        "hf-opt-e2e",
        help="benchmark complete Hugging Face OPT prefill, TTFT, and decode throughput",
    )
    hf_e2e.add_argument("--model-name-or-path", required=True)
    hf_e2e.add_argument("--batch-size", type=_positive_int, default=1)
    hf_e2e.add_argument("--input-tokens", type=_positive_int, default=256)
    hf_e2e.add_argument("--output-tokens", type=_positive_int, default=16)
    hf_e2e.add_argument(
        "--prompt",
        help="optional text prompt; omitted uses deterministic random token IDs like AMD PACE",
    )
    hf_e2e.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    hf_e2e.add_argument("--splits", type=_splits, default=[1, 2, 4, 5, 6, 7, 8])
    hf_e2e.add_argument("--threads", type=_positive_int, default=8)
    hf_e2e.add_argument("--warmup", type=_nonnegative_int, default=1)
    hf_e2e.add_argument("--repeats", type=_positive_int, default=5)
    hf_e2e.add_argument("--seed", type=int, default=20250917)
    hf_e2e.add_argument("--weight-layout", choices=("prepacked", "views"), default="prepacked")
    hf_e2e.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    hf_e2e.add_argument("--local-files-only", action="store_true")
    hf_e2e.add_argument("--output-dir", type=Path)
    hf_e2e.set_defaults(function=_hf_opt_e2e)
    return parser


def _validate_list_options(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command != "correctness":
        return
    unknown_dtypes = set(args.dtypes) - set(DTYPES)
    if unknown_dtypes:
        parser.error("unknown correctness dtypes: %s" % sorted(unknown_dtypes))
    unknown_layouts = set(args.layouts) - {"views", "prepacked"}
    if unknown_layouts:
        parser.error("unknown correctness layouts: %s" % sorted(unknown_layouts))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_list_options(args, parser)
    try:
        return int(args.function(args))
    except (ValueError, RuntimeError, MemoryError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
