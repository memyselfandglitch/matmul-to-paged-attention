#!/usr/bin/env python3
"""Benchmark PACE target-model decode/verification with a populated KV cache.

This is deliberately a target-verification microbenchmark, not an end-to-end
speculative-decoding benchmark.  It pre-fills PACE's real KV cache once for a
batch, submits one or more replayed draft tokens to the target model, and rolls
those tokens back outside the timed target forward pass.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import random
import resource
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .pace_mlp_bench import (
        allocator_mappings,
        collect_thread_affinity,
        percentile,
        summarize_thread_affinity,
    )
except ImportError:  # Direct execution: python3 src/pace_target_verify_bench.py
    from pace_mlp_bench import (
        allocator_mappings,
        collect_thread_affinity,
        percentile,
        summarize_thread_affinity,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
EXPECTED_PACE_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    parser.add_argument("--backend", choices=("tpp", "imbps"), required=True)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min-round-seconds", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20251001)
    parser.add_argument("--signature-rows", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.splits, args.rounds, args.iterations, args.signature_rows) <= 0:
        parser.error("splits, rounds, iterations, and signature-rows must be positive")
    if args.warmups < 0 or args.min_round_seconds < 0:
        parser.error("warmups and min-round-seconds must be nonnegative")
    if args.backend == "tpp" and args.splits != 1:
        parser.error("the TPP baseline must use K=1")
    if args.backend == "imbps" and args.splits == 1:
        parser.error("IMBPS must use K>1")
    return args


def load_claim(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    try:
        claim = registry["target_verification"][name]
    except KeyError as error:
        choices = ", ".join(sorted(registry.get("target_verification", {})))
        raise ValueError(
            f"unknown target-verification claim {name!r}; choose {choices}"
        ) from error
    return registry, claim


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("target-verification claim has no cases")
    identifiers: set[str] = set()
    for case in cases:
        required = {"case_id", "mode", "batch", "draft_tokens", "context_tokens"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"case is missing fields: {sorted(missing)}")
        if case["case_id"] in identifiers:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        identifiers.add(case["case_id"])
        if case["mode"] not in {"normal_decode", "target_verification"}:
            raise ValueError(f"unsupported mode: {case['mode']}")
        if min(case["batch"], case["draft_tokens"], case["context_tokens"]) <= 0:
            raise ValueError(f"case dimensions must be positive: {case}")
        if case["mode"] == "normal_decode" and case["draft_tokens"] != 1:
            raise ValueError("normal_decode cases must use draft_tokens=1")


def grouped_cases(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group cases that can reuse the same prefilled cache."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(case["batch"], case["context_tokens"])].append(case)
    return [
        sorted(groups[key], key=lambda case: (case["draft_tokens"], case["case_id"]))
        for key in sorted(groups)
    ]


def nominal_kv_bytes(
    batch: int,
    context_tokens: int,
    num_layers: int,
    hidden_size: int,
    bytes_per_element: int = 2,
) -> int:
    """Return K+V bytes for an OPT-style full-head cache."""
    return 2 * batch * context_tokens * num_layers * hidden_size * bytes_per_element


def allocated_cache_bytes(manager: Any) -> int:
    total = 0
    for cache in manager.cache_objects:
        for tensor in (cache.key, cache.value):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def snapshot_commit(path: Path) -> str | None:
    parts = path.resolve().parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def topk_signature(torch: Any, logits: Any, rows: int) -> dict[str, Any]:
    flattened = logits.reshape(-1, logits.shape[-1])
    sampled = flattened[: min(rows, flattened.shape[0])].float()
    top5 = torch.topk(sampled, k=5, dim=-1).indices.cpu().tolist()
    return {
        "sample_rows": len(top5),
        "top1_token_ids": [row[0] for row in top5],
        "top5_token_ids": top5,
        "sampled_logits_sum_fp32": float(sampled.sum().item()),
        "logits_shape": list(logits.shape),
    }


def run_target_forward(
    torch: Any,
    target_model: Any,
    cache_manager: Any,
    full_inputs: Any,
    attention_mask: Any,
    context_tokens: int,
) -> tuple[float, float, Any]:
    draft_tokens = full_inputs.shape[1] - context_tokens
    started = time.perf_counter_ns()
    output = target_model(
        full_inputs,
        kv_cache=cache_manager,
        attention_mask=attention_mask,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if len(cache_manager) != context_tokens + draft_tokens:
        raise RuntimeError(
            "target forward produced an unexpected cache length: "
            f"expected {context_tokens + draft_tokens}, found {len(cache_manager)}"
        )
    rollback_started = time.perf_counter_ns()
    cache_manager.remove_cache(draft_tokens)
    rollback_ms = (time.perf_counter_ns() - rollback_started) / 1_000_000
    if len(cache_manager) != context_tokens:
        raise RuntimeError(
            f"cache rollback failed: expected {context_tokens}, found {len(cache_manager)}"
        )
    return elapsed_ms, rollback_ms, output.logits


def main() -> None:
    args = parse_args()
    registry, claim = load_claim(args.claim)
    cases = [dict(case) for case in claim["cases"]]
    validate_cases(cases)
    if claim.get("dtype") != "bf16" or claim.get("kv_cache_type") != "BMC":
        raise ValueError("this runner requires a BF16 claim with PACE BMC cache")
    if args.backend == "imbps" and args.splits not in claim["splits"]:
        raise ValueError(
            f"K={args.splits} is not preregistered for claim {args.claim!r}"
        )
    model_name = claim["model"]
    model_record = registry["models"][model_name]
    if model_record["intermediate"] % args.splits != 0:
        raise ValueError(
            f"{model_name} intermediate={model_record['intermediate']} is not "
            f"divisible by K={args.splits}"
        )
    model_reference = args.model or model_record["hf_id"]
    os.environ["IMBPS_BLOCK_SIZE"] = str(args.splits)

    import torch
    from pace.llm import (
        KVCacheType,
        LLMBackendType,
        LLMModel,
        LLMOperatorType,
        OperatorConfig,
        SamplingConfig,
    )
    from transformers import BatchEncoding

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    pace_version = importlib.metadata.version("pace")
    if pace_version != EXPECTED_PACE_VERSION:
        raise RuntimeError(
            f"expected PACE {EXPECTED_PACE_VERSION}, found {pace_version}"
        )
    mapped_allocators = allocator_mappings()
    if os.environ.get("REQUIRE_TCMALLOC") == "1" and not any(
        "tcmalloc" in path.lower() for path in mapped_allocators
    ):
        raise RuntimeError(
            "REQUIRE_TCMALLOC=1 but no tcmalloc library is mapped in this process"
        )

    mlp_backend = (
        LLMBackendType.IMBPS if args.backend == "imbps" else LLMBackendType.TPP
    )
    opconfig = OperatorConfig(
        **{
            LLMOperatorType.Norm: LLMBackendType.NATIVE,
            LLMOperatorType.QKVProjection: LLMBackendType.TPP,
            LLMOperatorType.Attention: LLMBackendType.JIT,
            LLMOperatorType.OutProjection: LLMBackendType.TPP,
            LLMOperatorType.MLP: mlp_backend,
            LLMOperatorType.LMHead: LLMBackendType.TPP,
        }
    )

    load_started = time.perf_counter()
    llm = LLMModel(
        model_reference,
        dtype=torch.bfloat16,
        kv_cache_type=KVCacheType.BMC,
        opconfig=opconfig,
        disable_tqdm=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    generator = llm.generator
    config = llm.get_config()
    observed_shape = (
        int(config.hidden_size),
        int(getattr(config, "ffn_dim", -1)),
    )
    expected_shape = (
        int(model_record["hidden"]),
        int(model_record["intermediate"]),
    )
    if observed_shape != expected_shape:
        raise RuntimeError(
            f"resolved model shape {observed_shape} does not match "
            f"registered {model_name} shape {expected_shape}"
        )
    resolved_model_path = Path(generator.model_path).resolve()
    vocab_size = config.vocab_size

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    affinity_checked = False
    affinity_union: list[int] = []
    thread_affinities: dict[int, list[int]] = {}

    with torch.inference_mode():
        for group_index, group in enumerate(grouped_cases(cases), start=1):
            batch = group[0]["batch"]
            context_tokens = group[0]["context_tokens"]
            max_draft_tokens = max(case["draft_tokens"] for case in group)
            cache_prime_tokens = max_draft_tokens + 1
            tensor_generator = torch.Generator(device="cpu").manual_seed(
                args.seed + batch * 1009 + context_tokens * 9176
            )
            prompts = torch.randint(
                0,
                vocab_size,
                (batch, context_tokens),
                generator=tensor_generator,
                dtype=torch.long,
            )
            drafts = torch.randint(
                0,
                vocab_size,
                (batch, cache_prime_tokens),
                generator=tensor_generator,
                dtype=torch.long,
            )
            encoded = BatchEncoding(
                {
                    "input_ids": prompts,
                    "attention_mask": torch.ones_like(prompts),
                }
            )
            sampling = SamplingConfig(
                max_new_tokens=cache_prime_tokens,
                min_new_tokens=cache_prime_tokens,
                do_sample=False,
                temperature=0,
                seed=args.seed,
            )
            prepared = generator.prepare_for_generate(encoded, sampling)
            prefill_started = time.perf_counter_ns()
            hidden_states = generator.model.model(
                prepared,
                kv_cache=generator.kv_cache_manager,
                attention_mask=encoded["attention_mask"],
            )
            prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000
            del hidden_states
            if len(generator.kv_cache_manager) != context_tokens:
                raise RuntimeError(
                    f"prefill cache length mismatch: expected {context_tokens}, "
                    f"found {len(generator.kv_cache_manager)}"
                )

            # Prime one token beyond the largest measured block. PACE v1.0's
            # BMC cache grows in segments and uses a >= boundary check; this
            # prevents a repeated segment allocation from contaminating a
            # steady-state target-forward measurement at an exact boundary.
            prime_inputs = torch.cat(
                [prompts, drafts[:, :cache_prime_tokens]], dim=1
            )
            prime_mask = torch.ones(
                (batch, context_tokens + cache_prime_tokens), dtype=torch.long
            )
            prime_ms, prime_rollback_ms, prime_logits = run_target_forward(
                torch,
                generator.model,
                generator.kv_cache_manager,
                prime_inputs,
                prime_mask,
                context_tokens,
            )
            del prime_logits, prime_inputs, prime_mask

            full_inputs: dict[str, Any] = {}
            masks: dict[str, Any] = {}
            for case in group:
                gamma = case["draft_tokens"]
                full_inputs[case["case_id"]] = torch.cat(
                    [prompts, drafts[:, :gamma]], dim=1
                )
                masks[case["case_id"]] = torch.ones(
                    (batch, context_tokens + gamma), dtype=torch.long
                )

            signatures: dict[str, dict[str, Any]] = {}
            for case in group:
                for _ in range(args.warmups):
                    _, _, warmup_logits = run_target_forward(
                        torch,
                        generator.model,
                        generator.kv_cache_manager,
                        full_inputs[case["case_id"]],
                        masks[case["case_id"]],
                        context_tokens,
                    )
                    del warmup_logits

            if not affinity_checked:
                thread_affinities, affinity_union = collect_thread_affinity()
                if len(affinity_union) < torch.get_num_threads():
                    raise RuntimeError(
                        "OpenMP oversubscription after warmup: "
                        f"torch has {torch.get_num_threads()} threads but worker "
                        f"masks cover only {len(affinity_union)} CPUs"
                    )
                affinity_checked = True

            per_case_rounds: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for round_number in range(1, args.rounds + 1):
                order = list(group)
                rng.shuffle(order)
                for order_number, case in enumerate(order, start=1):
                    durations_ms: list[float] = []
                    rollback_ms: list[float] = []
                    round_started = time.perf_counter()
                    logits = None
                    while (
                        len(durations_ms) < args.iterations
                        or time.perf_counter() - round_started < args.min_round_seconds
                    ):
                        elapsed, rollback, logits = run_target_forward(
                            torch,
                            generator.model,
                            generator.kv_cache_manager,
                            full_inputs[case["case_id"]],
                            masks[case["case_id"]],
                            context_tokens,
                        )
                        durations_ms.append(elapsed)
                        rollback_ms.append(rollback)
                    assert logits is not None
                    signatures[case["case_id"]] = topk_signature(
                        torch, logits, args.signature_rows
                    )
                    del logits
                    round_record = {
                        "round": round_number,
                        "order_within_cache_group": order_number,
                        "durations_ms": durations_ms,
                        "rollback_durations_ms": rollback_ms,
                        "median_ms": statistics.median(durations_ms),
                        "median_rollback_ms": statistics.median(rollback_ms),
                        "iterations": len(durations_ms),
                    }
                    per_case_rounds[case["case_id"]].append(round_record)
                    print(
                        f"backend={args.backend} K={args.splits} "
                        f"case={case['case_id']} round={round_number} "
                        f"median_ms={round_record['median_ms']:.3f}",
                        flush=True,
                    )

            actual_cache_bytes = allocated_cache_bytes(generator.kv_cache_manager)
            nominal_bytes = nominal_kv_bytes(
                batch,
                context_tokens,
                config.num_hidden_layers,
                config.hidden_size,
            )
            group_reports.append(
                {
                    "group": group_index,
                    "batch": batch,
                    "context_tokens": context_tokens,
                    "max_draft_tokens": max_draft_tokens,
                    "cache_prime_tokens": cache_prime_tokens,
                    "cache_prime_ms": prime_ms,
                    "cache_prime_rollback_ms": prime_rollback_ms,
                    "prefill_ms": prefill_ms,
                    "nominal_kv_cache_bytes": nominal_bytes,
                    "allocated_kv_cache_bytes": actual_cache_bytes,
                    "case_ids": [case["case_id"] for case in group],
                }
            )
            for case in group:
                all_durations = [
                    value
                    for round_record in per_case_rounds[case["case_id"]]
                    for value in round_record["durations_ms"]
                ]
                all_rollbacks = [
                    value
                    for round_record in per_case_rounds[case["case_id"]]
                    for value in round_record["rollback_durations_ms"]
                ]
                active_rows = case["batch"] * case["draft_tokens"]
                records.append(
                    {
                        "case": {
                            **case,
                            "active_rows": active_rows,
                        },
                        "rounds": per_case_rounds[case["case_id"]],
                        "run": {
                            "median_ms": statistics.median(all_durations),
                            "mean_ms": statistics.fmean(all_durations),
                            "q1_ms": percentile(all_durations, 0.25),
                            "q3_ms": percentile(all_durations, 0.75),
                            "iterations": len(all_durations),
                            "median_rollback_ms": statistics.median(all_rollbacks),
                            "active_rows_per_second": active_rows
                            / (statistics.median(all_durations) / 1000),
                        },
                        "numerics": signatures[case["case_id"]],
                        "cache": {
                            "type": "BMC",
                            "prefill_ms": prefill_ms,
                            "nominal_bytes": nominal_bytes,
                            "allocated_bytes": actual_cache_bytes,
                        },
                    }
                )

            del full_inputs, masks, prompts, drafts, prepared, encoded
            del generator.kv_cache_manager
            gc.collect()

    result = {
        "schema_version": 1,
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": "kv_cache_target_verification",
        "scope_note": (
            "Includes target-model attention, KV-cache read/write, MLP, and LM head. "
            "Excludes draft-model generation, token acceptance, sampling, and prefill "
            "from the reported target-forward latency. A one-block cache-allocation "
            "prime is also excluded so timings represent steady-state forwards."
        ),
        "claim": args.claim,
        "backend": args.backend,
        "split": args.splits,
        "expected_variants": [
            {"backend": "tpp", "split": 1},
            *[
                {"backend": "imbps", "split": int(split)}
                for split in claim["splits"]
            ],
        ],
        "model": {
            "name": model_name,
            "requested_reference": model_reference,
            "resolved_path": str(resolved_model_path),
            "snapshot_commit": snapshot_commit(resolved_model_path),
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": getattr(config, "ffn_dim", None),
            "vocab_size": config.vocab_size,
            "dtype": "bf16",
        },
        "runner": {
            "rounds": args.rounds,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "min_round_seconds": args.min_round_seconds,
            "seed": args.seed,
            "signature_rows": args.signature_rows,
            "model_load_seconds": model_load_seconds,
        },
        "cache_groups": group_reports,
        "records": records,
        "environment": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "pace_version": pace_version,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "affinity": affinity_union,
            "main_thread_affinity": sorted(os.sched_getaffinity(0)),
            "process_thread_count": len(thread_affinities),
            "thread_affinity_masks": summarize_thread_affinity(thread_affinities),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "omp_places": os.environ.get("OMP_PLACES"),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "imbps_block_size": os.environ.get("IMBPS_BLOCK_SIZE"),
            "libxsmm_block_size": os.environ.get("LIBXSMM_BLOCK_SIZE"),
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
            "slurm_mem_per_node_mib": os.environ.get("SLURM_MEM_PER_NODE"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "ld_preload": os.environ.get("LD_PRELOAD"),
            "allocator_mappings": mapped_allocators,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
