#!/usr/bin/env python3
"""Measure complete PACE normal and PARD speculative generation.

Unlike ``pace_target_verify_bench.py``, this runner times the public
``LLMModel.generate`` path.  The measurement therefore includes prompt
prefill, KV-cache creation and traffic, iterative decoding, sampling, and (for
PARD) draft generation, target verification, acceptance, and cache rollback.
Model loading and benchmark warmups remain outside the measured interval.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import statistics
import time
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
    from .pace_target_verify_bench import snapshot_commit
except ImportError:  # Direct execution: python3 src/full_decode_bench.py
    from pace_mlp_bench import (
        allocator_mappings,
        collect_thread_affinity,
        percentile,
        summarize_thread_affinity,
    )
    from pace_target_verify_bench import snapshot_commit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = PROJECT_ROOT / "configs" / "claims.json"
EXPECTED_PACE_VERSION = "1.0.0"


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", default="qwen25_7b_pard")
    parser.add_argument("--mode", choices=("normal", "pard"), required=True)
    parser.add_argument("--backend", choices=("tpp", "imbps"), required=True)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--draft-tokens", type=int, default=0)
    parser.add_argument("--contexts", type=comma_ints, default=None)
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20251001)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.splits, args.rounds) <= 0 or args.warmups < 0:
        parser.error("splits/rounds must be positive and warmups nonnegative")
    if args.output_tokens is not None and args.output_tokens <= 0:
        parser.error("output-tokens must be positive")
    if args.backend == "tpp" and args.splits != 1:
        parser.error("TPP must use K=1")
    if args.backend == "imbps" and args.splits <= 1:
        parser.error("IMBPS must use K>1")
    if args.mode == "normal" and args.draft_tokens != 0:
        parser.error("normal decode must use --draft-tokens 0")
    if args.mode == "pard" and args.draft_tokens <= 1:
        parser.error("PARD must use at least two speculative tokens")
    return args


def load_claim(name: str) -> dict[str, Any]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    try:
        return registry["full_decode"][name]
    except KeyError as error:
        choices = ", ".join(sorted(registry.get("full_decode", {})))
        raise ValueError(f"unknown full-decode claim {name!r}; choose {choices}") from error


def expected_variants(claim: dict[str, Any]) -> list[dict[str, Any]]:
    backends = [("tpp", 1), *[("imbps", int(k)) for k in claim["splits"]]]
    variants = []
    for backend, split in backends:
        variants.append(
            {
                "mode": "normal",
                "draft_tokens": 0,
                "backend": backend,
                "split": split,
            }
        )
        for draft_tokens in claim["draft_tokens"]:
            variants.append(
                {
                    "mode": "pard",
                    "draft_tokens": int(draft_tokens),
                    "backend": backend,
                    "split": split,
                }
            )
    return variants


class EmissionTimer:
    """Low-overhead streamer that records grouped token emissions."""

    def __init__(self) -> None:
        self.measurement_start_ns = 0
        self.saw_prompt = False
        self.events: list[dict[str, Any]] = []

    def reset(self, start_ns: int) -> None:
        self.measurement_start_ns = start_ns
        self.saw_prompt = False
        self.events = []

    def put(self, value: Any) -> None:
        now_ns = time.perf_counter_ns()
        if not self.saw_prompt:
            self.saw_prompt = True
            return
        token_count = int(value.numel())
        self.events.append(
            {
                "elapsed_ms": (now_ns - self.measurement_start_ns) / 1_000_000,
                "token_count": token_count,
            }
        )

    def end(self) -> None:
        return None


def output_signature(output_ids: Any, context_tokens: int) -> dict[str, Any]:
    generated = output_ids[:, context_tokens:].cpu().reshape(-1).tolist()
    payload = ",".join(str(token) for token in generated).encode("ascii")
    return {
        "generated_token_count": len(generated),
        "generated_token_sha256": hashlib.sha256(payload).hexdigest(),
        "first_generated_token_ids": generated[:16],
        "last_generated_token_ids": generated[-16:],
    }


def summarize_emissions(
    events: list[dict[str, Any]], total_ms: float
) -> dict[str, Any]:
    if not events:
        raise RuntimeError("PACE generate returned without a generated-token emission")
    first_ms = float(events[0]["elapsed_ms"])
    first_count = int(events[0]["token_count"])
    emitted = sum(int(event["token_count"]) for event in events)
    tail_tokens = emitted - first_count
    tail_ms = max(0.0, total_ms - first_ms)
    return {
        "time_to_first_emission_ms": first_ms,
        "first_emission_tokens": first_count,
        "emission_count": len(events),
        "emitted_tokens": emitted,
        "post_first_emission_ms": tail_ms,
        "post_first_tokens_per_second": (
            tail_tokens / (tail_ms / 1000) if tail_tokens > 0 and tail_ms > 0 else None
        ),
        "events": events,
    }


def main() -> None:
    args = parse_args()
    claim = load_claim(args.claim)
    if claim.get("batch") != 1:
        raise ValueError("PACE v1.0 PARD full-decode runner requires batch=1")
    if args.backend == "imbps" and args.splits not in claim["splits"]:
        raise ValueError(f"K={args.splits} is not preregistered")
    if args.mode == "pard" and args.draft_tokens not in claim["draft_tokens"]:
        raise ValueError(f"draft size {args.draft_tokens} is not preregistered")

    contexts = args.contexts or [int(value) for value in claim["context_tokens"]]
    output_tokens = args.output_tokens or int(claim["output_tokens"])
    if len(set(contexts)) != len(contexts):
        raise ValueError("context lengths must be unique")
    os.environ["IMBPS_BLOCK_SIZE"] = str(args.splits)

    import torch
    from pace.llm import (
        KVCacheType,
        LLMBackendType,
        LLMModel,
        LLMOperatorType,
        OperatorConfig,
        PardSpecDecodeConfig,
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
        raise RuntimeError("REQUIRE_TCMALLOC=1 but tcmalloc is not mapped")

    target_reference = args.target_model or claim["target_model"]
    draft_reference = args.draft_model or claim["draft_model"]
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
    pard_config = None
    if args.mode == "pard":
        pard_config = PardSpecDecodeConfig(
            model_name_or_path=draft_reference,
            num_speculative_tokens=args.draft_tokens,
        )

    load_started = time.perf_counter()
    llm = LLMModel(
        target_reference,
        dtype=torch.bfloat16,
        kv_cache_type=KVCacheType.BMC,
        opconfig=opconfig,
        pard_config=pard_config,
        disable_tqdm=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    target_path = Path(llm.generator.model_path).resolve()
    draft_path = (
        Path(llm.generator.pard_model_path).resolve()
        if args.mode == "pard"
        else None
    )
    config = llm.get_config()
    vocab_size = int(config.vocab_size)
    sampling = SamplingConfig(
        max_new_tokens=output_tokens,
        min_new_tokens=output_tokens,
        do_sample=False,
        temperature=0,
        seed=args.seed,
    )
    streamer = EmissionTimer()
    prompts: dict[int, Any] = {}
    for context in contexts:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + context * 97)
        prompts[context] = torch.randint(
            100,
            max(101, vocab_size - 1000),
            (1, context),
            generator=generator,
            dtype=torch.long,
        )

    with torch.inference_mode():
        for context in contexts:
            encoded = BatchEncoding(
                {
                    "input_ids": prompts[context],
                    "attention_mask": torch.ones_like(prompts[context]),
                }
            )
            for _ in range(args.warmups):
                streamer.reset(time.perf_counter_ns())
                llm.generate(encoded, sampling, text_streamer=streamer)

        thread_affinities, affinity_union = collect_thread_affinity()
        if len(affinity_union) < torch.get_num_threads():
            raise RuntimeError(
                f"torch has {torch.get_num_threads()} threads but worker masks cover "
                f"only {len(affinity_union)} CPUs"
            )

        rng = random.Random(args.seed)
        records_by_context: dict[int, list[dict[str, Any]]] = {
            context: [] for context in contexts
        }
        for round_number in range(1, args.rounds + 1):
            order = list(contexts)
            rng.shuffle(order)
            for order_number, context in enumerate(order, start=1):
                encoded = BatchEncoding(
                    {
                        "input_ids": prompts[context],
                        "attention_mask": torch.ones_like(prompts[context]),
                    }
                )
                started_ns = time.perf_counter_ns()
                streamer.reset(started_ns)
                output = llm.generate(encoded, sampling, text_streamer=streamer)
                total_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                signature = output_signature(output.output_token_ids, context)
                emission = summarize_emissions(streamer.events, total_ms)
                generated_tokens = signature["generated_token_count"]
                speculative_rounds: list[int] = []
                if args.mode == "pard":
                    speculative_rounds = [
                        int(value) for value in llm.generator.total_speculated_tokens
                    ]
                accepted_draft = sum(max(0, value - 1) for value in speculative_rounds)
                proposed_draft = len(speculative_rounds) * max(
                    0, args.draft_tokens - 1
                )
                target_forward_iterations = (
                    len(speculative_rounds)
                    if args.mode == "pard"
                    else int(emission["emission_count"])
                )
                if emission["emitted_tokens"] != generated_tokens:
                    raise RuntimeError(
                        "streamed/generated token mismatch: "
                        f"{emission['emitted_tokens']} != {generated_tokens}"
                    )
                record = {
                    "round": round_number,
                    "order": order_number,
                    "context_tokens": context,
                    "requested_output_tokens": output_tokens,
                    "generated_tokens": generated_tokens,
                    "total_ms": total_ms,
                    "output_tokens_per_second": generated_tokens / (total_ms / 1000),
                    "emission": emission,
                    "speculative_iteration_kept_tokens": speculative_rounds,
                    "mean_kept_tokens_per_speculative_iteration": (
                        statistics.fmean(speculative_rounds)
                        if speculative_rounds
                        else None
                    ),
                    "accepted_draft_fraction": (
                        accepted_draft / proposed_draft if proposed_draft else None
                    ),
                    "accepted_draft_tokens": (
                        accepted_draft if args.mode == "pard" else None
                    ),
                    "proposed_draft_tokens": (
                        proposed_draft if args.mode == "pard" else None
                    ),
                    "target_forward_iterations": target_forward_iterations,
                    "generated_tokens_per_target_forward": (
                        generated_tokens / target_forward_iterations
                        if target_forward_iterations
                        else None
                    ),
                    "signature": signature,
                }
                records_by_context[context].append(record)
                print(
                    f"round={round_number} context={context} mode={args.mode} "
                    f"backend={args.backend} K={args.splits} draft={args.draft_tokens} "
                    f"latency_ms={total_ms:.3f} output_tps={record['output_tokens_per_second']:.3f}",
                    flush=True,
                )
                gc.collect()

    records = []
    for context in contexts:
        rounds = records_by_context[context]
        total_times = [float(record["total_ms"]) for record in rounds]
        ttft = [
            float(record["emission"]["time_to_first_emission_ms"])
            for record in rounds
        ]
        throughput = [float(record["output_tokens_per_second"]) for record in rounds]
        post_first_throughput = [
            float(record["emission"]["post_first_tokens_per_second"])
            for record in rounds
            if record["emission"]["post_first_tokens_per_second"] is not None
        ]
        generated_per_target_forward = [
            float(record["generated_tokens_per_target_forward"])
            for record in rounds
            if record["generated_tokens_per_target_forward"] is not None
        ]
        records.append(
            {
                "context_tokens": context,
                "rounds": rounds,
                "summary": {
                    "median_total_ms": statistics.median(total_times),
                    "q1_total_ms": percentile(total_times, 0.25),
                    "q3_total_ms": percentile(total_times, 0.75),
                    "median_ttft_ms": statistics.median(ttft),
                    "median_output_tokens_per_second": statistics.median(throughput),
                    "median_post_first_tokens_per_second": (
                        statistics.median(post_first_throughput)
                        if post_first_throughput
                        else None
                    ),
                    "median_generated_tokens_per_target_forward": (
                        statistics.median(generated_per_target_forward)
                        if generated_per_target_forward
                        else None
                    ),
                },
            }
        )

    result = {
        "schema_version": 1,
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_kind": "full_autoregressive_and_pard_generation",
        "scope_note": (
            "Public PACE generate path including prefill, attention, populated KV "
            "caches, MLP, LM head, greedy sampling, and iterative generation. PARD "
            "also includes draft-model inference, target verification, acceptance, "
            "and rollback. Model loading and warmups are excluded."
        ),
        "claim": args.claim,
        "mode": args.mode,
        "backend": args.backend,
        "split": args.splits,
        "draft_tokens": args.draft_tokens,
        "expected_variants": expected_variants(claim),
        "model": {
            "target_requested_reference": target_reference,
            "target_resolved_path": str(target_path),
            "target_snapshot_commit": snapshot_commit(target_path),
            "draft_requested_reference": draft_reference if args.mode == "pard" else None,
            "draft_resolved_path": str(draft_path) if draft_path else None,
            "draft_snapshot_commit": snapshot_commit(draft_path) if draft_path else None,
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": getattr(config, "intermediate_size", None),
            "vocab_size": vocab_size,
            "dtype": "bf16",
        },
        "runner": {
            "contexts": contexts,
            "output_tokens": output_tokens,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "seed": args.seed,
            "model_load_seconds": model_load_seconds,
            "timing_boundary": "LLMModel.generate call",
        },
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
