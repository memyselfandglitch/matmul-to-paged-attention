"""Hugging Face OPT layer and end-to-end benchmark runners."""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import socket
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from imbps_bench.hf_opt import (
    HFOPTIMBPSMLP,
    HFOPTReferenceMLP,
    OPTModelIMBPSPatcher,
    find_opt_decoder_layers,
)
from imbps_bench.metadata import collect_metadata
from imbps_bench.runner import DTYPES, error_metrics, tolerances


def _transformers():
    try:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face benchmarks require the optional dependencies; "
            "run ./scripts/bootstrap_hf_venv.sh"
        ) from error
    return transformers, AutoConfig, AutoModelForCausalLM, AutoTokenizer, OPTDecoderLayer


def _set_threads(threads: int) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _json_write(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_load_kwargs(dtype: torch.dtype, local_files_only: bool, attn_implementation: str):
    return {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }


def _opt_config_metadata(config) -> Dict[str, Any]:
    keys = (
        "model_type",
        "hidden_size",
        "word_embed_proj_dim",
        "ffn_dim",
        "num_hidden_layers",
        "num_attention_heads",
        "max_position_embeddings",
        "vocab_size",
        "activation_function",
        "enable_bias",
        "do_layer_norm_before",
        "dropout",
        "attention_dropout",
        "layerdrop",
    )
    result = {key: getattr(config, key, None) for key in keys}
    result["commit_hash"] = getattr(config, "_commit_hash", None)
    return result


def _module_size(module: torch.nn.Module) -> Dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    }


@dataclass(frozen=True)
class HFOPTLayerConfig:
    model_name_or_path: str
    model_mode: str
    layer_index: int
    input_source: str
    batch_size: int
    sequence_length: int
    dtype_name: str
    splits: Sequence[int]
    threads: int
    warmup: int
    repeats: int
    seed: int
    weight_layout: str
    accumulation_dtype: str
    attn_implementation: str
    local_files_only: bool
    output_dir: Path


class _ActivationCaptured(Exception):
    pass


def _capture_fc1_input(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs):
        del module
        captured["hidden_states"] = inputs[0].detach().clone()
        raise _ActivationCaptured()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    handle = layer.fc1.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _ActivationCaptured:
        pass
    finally:
        handle.remove()
    if "hidden_states" not in captured:
        raise RuntimeError("failed to capture the input to the selected OPT fc1 module")
    return captured["hidden_states"]


def _load_layer_fixture(config: HFOPTLayerConfig):
    transformers, AutoConfig, AutoModelForCausalLM, _, OPTDecoderLayer = _transformers()
    dtype = DTYPES[config.dtype_name]
    hf_config = AutoConfig.from_pretrained(
        config.model_name_or_path, local_files_only=config.local_files_only
    )
    if getattr(hf_config, "model_type", None) != "opt":
        raise ValueError("hf-opt-layer requires an OPT model/configuration")
    setattr(hf_config, "_attn_implementation", config.attn_implementation)

    full_model = None
    if config.model_mode == "pretrained":
        full_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **_model_load_kwargs(dtype, config.local_files_only, config.attn_implementation),
        )
        full_model.eval()
        layers = find_opt_decoder_layers(full_model)
    elif config.model_mode == "random-config":
        layer = OPTDecoderLayer(hf_config).to(dtype=dtype, device="cpu").eval()
        layers = [layer]
    else:
        raise ValueError("model_mode must be 'pretrained' or 'random-config'")

    if config.layer_index < 0 or config.layer_index >= len(layers):
        raise ValueError("layer_index is outside the loaded decoder")
    layer = layers[config.layer_index]
    activation_name = str(getattr(hf_config, "activation_function", "unknown"))

    if config.input_source == "captured":
        if full_model is None:
            raise ValueError("captured inputs require --model-mode pretrained")
        hidden_states = _capture_fc1_input(
            full_model,
            layer,
            config.batch_size,
            config.sequence_length,
            int(hf_config.vocab_size),
            config.seed + 1,
        )
    elif config.input_source == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 1)
        hidden_states = torch.empty(
            (config.batch_size * config.sequence_length, int(hf_config.hidden_size)),
            dtype=dtype,
            device="cpu",
        ).normal_(mean=0.0, std=float(getattr(hf_config, "init_std", 0.02)), generator=generator)
    else:
        raise ValueError("input_source must be 'captured' or 'random'")

    software = {
        "transformers_version": transformers.__version__,
        "activation_function": activation_name,
        "enable_bias": bool(getattr(hf_config, "enable_bias", layer.fc1.bias is not None)),
        "config_class": type(hf_config).__name__,
        "layer_class": type(layer).__name__,
        "opt_config": _opt_config_metadata(hf_config),
        "loaded_module": _module_size(full_model if full_model is not None else layer),
    }
    return full_model, layer, hidden_states, activation_name, software


LAYER_RAW_FIELDS = (
    "run_id", "timestamp_utc", "hostname", "variant", "model_name_or_path", "model_mode",
    "layer_index", "input_source", "batch_size", "sequence_length", "rows", "hidden_size",
    "intermediate_size", "activation_function", "fc1_bias", "fc2_bias", "dtype", "threads",
    "weight_layout", "accumulation_dtype", "split_k", "iteration", "latency_ns", "latency_ms", "rows_per_second",
    "checksum", "packing_ms", "canonical_parameter_bytes", "packed_parameter_bytes",
    "workspace_bytes", "logical_activation_bytes", "logical_output_bytes",
    "max_abs_error", "max_rel_error", "allclose",
)


LAYER_SUMMARY_FIELDS = (
    "run_id", "variant", "model_name_or_path", "model_mode", "layer_index", "input_source",
    "batch_size", "sequence_length", "rows", "hidden_size", "intermediate_size",
    "activation_function", "dtype", "threads", "weight_layout", "accumulation_dtype", "split_k", "samples",
    "latency_min_ms", "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
    "latency_stdev_ms", "reference_median_ms", "speedup_vs_reference", "rows_per_second",
    "packing_ms", "canonical_parameter_bytes", "packed_parameter_bytes", "workspace_bytes",
    "logical_activation_bytes", "logical_output_bytes", "max_abs_error", "max_rel_error", "allclose",
)


def _summarize_layer_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["split_k"]), str(row["variant"])), []).append(row)
    result = []
    for split_k in sorted({key[0] for key in groups}):
        reference = groups[(split_k, "reference")]
        reference_ms = statistics.median(float(row["latency_ms"]) for row in reference)
        for variant in ("reference", "imbps"):
            group = groups[(split_k, variant)]
            latencies = [float(row["latency_ms"]) for row in group]
            median_ms = statistics.median(latencies)
            first = group[0]
            summary = {field: first.get(field, "") for field in LAYER_SUMMARY_FIELDS}
            summary.update(
                {
                    "samples": len(group),
                    "latency_min_ms": min(latencies),
                    "latency_mean_ms": statistics.fmean(latencies),
                    "latency_median_ms": median_ms,
                    "latency_p95_ms": _percentile(latencies, 0.95),
                    "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    "reference_median_ms": reference_ms,
                    "speedup_vs_reference": reference_ms / median_ms,
                    "rows_per_second": float(first["rows"]) / (median_ms / 1000.0),
                }
            )
            result.append(summary)
    return result


def run_hf_opt_layer(config: HFOPTLayerConfig) -> Dict[str, Any]:
    if config.dtype_name not in DTYPES:
        raise ValueError("unsupported dtype: %s" % config.dtype_name)
    _set_threads(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "%s-hf-opt-layer" % _utc_stamp()
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["splits"] = list(config.splits)
    metadata_path = config.output_dir / "metadata.json"
    raw_path = config.output_dir / "raw.csv"
    summary_path = config.output_dir / "summary.csv"
    manifest_path = config.output_dir / "manifest.json"

    full_model, layer, hidden_states, activation_name, software = _load_layer_fixture(config)
    collect_metadata(metadata_path, arguments={**config_dict, **software})
    reference = HFOPTReferenceMLP(layer.fc1, layer.activation_fn, layer.fc2).eval()
    dtype = DTYPES[config.dtype_name]
    rtol, atol = tolerances(dtype)
    rng = random.Random(config.seed)
    rows: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for split_k in config.splits:
            split = HFOPTIMBPSMLP(
                layer.fc1,
                layer.activation_fn,
                layer.fc2,
                split_k,
                config.weight_layout,
                activation_name,
                config.accumulation_dtype,
            ).eval()
            expected = reference(hidden_states).clone()
            actual = split(hidden_states).clone()
            max_abs, max_rel = error_metrics(actual, expected)
            allclose = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
            if not allclose:
                raise RuntimeError(
                    "HF OPT layer correctness failed for K=%d: max_abs=%g max_rel=%g"
                    % (split_k, max_abs, max_rel)
                )
            for _ in range(config.warmup):
                reference(hidden_states)
                split(hidden_states)

            for iteration in range(config.repeats):
                order = ["reference", "imbps"]
                rng.shuffle(order)
                for variant in order:
                    function = reference if variant == "reference" else split
                    started = perf_counter_ns()
                    output = function(hidden_states)
                    elapsed_ns = perf_counter_ns() - started
                    row = {
                        "run_id": run_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hostname": socket.gethostname(),
                        "variant": variant,
                        "model_name_or_path": config.model_name_or_path,
                        "model_mode": config.model_mode,
                        "layer_index": config.layer_index,
                        "input_source": config.input_source,
                        "batch_size": config.batch_size,
                        "sequence_length": config.sequence_length,
                        "rows": hidden_states.numel() // hidden_states.shape[-1],
                        "hidden_size": layer.fc1.in_features,
                        "intermediate_size": layer.fc1.out_features,
                        "activation_function": activation_name,
                        "fc1_bias": layer.fc1.bias is not None,
                        "fc2_bias": layer.fc2.bias is not None,
                        "dtype": config.dtype_name,
                        "threads": torch.get_num_threads(),
                        "weight_layout": "native_linear" if variant == "reference" else config.weight_layout,
                        "accumulation_dtype": "native" if variant == "reference" else config.accumulation_dtype,
                        "split_k": split_k,
                        "iteration": iteration,
                        "latency_ns": elapsed_ns,
                        "latency_ms": elapsed_ns / 1_000_000.0,
                        "rows_per_second": (hidden_states.numel() // hidden_states.shape[-1]) / (elapsed_ns / 1e9),
                        "checksum": float(output.reshape(-1)[0].float().item()),
                        "packing_ms": split.packing_ms if variant == "imbps" else 0.0,
                        "canonical_parameter_bytes": split.canonical_parameter_bytes,
                        "packed_parameter_bytes": split.packed_parameter_bytes if variant == "imbps" else 0,
                        "workspace_bytes": split.workspace_bytes if variant == "imbps" else 0,
                        "logical_activation_bytes": (
                            (hidden_states.numel() // hidden_states.shape[-1])
                            * (split.max_width if variant == "imbps" else layer.fc1.out_features)
                            * hidden_states.element_size()
                        ),
                        "logical_output_bytes": hidden_states.numel() * hidden_states.element_size(),
                        "max_abs_error": max_abs,
                        "max_rel_error": max_rel,
                        "allclose": allclose,
                    }
                    rows.append(row)
            del split
            gc.collect()

    summaries = _summarize_layer_rows(rows)
    _write_csv(raw_path, LAYER_RAW_FIELDS, rows)
    _write_csv(summary_path, LAYER_SUMMARY_FIELDS, summaries)
    _json_write(
        manifest_path,
        {
            "run_id": run_id,
            "status": "complete",
            "config": config_dict,
            "software": software,
            "row_count": len(rows),
            "paths": {"raw_csv": str(raw_path), "summary_csv": str(summary_path), "metadata": str(metadata_path)},
        },
    )
    del reference, layer, hidden_states, full_model
    gc.collect()
    best = max(
        (row for row in summaries if row["variant"] == "imbps"),
        key=lambda row: float(row["speedup_vs_reference"]),
    )
    return {
        "run_id": run_id,
        "output_dir": str(config.output_dir),
        "activation_function": activation_name,
        "best_imbps": best,
    }


@dataclass(frozen=True)
class HFOPTE2EConfig:
    model_name_or_path: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    prompt: Optional[str]
    dtype_name: str
    splits: Sequence[int]
    threads: int
    warmup: int
    repeats: int
    seed: int
    weight_layout: str
    accumulation_dtype: str
    attn_implementation: str
    local_files_only: bool
    allow_correctness_failure: bool
    output_dir: Path


@dataclass
class GenerationMeasurement:
    prefill_ms: float
    ttft_ms: float
    decode_ms: float
    total_ms: float
    generated_ids: torch.Tensor
    first_token_logits: torch.Tensor
    decode_token_latencies_ms: Sequence[float]


def _make_e2e_inputs(config: HFOPTE2EConfig, model_config, tokenizer_class):
    if config.prompt is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + 1)
        input_ids = torch.randint(
            0,
            int(model_config.vocab_size),
            (config.batch_size, config.input_tokens),
            generator=generator,
            dtype=torch.long,
        )
        return input_ids, torch.ones_like(input_ids), "random_token_ids"

    tokenizer = tokenizer_class.from_pretrained(
        config.model_name_or_path, local_files_only=config.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation must end each row at a real prompt token. Left
    # padding keeps the last position valid when the prompt is shorter than the
    # requested fixed input length.
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [config.prompt] * config.batch_size,
        padding="max_length",
        truncation=True,
        max_length=config.input_tokens,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], "tokenized_prompt"


def _manual_greedy_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_tokens: int,
) -> GenerationMeasurement:
    started = perf_counter_ns()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    prefill_end = perf_counter_ns()
    first_logits = outputs.logits[:, -1, :].detach().clone()
    next_token = first_logits.argmax(dim=-1, keepdim=True)
    first_token_end = perf_counter_ns()
    generated = [next_token]
    past_key_values = outputs.past_key_values
    token_latencies = []

    for _ in range(1, output_tokens):
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
            dim=1,
        )
        token_started = perf_counter_ns()
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        token_end = perf_counter_ns()
        token_latencies.append((token_end - token_started) / 1_000_000.0)
        generated.append(next_token)

    completed = perf_counter_ns()
    return GenerationMeasurement(
        prefill_ms=(prefill_end - started) / 1_000_000.0,
        ttft_ms=(first_token_end - started) / 1_000_000.0,
        decode_ms=(completed - first_token_end) / 1_000_000.0,
        total_ms=(completed - started) / 1_000_000.0,
        generated_ids=torch.cat(generated, dim=1),
        first_token_logits=first_logits,
        decode_token_latencies_ms=tuple(token_latencies),
    )


E2E_RAW_FIELDS = (
    "run_id", "timestamp_utc", "hostname", "variant", "model_name_or_path", "input_mode",
    "batch_size", "input_tokens", "output_tokens", "activation_function", "dtype", "threads",
    "hidden_size", "intermediate_size", "attn_implementation", "weight_layout", "accumulation_dtype", "split_k",
    "iteration", "prefill_ms", "ttft_ms",
    "decode_ms", "total_ms", "output_tokens_per_second", "decode_tokens_per_second",
    "mean_decode_token_ms", "p95_decode_token_ms", "first_token_id_checksum", "packing_ms",
    "packed_parameter_bytes", "workspace_bytes", "prefill_activation_bytes_per_layer",
    "decode_activation_bytes_per_layer", "max_abs_first_logits", "max_rel_first_logits",
    "first_logits_allclose", "generated_tokens_match",
)


E2E_SUMMARY_FIELDS = (
    "run_id", "variant", "model_name_or_path", "input_mode", "batch_size", "input_tokens",
    "output_tokens", "activation_function", "dtype", "threads", "attn_implementation",
    "hidden_size", "intermediate_size", "weight_layout", "accumulation_dtype", "split_k", "samples",
    "prefill_median_ms", "prefill_p95_ms",
    "ttft_median_ms", "ttft_p95_ms", "decode_median_ms", "total_median_ms",
    "output_tokens_per_second_at_median", "decode_tokens_per_second_at_median",
    "reference_prefill_median_ms", "prefill_speedup_vs_reference", "reference_ttft_median_ms",
    "ttft_speedup_vs_reference", "reference_total_median_ms", "total_speedup_vs_reference",
    "packing_ms", "packed_parameter_bytes", "workspace_bytes", "prefill_activation_bytes_per_layer",
    "decode_activation_bytes_per_layer", "max_abs_first_logits",
    "max_rel_first_logits", "first_logits_allclose", "generated_tokens_match",
)


def _summarize_e2e_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["split_k"]), str(row["variant"])), []).append(row)
    summaries = []
    for split_k in sorted({key[0] for key in groups}):
        ref = groups[(split_k, "reference")]
        ref_prefill = statistics.median(float(row["prefill_ms"]) for row in ref)
        ref_ttft = statistics.median(float(row["ttft_ms"]) for row in ref)
        ref_total = statistics.median(float(row["total_ms"]) for row in ref)
        for variant in ("reference", "imbps"):
            group = groups[(split_k, variant)]
            prefill = [float(row["prefill_ms"]) for row in group]
            ttft = [float(row["ttft_ms"]) for row in group]
            decode = [float(row["decode_ms"]) for row in group]
            total = [float(row["total_ms"]) for row in group]
            first = group[0]
            median_prefill = statistics.median(prefill)
            median_ttft = statistics.median(ttft)
            median_decode = statistics.median(decode)
            median_total = statistics.median(total)
            output_count = int(first["batch_size"]) * int(first["output_tokens"])
            decode_count = int(first["batch_size"]) * max(int(first["output_tokens"]) - 1, 0)
            summary = {field: first.get(field, "") for field in E2E_SUMMARY_FIELDS}
            summary.update(
                {
                    "samples": len(group),
                    "prefill_median_ms": median_prefill,
                    "prefill_p95_ms": _percentile(prefill, 0.95),
                    "ttft_median_ms": median_ttft,
                    "ttft_p95_ms": _percentile(ttft, 0.95),
                    "decode_median_ms": median_decode,
                    "total_median_ms": median_total,
                    "output_tokens_per_second_at_median": output_count / (median_total / 1000.0),
                    "decode_tokens_per_second_at_median": (
                        decode_count / (median_decode / 1000.0) if decode_count and median_decode else math.nan
                    ),
                    "reference_prefill_median_ms": ref_prefill,
                    "prefill_speedup_vs_reference": ref_prefill / median_prefill,
                    "reference_ttft_median_ms": ref_ttft,
                    "ttft_speedup_vs_reference": ref_ttft / median_ttft,
                    "reference_total_median_ms": ref_total,
                    "total_speedup_vs_reference": ref_total / median_total,
                }
            )
            summaries.append(summary)
    return summaries


def run_hf_opt_e2e(config: HFOPTE2EConfig) -> Dict[str, Any]:
    if config.dtype_name not in DTYPES:
        raise ValueError("unsupported dtype: %s" % config.dtype_name)
    if config.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    _set_threads(config.threads)
    transformers, AutoConfig, AutoModelForCausalLM, AutoTokenizer, _ = _transformers()
    dtype = DTYPES[config.dtype_name]
    model_config = AutoConfig.from_pretrained(
        config.model_name_or_path, local_files_only=config.local_files_only
    )
    if getattr(model_config, "model_type", None) != "opt":
        raise ValueError("hf-opt-e2e requires an OPT model")
    max_positions = int(getattr(model_config, "max_position_embeddings", 0))
    if max_positions and config.input_tokens + config.output_tokens > max_positions:
        raise ValueError("input_tokens + output_tokens exceeds the model position limit")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **_model_load_kwargs(dtype, config.local_files_only, config.attn_implementation),
    ).eval()
    input_ids, attention_mask, input_mode = _make_e2e_inputs(config, model_config, AutoTokenizer)
    activation_name = str(getattr(model_config, "activation_function", "unknown"))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "%s-hf-opt-e2e" % _utc_stamp()
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["splits"] = list(config.splits)
    metadata_path = config.output_dir / "metadata.json"
    raw_path = config.output_dir / "raw.csv"
    summary_path = config.output_dir / "summary.csv"
    manifest_path = config.output_dir / "manifest.json"
    collect_metadata(
        metadata_path,
        arguments={
            **config_dict,
            "transformers_version": transformers.__version__,
            "input_mode": input_mode,
            "activation_function": activation_name,
            "measurement": "manual fixed-length greedy generation with KV cache",
            "opt_config": _opt_config_metadata(model_config),
            "loaded_module": _module_size(model),
        },
    )

    rtol, atol = tolerances(dtype)
    rng = random.Random(config.seed)
    rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for split_k in config.splits:
            print("[imbps:hf-e2e] K=%d: packing all OPT MLP layers" % split_k, file=sys.stderr, flush=True)
            patcher = OPTModelIMBPSPatcher(
                model,
                split_k,
                config.weight_layout,
                activation_name,
                config.accumulation_dtype,
            )
            patcher.enable_reference()
            expected = _manual_greedy_generate(
                model, input_ids, attention_mask, config.output_tokens
            )
            patcher.enable_imbps()
            actual = _manual_greedy_generate(
                model, input_ids, attention_mask, config.output_tokens
            )
            max_abs, max_rel = error_metrics(actual.first_token_logits, expected.first_token_logits)
            logits_allclose = bool(
                torch.allclose(actual.first_token_logits, expected.first_token_logits, rtol=rtol, atol=atol)
            )
            token_match = bool(torch.equal(actual.generated_ids, expected.generated_ids))
            if not logits_allclose or not token_match:
                message = (
                    "end-to-end correctness failed for K=%d: "
                    "first_logits_allclose=%s generated_tokens_match=%s "
                    "max_abs=%g max_rel=%g"
                    % (split_k, logits_allclose, token_match, max_abs, max_rel)
                )
                if not config.allow_correctness_failure:
                    patcher.close()
                    raise RuntimeError(message)
                print("[imbps:hf-e2e] WARNING: " + message, file=sys.stderr, flush=True)

            for _ in range(config.warmup):
                patcher.enable_reference()
                _manual_greedy_generate(model, input_ids, attention_mask, config.output_tokens)
                patcher.enable_imbps()
                _manual_greedy_generate(model, input_ids, attention_mask, config.output_tokens)

            for iteration in range(config.repeats):
                order = ["reference", "imbps"]
                rng.shuffle(order)
                for variant in order:
                    if variant == "reference":
                        patcher.enable_reference()
                    else:
                        patcher.enable_imbps()
                    measured = _manual_greedy_generate(
                        model, input_ids, attention_mask, config.output_tokens
                    )
                    decode_count = config.batch_size * max(config.output_tokens - 1, 0)
                    output_count = config.batch_size * config.output_tokens
                    decode_tps = (
                        decode_count / (measured.decode_ms / 1000.0)
                        if decode_count and measured.decode_ms
                        else math.nan
                    )
                    rows.append(
                        {
                            "run_id": run_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "hostname": socket.gethostname(),
                            "variant": variant,
                            "model_name_or_path": config.model_name_or_path,
                            "input_mode": input_mode,
                            "batch_size": config.batch_size,
                            "input_tokens": input_ids.shape[1],
                            "output_tokens": config.output_tokens,
                            "activation_function": activation_name,
                            "dtype": config.dtype_name,
                            "threads": torch.get_num_threads(),
                            "hidden_size": int(model_config.hidden_size),
                            "intermediate_size": int(model_config.ffn_dim),
                            "attn_implementation": config.attn_implementation,
                            "weight_layout": "native_linear" if variant == "reference" else config.weight_layout,
                            "accumulation_dtype": "native" if variant == "reference" else config.accumulation_dtype,
                            "split_k": split_k,
                            "iteration": iteration,
                            "prefill_ms": measured.prefill_ms,
                            "ttft_ms": measured.ttft_ms,
                            "decode_ms": measured.decode_ms,
                            "total_ms": measured.total_ms,
                            "output_tokens_per_second": output_count / (measured.total_ms / 1000.0),
                            "decode_tokens_per_second": decode_tps,
                            "mean_decode_token_ms": (
                                statistics.fmean(measured.decode_token_latencies_ms)
                                if measured.decode_token_latencies_ms else math.nan
                            ),
                            "p95_decode_token_ms": _percentile(measured.decode_token_latencies_ms, 0.95),
                            "first_token_id_checksum": int(measured.generated_ids[:, 0].sum().item()),
                            "packing_ms": patcher.packing_ms if variant == "imbps" else 0.0,
                            "packed_parameter_bytes": patcher.packed_parameter_bytes if variant == "imbps" else 0,
                            "workspace_bytes": patcher.workspace_bytes if variant == "imbps" else 0,
                            "prefill_activation_bytes_per_layer": (
                                config.batch_size
                                * input_ids.shape[1]
                                * (
                                    math.ceil(int(model_config.ffn_dim) / split_k)
                                    if variant == "imbps"
                                    else int(model_config.ffn_dim)
                                )
                                * torch.empty((), dtype=dtype).element_size()
                            ),
                            "decode_activation_bytes_per_layer": (
                                config.batch_size
                                * (
                                    math.ceil(int(model_config.ffn_dim) / split_k)
                                    if variant == "imbps"
                                    else int(model_config.ffn_dim)
                                )
                                * torch.empty((), dtype=dtype).element_size()
                            ),
                            "max_abs_first_logits": max_abs,
                            "max_rel_first_logits": max_rel,
                            "first_logits_allclose": logits_allclose,
                            "generated_tokens_match": token_match,
                        }
                    )
            patcher.close()
            del patcher, expected, actual
            gc.collect()

    summaries = _summarize_e2e_rows(rows)
    _write_csv(raw_path, E2E_RAW_FIELDS, rows)
    _write_csv(summary_path, E2E_SUMMARY_FIELDS, summaries)
    _json_write(
        manifest_path,
        {
            "run_id": run_id,
            "status": "complete",
            "config": config_dict,
            "transformers_version": transformers.__version__,
            "input_mode": input_mode,
            "activation_function": activation_name,
            "row_count": len(rows),
            "paths": {"raw_csv": str(raw_path), "summary_csv": str(summary_path), "metadata": str(metadata_path)},
        },
    )
    best = max(
        (row for row in summaries if row["variant"] == "imbps"),
        key=lambda row: float(row["ttft_speedup_vs_reference"]),
    )
    return {
        "run_id": run_id,
        "output_dir": str(config.output_dir),
        "input_mode": input_mode,
        "activation_function": activation_name,
        "best_imbps_by_ttft": best,
    }
