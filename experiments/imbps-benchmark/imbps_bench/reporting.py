"""CSV schemas and summary calculations."""

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


RAW_FIELDS = (
    "run_id",
    "timestamp_utc",
    "hostname",
    "pair_index",
    "order_in_pair",
    "variant",
    "mlp_kind",
    "weight_layout",
    "dtype",
    "threads",
    "tokens",
    "hidden_size",
    "intermediate_size",
    "split_k",
    "iteration",
    "latency_ns",
    "latency_ms",
    "tokens_per_second",
    "gflops",
    "checksum",
    "packing_ms",
    "canonical_weight_bytes",
    "packed_weight_bytes",
    "workspace_bytes",
    "input_bytes",
    "output_bytes",
    "max_abs_error",
    "max_rel_error",
    "allclose",
    "seed",
    "gelu_approximate",
)


SUMMARY_FIELDS = (
    "run_id",
    "mlp_kind",
    "variant",
    "weight_layout",
    "dtype",
    "threads",
    "tokens",
    "hidden_size",
    "intermediate_size",
    "split_k",
    "samples",
    "latency_min_ms",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_stdev_ms",
    "tokens_per_second_at_median",
    "gflops_at_median",
    "reference_median_ms",
    "speedup_vs_reference",
    "packing_ms",
    "workspace_bytes",
    "packed_weight_bytes",
    "max_abs_error",
    "max_rel_error",
    "allclose",
)


def write_raw_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()


def append_raw_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RAW_FIELDS})
        handle.flush()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row["run_id"],
        row["mlp_kind"],
        row["variant"],
        row["weight_layout"],
        row["dtype"],
        int(row["threads"]),
        int(row["tokens"]),
        int(row["hidden_size"]),
        int(row["intermediate_size"]),
        int(row["split_k"]),
    )


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    reference_medians: Dict[Tuple[Any, ...], float] = {}
    for key, group in groups.items():
        if key[2] != "reference":
            continue
        pair_key = key[0:2] + key[4:]
        reference_medians[pair_key] = statistics.median(float(row["latency_ms"]) for row in group)

    summaries: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        group = groups[key]
        latencies = [float(row["latency_ms"]) for row in group]
        median_ms = statistics.median(latencies)
        pair_key = key[0:2] + key[4:]
        reference_ms = reference_medians.get(pair_key, math.nan)
        first = group[0]
        summaries.append(
            {
                "run_id": key[0],
                "mlp_kind": key[1],
                "variant": key[2],
                "weight_layout": key[3],
                "dtype": key[4],
                "threads": key[5],
                "tokens": key[6],
                "hidden_size": key[7],
                "intermediate_size": key[8],
                "split_k": key[9],
                "samples": len(group),
                "latency_min_ms": min(latencies),
                "latency_mean_ms": statistics.fmean(latencies),
                "latency_median_ms": median_ms,
                "latency_p95_ms": _percentile(latencies, 0.95),
                "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "tokens_per_second_at_median": float(first["tokens"]) / (median_ms / 1000.0),
                "gflops_at_median": float(first["gflops"]) * float(first["latency_ms"]) / median_ms,
                "reference_median_ms": reference_ms,
                "speedup_vs_reference": reference_ms / median_ms if median_ms and not math.isnan(reference_ms) else math.nan,
                "packing_ms": first["packing_ms"],
                "workspace_bytes": first["workspace_bytes"],
                "packed_weight_bytes": first["packed_weight_bytes"],
                "max_abs_error": first["max_abs_error"],
                "max_rel_error": first["max_rel_error"],
                "allclose": first["allclose"],
            }
        )
    return summaries


def write_summary(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

