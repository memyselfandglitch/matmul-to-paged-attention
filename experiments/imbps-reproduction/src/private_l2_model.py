#!/usr/bin/env python3
"""Model IMBPS residency when L2 is private to each worker core.

The paper-style equation treats cache capacity as one pool.  That is an
optimistic bound on EPYC because every core has a private L2.  This module
reports that bound together with two progressively more conservative
ownership assumptions.  It deliberately does not claim to know oneDNN's
internal GEMM tile schedule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .cache_model import MIB
except ImportError:  # Direct execution: python3 src/private_l2_model.py
    from cache_model import MIB


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def strict_candidate(variable_bytes: float, available_bytes: float) -> dict[str, Any]:
    """Solve ``available > variable / K`` using the paper's strict bound."""

    if available_bytes <= 0:
        return {
            "finite": False,
            "strict_lower_bound": None,
            "strict_integer_candidate": None,
            "next_power_of_two_candidate": None,
        }
    lower = variable_bytes / available_bytes
    integer = math.floor(lower) + 1
    power = 1 << max(0, (integer - 1).bit_length())
    return {
        "finite": True,
        "strict_lower_bound": lower,
        "strict_integer_candidate": integer,
        "next_power_of_two_candidate": power,
    }


def private_l2_working_set(
    rows: int,
    hidden: int,
    intermediate: int,
    split: int,
    workers: int,
    l2_mib_per_worker: float = 1.0,
    bytes_per_element: int = 2,
) -> dict[str, Any]:
    """Return per-worker residency bounds for one IMBPS block.

    Models:

    ``perfect_partition``
        Every term is evenly owned by the active workers.  This is exactly the
        aggregate-private-L2 interpretation used by the earlier Equation-13
        analysis and is an optimistic lower bound.

    ``source_replicated``
        Every worker must retain the complete MxH source while the split terms
        are partitioned.  This is conservative for a row-tiled up projection.

    ``activations_replicated``
        Every worker retains the complete source and current Mx(I/K)
        intermediate block; only weights are evenly partitioned.  This is the
        most conservative bound reported here.
    """

    if min(rows, hidden, intermediate, split, workers, bytes_per_element) <= 0:
        raise ValueError("shape, split, workers, and element size must be positive")
    if l2_mib_per_worker <= 0:
        raise ValueError("per-worker L2 capacity must be positive")

    source = bytes_per_element * rows * hidden
    intermediate_block = bytes_per_element * rows * intermediate / split
    weight_block = bytes_per_element * hidden * intermediate / split
    split_unsplit = bytes_per_element * intermediate * (rows + hidden)
    capacity = l2_mib_per_worker * MIB

    perfect = (source + intermediate_block + weight_block) / workers
    source_replicated = source + (intermediate_block + weight_block) / workers
    activations_replicated = source + intermediate_block + weight_block / workers

    return {
        "rows": rows,
        "hidden": hidden,
        "intermediate": intermediate,
        "split": split,
        "workers": workers,
        "bytes_per_element": bytes_per_element,
        "l2_mib_per_worker": l2_mib_per_worker,
        "aggregate_private_l2_mib": workers * l2_mib_per_worker,
        "terms_mib": {
            "source_mh": source / MIB,
            "intermediate_block_mi_over_k": intermediate_block / MIB,
            "weight_block_hi_over_k": weight_block / MIB,
        },
        "models": {
            "perfect_partition": {
                "per_worker_mib": perfect / MIB,
                "fits_strict": perfect < capacity,
            },
            "source_replicated": {
                "per_worker_mib": source_replicated / MIB,
                "fits_strict": source_replicated < capacity,
            },
            "activations_replicated": {
                "per_worker_mib": activations_replicated / MIB,
                "fits_strict": activations_replicated < capacity,
            },
        },
        "equation_candidates": {
            "perfect_partition": strict_candidate(
                split_unsplit, workers * capacity - source
            ),
            "source_replicated": strict_candidate(
                split_unsplit / workers, capacity - source
            ),
            "activations_replicated": strict_candidate(
                bytes_per_element * rows * intermediate
                + bytes_per_element * hidden * intermediate / workers,
                capacity - source,
            ),
        },
        "interpretation": (
            "PACE v1.0 does not assign one IMBPS block to one core. The perfect-"
            "partition result is an optimistic capacity bound; the replicated "
            "bounds show how the answer changes when oneDNN tile ownership does "
            "not evenly partition activations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=comma_ints, required=True)
    parser.add_argument("--splits", type=comma_ints, required=True)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=28672)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--l2-mib-per-worker", type=float, default=1.0)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        private_l2_working_set(
            rows,
            args.hidden,
            args.intermediate,
            split,
            args.workers,
            args.l2_mib_per_worker,
            args.bytes_per_element,
        )
        for rows in args.rows
        for split in args.splits
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "model_kind": "private_l2_residency_bounds",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
