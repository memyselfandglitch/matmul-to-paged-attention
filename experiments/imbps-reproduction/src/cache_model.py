#!/usr/bin/env python3
"""Evaluate IMBPS Equations 12 and 13 without hiding unit assumptions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


MIB = 1024**2


@dataclass(frozen=True)
class WorkingSet:
    input_bytes: int
    split_activation_bytes: int
    split_weight_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.input_bytes + self.split_activation_bytes + self.split_weight_bytes


def working_set_bytes(
    batch: int,
    sequence: int,
    hidden: int,
    intermediate: int,
    splits: int,
    bytes_per_element: int,
) -> WorkingSet:
    if min(batch, sequence, hidden, intermediate, splits, bytes_per_element) <= 0:
        raise ValueError("all dimensions, splits, and bytes-per-element must be positive")
    if intermediate % splits != 0:
        raise ValueError("intermediate dimension must be divisible by splits")
    input_bytes = bytes_per_element * batch * sequence * hidden
    split_activation_bytes = (
        bytes_per_element * batch * sequence * intermediate // splits
    )
    split_weight_bytes = bytes_per_element * hidden * intermediate // splits
    return WorkingSet(input_bytes, split_activation_bytes, split_weight_bytes)


def equation13_lower_bound(
    batch: int,
    sequence: int,
    hidden: int,
    intermediate: int,
    bytes_per_element: int,
    cache_bytes: int,
) -> float:
    """Return the strict lower bound on K, or infinity if no K can fit."""
    if min(batch, sequence, hidden, intermediate, bytes_per_element, cache_bytes) <= 0:
        raise ValueError("all arguments must be positive")
    fixed_input = bytes_per_element * batch * sequence * hidden
    denominator = cache_bytes - fixed_input
    if denominator <= 0:
        return math.inf
    numerator = bytes_per_element * intermediate * (batch * sequence + hidden)
    return numerator / denominator


def strict_integer_candidate(lower_bound: float) -> int | None:
    if not math.isfinite(lower_bound):
        return None
    return math.floor(lower_bound) + 1


def nearest_integer_candidate(lower_bound: float) -> int | None:
    if not math.isfinite(lower_bound):
        return None
    return max(1, math.floor(lower_bound + 0.5))


def next_power_of_two_candidate(lower_bound: float) -> int | None:
    if not math.isfinite(lower_bound):
        return None
    strict = strict_integer_candidate(lower_bound)
    assert strict is not None
    return 1 << (strict - 1).bit_length()


def build_report(args: argparse.Namespace) -> dict:
    cache_bytes = round(args.cache_mib * MIB)
    working_set = working_set_bytes(
        args.batch,
        args.sequence,
        args.hidden,
        args.intermediate,
        args.splits,
        args.bytes_per_element,
    )
    lower_bound = equation13_lower_bound(
        args.batch,
        args.sequence,
        args.hidden,
        args.intermediate,
        args.bytes_per_element,
        cache_bytes,
    )
    report = {
        "inputs": {
            "batch": args.batch,
            "sequence": args.sequence,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "expansion": args.intermediate / args.hidden,
            "bytes_per_element": args.bytes_per_element,
            "cache_bytes": cache_bytes,
            "cache_mib": cache_bytes / MIB,
            "evaluated_splits": args.splits,
        },
        "equation_12": {
            **asdict(working_set),
            "total_bytes": working_set.total_bytes,
            "input_mib": working_set.input_bytes / MIB,
            "split_activation_mib": working_set.split_activation_bytes / MIB,
            "split_weight_mib": working_set.split_weight_bytes / MIB,
            "total_mib": working_set.total_bytes / MIB,
            "fits": working_set.total_bytes < cache_bytes,
        },
        "equation_13": {
            "strict_lower_bound": lower_bound if math.isfinite(lower_bound) else None,
            "finite": math.isfinite(lower_bound),
            "strict_integer_candidate": strict_integer_candidate(lower_bound),
            "paper_nearest_integer_candidate": nearest_integer_candidate(lower_bound),
            "author_next_power_of_two_candidate": next_power_of_two_candidate(lower_bound),
        },
    }
    if not math.isfinite(lower_bound):
        report["equation_13"]["reason"] = (
            "No finite K: the unsplit input term is greater than or equal to cache capacity."
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--bytes-per-element", type=int, required=True)
    parser.add_argument("--cache-mib", type=float, required=True)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    eq12 = report["equation_12"]
    eq13 = report["equation_13"]
    print(f"Input term:            {eq12['input_mib']:.3f} MiB")
    print(f"Split activation term: {eq12['split_activation_mib']:.3f} MiB")
    print(f"Split weight term:     {eq12['split_weight_mib']:.3f} MiB")
    print(f"Equation 12 total:     {eq12['total_mib']:.3f} MiB")
    print(f"Fits strict inequality: {eq12['fits']}")
    if eq13["finite"]:
        print(f"Equation 13: K > {eq13['strict_lower_bound']:.6f}")
        print(f"Strict integer candidate: {eq13['strict_integer_candidate']}")
        print(f"Paper nearest-integer candidate: {eq13['paper_nearest_integer_candidate']}")
        print(
            "Author next-power-of-two candidate: "
            f"{eq13['author_next_power_of_two_candidate']}"
        )
    else:
        print(eq13["reason"])


if __name__ == "__main__":
    main()

