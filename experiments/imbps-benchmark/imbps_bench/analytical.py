"""The cache-capacity model stated in the IMBPS paper.

This module intentionally implements the published first-order equation without
silently adding topology or scratchpad corrections. It is a hypothesis to test,
not the benchmark's definition of an optimal split.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PaperPrediction:
    cache_bytes: int
    usable_cache_bytes: int
    input_bytes: int
    denominator_bytes: int
    continuous_k: Optional[float]
    ceiling_k: Optional[int]
    feasible: bool


def paper_working_set_bytes(
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    element_size: int,
    split_k: int,
) -> float:
    """Equation 12 with M=tokens and I=fH."""
    if min(tokens, hidden_size, intermediate_size, element_size, split_k) <= 0:
        raise ValueError("all dimensions, element_size, and split_k must be positive")
    return element_size * (
        tokens * hidden_size
        + (tokens * intermediate_size + hidden_size * intermediate_size) / split_k
    )


def paper_split_prediction(
    cache_bytes: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    element_size: int,
    cache_fraction: float = 1.0,
) -> PaperPrediction:
    """Equation 13, exposing infeasible negative-denominator cases."""
    if min(cache_bytes, tokens, hidden_size, intermediate_size, element_size) <= 0:
        raise ValueError("cache and tensor parameters must be positive")
    if not 0.0 < cache_fraction <= 1.0:
        raise ValueError("cache_fraction must be in (0, 1]")
    usable_cache_bytes = int(cache_bytes * cache_fraction)
    input_bytes = element_size * tokens * hidden_size
    denominator_bytes = usable_cache_bytes - input_bytes
    if denominator_bytes <= 0:
        return PaperPrediction(
            cache_bytes=cache_bytes,
            usable_cache_bytes=usable_cache_bytes,
            input_bytes=input_bytes,
            denominator_bytes=denominator_bytes,
            continuous_k=None,
            ceiling_k=None,
            feasible=False,
        )
    numerator = element_size * intermediate_size * (tokens + hidden_size)
    continuous_k = numerator / denominator_bytes
    return PaperPrediction(
        cache_bytes=cache_bytes,
        usable_cache_bytes=usable_cache_bytes,
        input_bytes=input_bytes,
        denominator_bytes=denominator_bytes,
        continuous_k=continuous_k,
        ceiling_k=max(1, int(math.ceil(continuous_k))),
        feasible=True,
    )

