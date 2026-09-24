"""Statistics for the native PACE K sweep.

The ``c + a/K + bK`` curve used here is deliberately an empirical surrogate.
It is not derived from the cache-spill model.  Held-out K values are required
to determine whether it has predictive value beyond the points used to fit it.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class SurrogateFit:
    c: float
    a: float
    b: float

    def predict(self, split_k: int) -> float:
        return self.c + self.a / split_k + self.b * split_k

    @property
    def continuous_optimum(self) -> float | None:
        if self.a <= 0 or self.b <= 0:
            return None
        return math.sqrt(self.a / self.b)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_ci(
    values: list[float],
    *,
    samples: int = 20_000,
    seed: int = 20260924,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    rng = random.Random(seed)
    bootstrapped = [
        statistics.median(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("surrogate design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def fit_empirical_surrogate(points: dict[int, float]) -> SurrogateFit:
    """Least-squares fit of T(K)=c+a/K+bK to at least three K values."""
    if len(points) < 3:
        raise ValueError("at least three distinct K values are required")
    if any(split_k <= 0 for split_k in points):
        raise ValueError("K values must be positive")
    rows = [[1.0, 1.0 / split_k, float(split_k)] for split_k in points]
    values = [points[split_k] for split_k in points]
    normal_matrix = [
        [sum(row[i] * row[j] for row in rows) for j in range(3)]
        for i in range(3)
    ]
    normal_vector = [
        sum(row[i] * value for row, value in zip(rows, values))
        for i in range(3)
    ]
    c, a, b = _solve_3x3(normal_matrix, normal_vector)
    return SurrogateFit(c=c, a=a, b=b)
