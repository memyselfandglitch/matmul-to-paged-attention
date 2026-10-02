from __future__ import annotations

import math
import unittest

from src.cache_model import (
    MIB,
    equation13_lower_bound,
    next_power_of_two_candidate,
    strict_integer_candidate,
    working_set_bytes,
)
from src.system_info import parse_cpu_list
from src.validate_topology import evaluate, parse_lscpu_csv


class CacheModelTests(unittest.TestCase):
    def test_table_ii_k4_working_set(self) -> None:
        result = working_set_bytes(16, 1920, 7168, 28672, 4, 2)
        self.assertAlmostEqual(result.input_bytes / MIB, 420.0)
        self.assertAlmostEqual(result.split_activation_bytes / MIB, 420.0)
        self.assertAlmostEqual(result.split_weight_bytes / MIB, 98.0)
        self.assertAlmostEqual(result.total_bytes / MIB, 938.0)

    def test_512_mib_requires_k23(self) -> None:
        bound = equation13_lower_bound(16, 1920, 7168, 28672, 2, 512 * MIB)
        self.assertGreater(bound, 22)
        self.assertLess(bound, 23)
        self.assertEqual(strict_integer_candidate(bound), 23)

    def test_750_mib_next_power_is_8(self) -> None:
        bound = equation13_lower_bound(16, 1920, 7168, 28672, 2, 750 * MIB)
        self.assertGreater(bound, 6)
        self.assertLess(bound, 7)
        self.assertEqual(next_power_of_two_candidate(bound), 8)

    def test_384_mib_has_no_finite_solution(self) -> None:
        bound = equation13_lower_bound(16, 1920, 7168, 28672, 2, 384 * MIB)
        self.assertTrue(math.isinf(bound))

    def test_opt30b_cache_fit_shapes_fit_from_k2(self) -> None:
        for batch, sequence in ((1, 1920), (2, 1920), (16, 128), (16, 256)):
            result = working_set_bytes(batch, sequence, 7168, 28672, 2, 2)
            self.assertLess(result.total_bytes, 384 * MIB)

    def test_cache_resident_controls_fit_unsplit(self) -> None:
        shapes = (
            (16, 128, 4096, 16384),
            (16, 256, 4096, 16384),
            (16, 384, 4096, 16384),
            (16, 128, 5120, 20480),
            (16, 192, 5120, 20480),
            (16, 224, 5120, 20480),
        )
        for batch, sequence, hidden, intermediate in shapes:
            result = working_set_bytes(
                batch, sequence, hidden, intermediate, 1, 2
            )
            self.assertLess(result.total_bytes, 384 * MIB)

    def test_parse_cpu_list(self) -> None:
        self.assertEqual(parse_cpu_list("0-3,8,10-11"), {0, 1, 2, 3, 8, 10, 11})

    def test_topology_validation(self) -> None:
        rows = parse_lscpu_csv("# comment\n0,0,0,0\n1,1,0,1\n2,0,0,0\n")
        passing = evaluate(rows, {0, 1}, 2, True, True)
        self.assertEqual(passing["status"], "pass")
        failing = evaluate(rows, {0, 1, 2}, 2, True, True)
        self.assertEqual(failing["status"], "fail")
        self.assertTrue(failing["smt_in_affinity"])


if __name__ == "__main__":
    unittest.main()
