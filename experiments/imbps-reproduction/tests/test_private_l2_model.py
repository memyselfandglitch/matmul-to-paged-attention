from __future__ import annotations

import unittest

from src.audit_pace_imbps import inspect_sources
from src.private_l2_model import private_l2_working_set


class PrivateL2ModelTests(unittest.TestCase):
    def test_aggregate_candidate_is_optimistic_k8_at_512_rows(self) -> None:
        result = private_l2_working_set(512, 7168, 28672, 8, 96)
        self.assertTrue(result["models"]["perfect_partition"]["fits_strict"])
        self.assertEqual(
            result["equation_candidates"]["perfect_partition"][
                "next_power_of_two_candidate"
            ],
            8,
        )

    def test_replicated_source_cannot_fit_decode_rows_128_or_larger(self) -> None:
        result = private_l2_working_set(128, 7168, 28672, 8, 96)
        self.assertFalse(result["models"]["source_replicated"]["fits_strict"])
        self.assertFalse(
            result["equation_candidates"]["source_replicated"]["finite"]
        )

    def test_source_audit_rejects_missing_parallel_todo(self) -> None:
        python_text = """
layer_up_proj_weight.chunk(chunks=chunks)
layer.down_proj.weight.chunk(chunks=chunks, dim=1)
"""
        cpp_text = """
for (int i = 0; i < iterations; ++i) {}
prim_vec.at(i).execute(); prim_vec.at(i + 1).execute();
"""
        with self.assertRaises(RuntimeError):
            inspect_sources(python_text, cpp_text)


if __name__ == "__main__":
    unittest.main()
