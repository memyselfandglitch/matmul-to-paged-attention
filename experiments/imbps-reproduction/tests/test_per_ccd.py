from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from src.per_ccd_mlp_bench import (
    assign_blocks,
    chunk_ranges,
    constrain_auxiliary_threads,
    linear_quantile_ranks,
    main,
    per_ccd_working_set,
)
from src.summarize_per_ccd import summarize


class PerCCDDesignTests(unittest.TestCase):
    def test_linear_quantile_ranks_match_torch_interpolation_definition(self) -> None:
        self.assertEqual(linear_quantile_ranks(4, 0.5), (1, 2, 0.5))
        self.assertEqual(linear_quantile_ranks(101, 0.99), (99, 99, 0.0))

    def test_linear_quantile_ranks_support_output_larger_than_2_to_24(self) -> None:
        count = 4096 * 7168
        lower, upper, weight = linear_quantile_ranks(count, 0.99)
        self.assertEqual((lower, upper), (29_066_525, 29_066_526))
        self.assertAlmostEqual(weight, 0.73, places=6)

    def test_auxiliary_thread_outside_ccd_is_constrained(self) -> None:
        task_paths = [type("Task", (), {"name": "101"})()]
        with (
            patch("src.per_ccd_mlp_bench.Path") as path_type,
            patch(
                "src.per_ccd_mlp_bench.os.sched_getaffinity",
                side_effect=[{0}, {8, 9}, {8, 9}],
                create=True,
            ),
            patch(
                "src.per_ccd_mlp_bench.os.sched_setaffinity", create=True
            ) as set_affinity,
            patch(
                "src.per_ccd_mlp_bench.collect_thread_affinity",
                return_value=({101: [8, 9]}, [8, 9]),
            ),
        ):
            path_type.return_value.iterdir.return_value = task_paths
            union = constrain_auxiliary_threads((8, 9))
        set_affinity.assert_called_once_with(101, {8, 9})
        self.assertEqual(union, [8, 9])

    def test_topology_is_captured_before_torch_initializes_openmp(self) -> None:
        source = inspect.getsource(main)
        self.assertLess(
            source.index("launch_affinity = set(os.sched_getaffinity(0))"),
            source.index("    import torch"),
        )

    def test_chunk_ranges_match_expected_fringe(self) -> None:
        ranges = chunk_ranges(28_672, 12)
        self.assertEqual(len(ranges), 12)
        self.assertEqual(ranges[0], (0, 2390))
        self.assertEqual(ranges[-1], (26_290, 28_672))
        self.assertEqual(sum(end - start for start, end in ranges), 28_672)

    def test_balanced_block_assignment(self) -> None:
        self.assertEqual(
            assign_blocks(24, 12),
            [list(range(index * 2, index * 2 + 2)) for index in range(12)],
        )

    def test_k24_fits_one_ccd_at_512_rows(self) -> None:
        model = per_ccd_working_set(512, 7168, 28672, 24)
        self.assertLess(model["modeled_working_set_mib"], 32)

    def test_k12_does_not_fit_one_ccd_at_512_rows(self) -> None:
        model = per_ccd_working_set(512, 7168, 28672, 12)
        self.assertGreater(model["modeled_working_set_mib"], 32)

    def test_summary_compares_parallel_with_global_same_k(self) -> None:
        variants = [
            {"variant_id": "tpp-all96", "mode": "tpp", "ccd_count": 12, "total_splits": 1},
            {"variant_id": "global-k12-all96", "mode": "global_imbps", "ccd_count": 12, "total_splits": 12},
            {"variant_id": "parallel-p1-k1", "mode": "per_ccd", "ccd_count": 1, "total_splits": 1},
            {"variant_id": "parallel-p12-k12", "mode": "per_ccd", "ccd_count": 12, "total_splits": 12},
        ]
        durations = {
            "tpp-all96": 10.0,
            "global-k12-all96": 8.0,
            "parallel-p1-k1": 60.0,
            "parallel-p12-k12": 6.0,
        }
        results = []
        for variant in variants:
            results.append(
                {
                    "claim": "synthetic",
                    "variant": variant,
                    "topology": {"cores_per_ccd": 8},
                    "records": [
                        {
                            "active_rows": 128,
                            "durations_ms": [durations[variant["variant_id"]]],
                            "q1_ms": durations[variant["variant_id"]],
                            "q3_ms": durations[variant["variant_id"]],
                            "cache_model_per_ccd": {
                                "modeled_working_set_mib": 20.0,
                                "fits_one_ccd_strict": True,
                            },
                            "numerics_vs_global_same_k": (
                                {
                                    "max_abs": 0.01,
                                    "q99_abs": 0.0,
                                    "output_coordinate_argmax_agreement": 1.0,
                                }
                                if variant["mode"] == "per_ccd"
                                else None
                            ),
                        }
                    ],
                }
            )
        rows = summarize(results)
        parallel = next(
            row for row in rows if row["variant_id"] == "parallel-p12-k12"
        )
        self.assertAlmostEqual(parallel["speedup_vs_global_imbps_same_k"], 8 / 6)


if __name__ == "__main__":
    unittest.main()
