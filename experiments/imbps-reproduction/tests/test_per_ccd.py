from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.per_ccd_mlp_bench import (
    assign_blocks,
    balanced_ranges,
    chunk_ranges,
    constrain_auxiliary_threads,
    linear_quantile_ranks,
    main,
    per_ccd_working_set,
)
from src.summarize_per_ccd import load_results, summarize


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

    def test_balanced_ranges_return_one_range_per_worker(self) -> None:
        ranges = balanced_ranges(128, 12)
        self.assertEqual(len(ranges), 12)
        self.assertEqual(ranges[0], (0, 11))
        self.assertEqual(ranges[-1], (118, 128))
        self.assertEqual(sum(end - start for start, end in ranges), 128)
        self.assertLessEqual(
            max(end - start for start, end in ranges)
            - min(end - start for start, end in ranges),
            1,
        )

    def test_shape_isolated_fragments_are_merged(self) -> None:
        variants = [
            {
                "variant_id": "tpp-all96",
                "mode": "tpp",
                "ccd_count": 12,
                "total_splits": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rows in (128, 256):
                case = root / f"tpp-all96-m{rows}"
                case.mkdir()
                (case / "result.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "claim": "isolated",
                            "variant": variants[0],
                            "expected_variants": variants,
                            "records": [{"active_rows": rows}],
                        }
                    ),
                    encoding="utf-8",
                )
            results = load_results(root)
        self.assertEqual(len(results), 1)
        self.assertEqual(
            [record["active_rows"] for record in results[0]["records"]],
            [128, 256],
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

    def test_summary_supports_pthread_parallel_mode(self) -> None:
        variants = [
            {"variant_id": "tpp-all96", "mode": "tpp", "ccd_count": 12, "total_splits": 1},
            {"variant_id": "global-k12-all96", "mode": "global_imbps", "ccd_count": 12, "total_splits": 12},
            {"variant_id": "thread-p1-k1", "mode": "per_ccd_threads", "ccd_count": 1, "total_splits": 1},
            {"variant_id": "thread-p12-k12", "mode": "per_ccd_threads", "ccd_count": 12, "total_splits": 12},
        ]
        durations = {
            "tpp-all96": 10.0,
            "global-k12-all96": 8.0,
            "thread-p1-k1": 60.0,
            "thread-p12-k12": 6.0,
        }
        results = []
        for variant in variants:
            results.append(
                {
                    "claim": "synthetic-threaded",
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
                                if variant["mode"] == "per_ccd_threads"
                                else None
                            ),
                            "median_compute_and_sync_ms": (
                                5.0 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_reduction_and_conversion_ms": (
                                1.0 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_worker_operator_min_ms": (
                                4.0 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_worker_operator_max_ms": (
                                5.0 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_worker_operator_imbalance": (
                                1.25 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_worker_publication_max_ms": (
                                0.2 if variant["mode"] == "per_ccd_threads" else None
                            ),
                            "median_worker_reduction_max_ms": (
                                0.0 if variant["mode"] == "per_ccd_threads" else None
                            ),
                        }
                    ],
                }
            )
        rows = summarize(results)
        threaded = next(
            row for row in rows if row["variant_id"] == "thread-p12-k12"
        )
        self.assertAlmostEqual(threaded["speedup_vs_global_imbps_same_k"], 8 / 6)
        self.assertAlmostEqual(threaded["parallel_speedup_vs_one_ccd"], 10.0)
        self.assertEqual(threaded["median_reduction_and_conversion_ms"], 1.0)
        self.assertEqual(threaded["median_worker_operator_imbalance"], 1.25)
        self.assertAlmostEqual(threaded["reduction_fraction_of_latency"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
