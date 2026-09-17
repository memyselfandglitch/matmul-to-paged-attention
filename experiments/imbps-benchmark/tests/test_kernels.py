import tempfile
import unittest
from pathlib import Path

import torch

from imbps_bench.analytical import paper_split_prediction, paper_working_set_bytes
from imbps_bench.kernels import (
    imbps_forward_into,
    make_input,
    make_reference_workspace,
    make_split_workspace,
    make_weights,
    pack_weights,
    reference_forward_into,
    split_ranges,
)
from imbps_bench.reporting import summarize_rows, write_summary


class SplitRangeTests(unittest.TestCase):
    def test_ranges_cover_dimension_without_overlap(self):
        ranges = split_ranges(17, 5)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 17)
        self.assertEqual(sum(end - start for start, end in ranges), 17)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))

    def test_rejects_more_splits_than_columns(self):
        with self.assertRaises(ValueError):
            split_ranges(4, 5)


class AnalyticalModelTests(unittest.TestCase):
    def test_prediction_satisfies_paper_capacity_bound(self):
        prediction = paper_split_prediction(
            cache_bytes=32 * 1024 * 1024,
            tokens=256,
            hidden_size=4096,
            intermediate_size=16384,
            element_size=2,
        )
        self.assertTrue(prediction.feasible)
        self.assertIsNotNone(prediction.ceiling_k)
        working_set = paper_working_set_bytes(256, 4096, 16384, 2, prediction.ceiling_k)
        self.assertLessEqual(working_set, 32 * 1024 * 1024)

    def test_prediction_exposes_input_larger_than_cache(self):
        prediction = paper_split_prediction(
            cache_bytes=32 * 1024 * 1024,
            tokens=8192,
            hidden_size=4096,
            intermediate_size=16384,
            element_size=2,
        )
        self.assertFalse(prediction.feasible)
        self.assertIsNone(prediction.ceiling_k)


class KernelCorrectnessTests(unittest.TestCase):
    def _check(self, kind: str, split_k: int, layout: str):
        dtype = torch.float32
        tokens, hidden, intermediate = 7, 11, 29
        weights = make_weights(kind, hidden, intermediate, dtype, seed=123)
        x = make_input(tokens, hidden, dtype, seed=456)
        reference_workspace = make_reference_workspace(
            kind, tokens, hidden, intermediate, dtype
        )
        packed = pack_weights(weights, split_k, layout)
        split_workspace = make_split_workspace(
            kind, tokens, hidden, packed.max_width, dtype
        )
        with torch.inference_mode():
            expected = reference_forward_into(x, weights, reference_workspace).clone()
            actual = imbps_forward_into(x, packed, split_workspace).clone()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=2e-4, atol=2e-5),
            msg="kind=%s K=%d layout=%s max_error=%g"
            % (kind, split_k, layout, (actual - expected).abs().max().item()),
        )

    def test_opt_for_even_and_uneven_splits(self):
        for split_k in (1, 2, 3, 7):
            for layout in ("views", "prepacked"):
                with self.subTest(split_k=split_k, layout=layout):
                    self._check("opt", split_k, layout)

    def test_swiglu_for_even_and_uneven_splits(self):
        for split_k in (1, 2, 3, 7):
            for layout in ("views", "prepacked"):
                with self.subTest(split_k=split_k, layout=layout):
                    self._check("swiglu", split_k, layout)

    def test_prepacked_blocks_are_contiguous(self):
        weights = make_weights("swiglu", 8, 19, torch.float32, seed=1)
        packed = pack_weights(weights, 4, "prepacked")
        for block in packed.blocks:
            self.assertTrue(block.up.is_contiguous())
            self.assertTrue(block.down.is_contiguous())
            self.assertIsNotNone(block.gate)
            self.assertTrue(block.gate.is_contiguous())


class ReportingTests(unittest.TestCase):
    def test_summary_computes_paired_speedup(self):
        common = {
            "run_id": "run",
            "mlp_kind": "opt",
            "dtype": "fp32",
            "threads": 1,
            "tokens": 8,
            "hidden_size": 16,
            "intermediate_size": 64,
            "split_k": 2,
            "packing_ms": 0.0,
            "workspace_bytes": 1,
            "packed_weight_bytes": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "allclose": True,
        }
        rows = []
        for variant, layout, latencies in (
            ("reference", "canonical", (10.0, 12.0)),
            ("imbps", "prepacked", (5.0, 6.0)),
        ):
            for latency in latencies:
                row = dict(common)
                row.update(
                    {
                        "variant": variant,
                        "weight_layout": layout,
                        "latency_ms": latency,
                        "gflops": 100.0,
                    }
                )
                rows.append(row)
        summaries = summarize_rows(rows)
        imbps = next(row for row in summaries if row["variant"] == "imbps")
        self.assertAlmostEqual(imbps["speedup_vs_reference"], 2.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.csv"
            write_summary(output, summaries)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
