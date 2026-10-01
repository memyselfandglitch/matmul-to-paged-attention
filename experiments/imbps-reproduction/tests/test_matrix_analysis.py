from __future__ import annotations

import unittest

from src.run_standalone_matrix import bootstrap_median_ci, summarize


def record(backend: str, split: int, round_index: int, milliseconds: float) -> dict:
    return {
        "case": {
            "model": "opt30b",
            "batch": 16,
            "activation": "relu",
            "backend": backend,
            "split": split,
        },
        "run": {"median_ms": milliseconds},
        "matrix": {"round": round_index, "order": 1},
    }


class MatrixAnalysisTests(unittest.TestCase):
    def test_paired_speedup(self) -> None:
        records = [
            record("tpp", 1, 1, 12.0),
            record("imbps", 4, 1, 10.0),
            record("tpp", 1, 2, 24.0),
            record("imbps", 4, 2, 20.0),
            record("tpp", 1, 3, 18.0),
            record("imbps", 4, 3, 15.0),
        ]
        claim = {
            "sequence": 1920,
            "targets": {"16": {"speedup": 1.2, "split": 4}},
        }
        rows = summarize(records, "table_ii", claim)
        candidate = next(row for row in rows if row["backend"] == "imbps")
        self.assertAlmostEqual(candidate["paired_speedup_median"], 1.2)
        self.assertTrue(candidate["paper_timing_target_in_ci95"])
        self.assertEqual(candidate["empirical_best_imbps_split"], 4)
        self.assertTrue(candidate["paper_best_split_reproduced"])

    def test_single_value_bootstrap_is_degenerate(self) -> None:
        self.assertEqual(bootstrap_median_ci([1.25]), (1.25, 1.25))

    def test_worker_splits_field_is_normalized(self) -> None:
        baseline = record("tpp", 1, 1, 12.0)
        candidate = record("imbps", 4, 1, 10.0)
        candidate["case"]["splits"] = candidate["case"].pop("split")
        claim = {
            "sequence": 1920,
            "targets": {"16": {"speedup": 1.2, "split": 4}},
        }
        rows = summarize([baseline, candidate], "table_ii", claim)
        imbps = next(row for row in rows if row["backend"] == "imbps")
        self.assertEqual(imbps["split"], 4)
        self.assertAlmostEqual(imbps["speedup"], 1.2)


if __name__ == "__main__":
    unittest.main()
