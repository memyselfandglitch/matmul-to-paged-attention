from __future__ import annotations

import unittest

from src.run_standalone_matrix import bootstrap_median_ci, build_cases, summarize


def record(
    backend: str,
    split: int,
    round_index: int,
    milliseconds: float,
    sequence: int = 1920,
) -> dict:
    return {
        "case": {
            "model": "opt30b",
            "batch": 16,
            "sequence": sequence,
            "activation": "relu",
            "backend": backend,
            "split": split,
        },
        "run": {"median_ms": milliseconds},
        "matrix": {"round": round_index, "order": 1},
    }


class MatrixAnalysisTests(unittest.TestCase):
    def test_cache_fit_cases_record_equation_12_fit(self) -> None:
        registry = {
            "models": {
                "opt30b": {
                    "hidden": 7168,
                    "intermediate": 28672,
                    "activation": "relu",
                }
            }
        }
        claim = {
            "dtype": "bf16",
            "cache_mib": 384,
            "models": ["opt30b"],
            "cases": [{"model": "opt30b", "batch": 2, "sequence": 1920}],
            "splits": [2, 4],
            "fit_requirement": "imbps",
        }
        cases = build_cases(registry, claim)
        self.assertEqual(len(cases), 3)
        baseline = next(case for case in cases if case["backend"] == "tpp")
        candidates = [case for case in cases if case["backend"] == "imbps"]
        self.assertFalse(baseline["cache_model"]["fits_strict"])
        self.assertTrue(all(case["cache_model"]["fits_strict"] for case in candidates))

        claim["fit_requirement"] = "all"
        with self.assertRaisesRegex(ValueError, "non-fitting required cases"):
            build_cases(registry, claim)

    def test_summary_stratifies_equal_batches_by_sequence(self) -> None:
        records = [
            record("tpp", 1, 1, 12.0, sequence=128),
            record("imbps", 2, 1, 10.0, sequence=128),
            record("tpp", 1, 1, 24.0, sequence=256),
            record("imbps", 2, 1, 16.0, sequence=256),
        ]
        rows = summarize(records, "cache_fit_opt30b", {"targets": {}})
        self.assertEqual(len(rows), 4)
        speedups = {
            row["sequence"]: row["speedup"]
            for row in rows
            if row["backend"] == "imbps"
        }
        self.assertEqual(speedups, {128: 1.2, 256: 1.5})

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
            "targets": {"16": {"speedup": 1.2, "split": 4}},
        }
        rows = summarize([baseline, candidate], "table_ii", claim)
        imbps = next(row for row in rows if row["backend"] == "imbps")
        self.assertEqual(imbps["split"], 4)
        self.assertAlmostEqual(imbps["speedup"], 1.2)


if __name__ == "__main__":
    unittest.main()
