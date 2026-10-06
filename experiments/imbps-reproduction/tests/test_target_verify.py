from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pace_target_verify_bench import (
    grouped_cases,
    nominal_kv_bytes,
    validate_cases,
)
from src.summarize_target_verify import load_results, summarize


class TargetVerificationDesignTests(unittest.TestCase):
    def test_nominal_opt30b_kv_bytes(self) -> None:
        # OPT-30B: 48 layers, hidden size 7168, BF16 K and V.
        value = nominal_kv_bytes(64, 128, 48, 7168)
        self.assertEqual(value, 11_274_289_152)

    def test_cases_group_by_reusable_prefill(self) -> None:
        cases = [
            {
                "case_id": "a",
                "mode": "target_verification",
                "batch": 128,
                "draft_tokens": 8,
                "context_tokens": 128,
            },
            {
                "case_id": "b",
                "mode": "target_verification",
                "batch": 128,
                "draft_tokens": 4,
                "context_tokens": 128,
            },
            {
                "case_id": "c",
                "mode": "normal_decode",
                "batch": 256,
                "draft_tokens": 1,
                "context_tokens": 128,
            },
        ]
        validate_cases(cases)
        groups = grouped_cases(cases)
        self.assertEqual(
            [[case["case_id"] for case in group] for group in groups],
            [["b", "a"], ["c"]],
        )

    def test_normal_decode_rejects_multi_token_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "normal_decode"):
            validate_cases(
                [
                    {
                        "case_id": "bad",
                        "mode": "normal_decode",
                        "batch": 1,
                        "draft_tokens": 2,
                        "context_tokens": 16,
                    }
                ]
            )


class TargetVerificationSummaryTests(unittest.TestCase):
    @staticmethod
    def _result(backend: str, split: int, medians: list[float]) -> dict:
        top5 = [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]]
        if backend == "imbps":
            top5 = [[2, 1, 3, 4, 5], [7, 8, 9, 10, 11]]
        top1 = [1, 7] if backend == "tpp" else [2, 7]
        return {
            "schema_version": 1,
            "status": "complete",
            "scope_note": "test",
            "claim": "synthetic",
            "backend": backend,
            "split": split,
            "expected_variants": [
                {"backend": "tpp", "split": 1},
                {"backend": "imbps", "split": 2},
            ],
            "model": {
                "name": "synthetic",
                "requested_reference": "synthetic",
                "resolved_path": "/synthetic",
                "snapshot_commit": "abc123",
            },
            "runner": {"model_load_seconds": 1.0},
            "environment": {"max_rss_kib": 1},
            "records": [
                {
                    "case": {
                        "case_id": "verify-b2-g4-m8-c16",
                        "mode": "target_verification",
                        "batch": 2,
                        "draft_tokens": 4,
                        "active_rows": 8,
                        "context_tokens": 16,
                    },
                    "rounds": [
                        {"median_ms": value, "durations_ms": [value]}
                        for value in medians
                    ],
                    "run": {
                        "iterations": len(medians),
                        "q1_ms": min(medians),
                        "q3_ms": max(medians),
                        "median_rollback_ms": 0.01,
                    },
                    "numerics": {
                        "top1_token_ids": top1,
                        "top5_token_ids": top5,
                    },
                    "cache": {
                        "nominal_bytes": 1024,
                        "allocated_bytes": 2048,
                        "prefill_ms": 3.0,
                    },
                }
            ],
        }

    def test_summary_reports_speedup_and_top5_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend, split, medians in (
                ("tpp", 1, [10.0] * 5),
                ("imbps", 2, [5.0] * 5),
            ):
                case_dir = root / f"{backend}-k{split}"
                case_dir.mkdir()
                (case_dir / "result.json").write_text(
                    json.dumps(self._result(backend, split, medians)),
                    encoding="utf-8",
                )
            rows = summarize(load_results(root))
            candidate = next(row for row in rows if row["backend"] == "imbps")
            self.assertEqual(candidate["speedup_vs_tpp"], 2.0)
            self.assertTrue(candidate["speedup_supports_faster"])
            self.assertEqual(candidate["top1_agreement"], 0.5)
            self.assertEqual(candidate["baseline_top1_in_candidate_top5"], 1.0)

    def test_summary_rejects_missing_backend_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_dir = root / "tpp-k1"
            case_dir.mkdir()
            (case_dir / "result.json").write_text(
                json.dumps(self._result("tpp", 1, [10.0] * 5)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing=.*imbps"):
                load_results(root)

    def test_summary_rejects_mixed_model_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._result("tpp", 1, [10.0] * 5)
            candidate = self._result("imbps", 2, [5.0] * 5)
            candidate["model"]["snapshot_commit"] = "different"
            for name, result in (("tpp-k1", baseline), ("imbps-k2", candidate)):
                case_dir = root / name
                case_dir.mkdir()
                (case_dir / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "different model artifacts"):
                load_results(root)


if __name__ == "__main__":
    unittest.main()
