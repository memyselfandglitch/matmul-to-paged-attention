from __future__ import annotations

import unittest

from src.summarize_batched_decode import summarize


class BatchedDecodeSummaryTests(unittest.TestCase):
    def test_summary_uses_decode_tail_as_separate_outcome(self) -> None:
        results = []
        variants = [
            {"mode": "normal", "draft_tokens": 0, "backend": "tpp", "split": 1},
            {"mode": "normal", "draft_tokens": 0, "backend": "imbps", "split": 2},
        ]
        for batch in (1, 2):
            for backend, split, total, ttft, tail in (
                ("tpp", 1, 100.0, 40.0, 60.0),
                ("imbps", 2, 90.0, 45.0, 45.0),
            ):
                results.append(
                    {
                        "claim": "synthetic",
                        "batch": batch,
                        "mode": "normal",
                        "backend": backend,
                        "split": split,
                        "expected_variants": variants,
                        "l2_equation_13_proxy": {
                            "strict_lower_bound": 1.4,
                            "next_power_of_two_candidate": 2,
                        },
                        "records": [
                            {
                                "context_tokens": 128,
                                "nominal_kv_cache": {
                                    "prompt_mib": 7.0 * batch,
                                    "final_mib": 8.0 * batch,
                                },
                                "summary": {
                                    "median_output_tokens_per_second": 10.0,
                                    "median_post_first_tokens_per_second": 11.0,
                                },
                                "rounds": [
                                    {
                                        "requested_output_tokens": 4,
                                        "total_ms": total,
                                        "emission": {
                                            "time_to_first_emission_ms": ttft,
                                            "post_first_emission_ms": tail,
                                        },
                                        "signature": {
                                            "generated_token_sha256": "same"
                                        },
                                    }
                                    for _ in range(3)
                                ],
                            }
                        ],
                    }
                )
        rows = summarize(results)
        self.assertEqual(len(rows), 4)
        candidate = next(
            row
            for row in rows
            if row["batch"] == 2 and row["backend"] == "imbps"
        )
        self.assertAlmostEqual(candidate["total_speedup_vs_tpp"], 100 / 90)
        self.assertAlmostEqual(candidate["decode_tail_speedup_vs_tpp"], 60 / 45)
        self.assertAlmostEqual(candidate["ttft_speedup_vs_tpp"], 40 / 45)
        self.assertAlmostEqual(candidate["median_decode_step_ms"], 15.0)
        self.assertAlmostEqual(
            candidate["median_aggregate_decode_ms_per_token"], 7.5
        )


if __name__ == "__main__":
    unittest.main()
