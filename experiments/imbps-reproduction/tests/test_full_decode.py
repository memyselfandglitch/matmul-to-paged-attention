from __future__ import annotations

import unittest

from src.full_decode_bench import (
    equation_13_l2_proxy,
    expected_variants,
    nominal_kv_cache_bytes,
    summarize_emissions,
)
from src.summarize_full_decode import summarize


class FullDecodeDesignTests(unittest.TestCase):
    def test_expected_matrix_contains_normal_and_every_pard_variant(self) -> None:
        claim = {"splits": [2, 4, 8], "draft_tokens": [4, 8, 12]}
        variants = expected_variants(claim)
        self.assertEqual(len(variants), 16)
        self.assertIn(
            {"mode": "normal", "draft_tokens": 0, "backend": "tpp", "split": 1},
            variants,
        )
        self.assertIn(
            {"mode": "pard", "draft_tokens": 12, "backend": "imbps", "split": 8},
            variants,
        )

    def test_emission_summary_handles_multi_token_pard_events(self) -> None:
        result = summarize_emissions(
            [
                {"elapsed_ms": 10.0, "token_count": 3},
                {"elapsed_ms": 14.0, "token_count": 2},
            ],
            total_ms=20.0,
        )
        self.assertEqual(result["emitted_tokens"], 5)
        self.assertEqual(result["first_emission_tokens"], 3)
        self.assertEqual(result["post_first_tokens_per_second"], 200.0)

    def test_summary_builds_all_comparison_views(self) -> None:
        claim = {"splits": [2, 4, 8], "draft_tokens": [4, 8, 12]}
        results = []
        for variant in expected_variants(claim):
            total = 100.0
            if variant["mode"] == "pard":
                total = 80.0
            if variant["backend"] == "imbps":
                total += float(variant["split"])
            results.append(
                {
                    "claim": "synthetic",
                    **{
                        "mode": variant["mode"],
                        "draft_tokens": variant["draft_tokens"],
                        "backend": variant["backend"],
                        "split": variant["split"],
                    },
                    "records": [
                        {
                            "context_tokens": 16,
                            "summary": {
                                "median_ttft_ms": 10.0,
                                "median_output_tokens_per_second": 1.0,
                                "median_post_first_tokens_per_second": 1.1,
                            },
                            "rounds": [
                                {
                                    "requested_output_tokens": 4,
                                    "total_ms": total,
                                    "accepted_draft_fraction": (
                                        0.5 if variant["mode"] == "pard" else None
                                    ),
                                    "mean_kept_tokens_per_speculative_iteration": (
                                        2.0 if variant["mode"] == "pard" else None
                                    ),
                                    "accepted_draft_tokens": (
                                        2 if variant["mode"] == "pard" else None
                                    ),
                                    "proposed_draft_tokens": (
                                        4 if variant["mode"] == "pard" else None
                                    ),
                                    "target_forward_iterations": 2,
                                    "generated_tokens_per_target_forward": 2.0,
                                    "signature": {
                                        "generated_token_sha256": "same"
                                    },
                                }
                            ],
                        }
                    ],
                }
            )
        rows = summarize(results)
        self.assertEqual(len(rows), 16)
        pard_tpp = next(
            row
            for row in rows
            if row["mode"] == "pard"
            and row["draft_tokens"] == 4
            and row["backend"] == "tpp"
        )
        self.assertEqual(pard_tpp["pard_speedup_vs_normal_same_backend"], 1.25)
        self.assertEqual(pard_tpp["median_generated_tokens_per_target_forward"], 2.0)
        self.assertEqual(pard_tpp["median_accepted_draft_fraction"], 0.5)

    def test_normal_only_claim_has_no_pard_variants(self) -> None:
        claim = {"splits": [2, 4, 8], "draft_tokens": []}
        variants = expected_variants(claim)
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(variant["mode"] == "normal" for variant in variants))

    def test_equation_13_qwen_batch_range_selects_k2(self) -> None:
        for batch in (1, 2, 4, 8, 16):
            model = equation_13_l2_proxy(batch, 3584, 18944, 96)
            self.assertEqual(model["next_power_of_two_candidate"], 2)
            self.assertLess(model["working_set_mib_at_candidate"], 96)

    def test_nominal_kv_bytes_accounts_for_gqa(self) -> None:
        class Config:
            num_attention_heads = 28
            num_key_value_heads = 4
            hidden_size = 3584
            num_hidden_layers = 28

        expected = 2 * 2 * 128 * 28 * 4 * 128 * 2
        self.assertEqual(nominal_kv_cache_bytes(Config(), 2, 128), expected)


if __name__ == "__main__":
    unittest.main()
