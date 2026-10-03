from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.run_uprof_matrix import experiment_cases, normalized_record
from src.summarize_uprof_matrix import compare_k4_k8, mechanism_checks, summarize_pass


class UprofMatrixTests(unittest.TestCase):
    def test_normalizes_cache_counters_per_invocation(self) -> None:
        case = {
            "model": "opt30b",
            "batch": 16,
            "sequence": 1920,
            "hidden": 7168,
            "intermediate": 28672,
            "backend": "imbps",
            "split": 4,
            "cache_mib": 384,
            "equation12_working_set_mib": 938,
            "equation12_fits": False,
            "equation13_strict_lower_bound": 22.5,
            "equation13_strict_integer_candidate": 23,
            "equation13_author_power_of_two_candidate": 32,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "benchmark.json").write_text(
                json.dumps(
                    {"run": {"median_ms": 2.0, "measurement_wall_seconds": 6.0}}
                )
            )
            (path / "manifest.json").write_text(
                json.dumps(
                    {
                        "measurement": {
                            "iterations": 3,
                            "counter_active_seconds": 6.1,
                        }
                    }
                )
            )
            (path / "uprof.csv").write_text(
                """CORE METRICS
Metric,System (Aggregated)
IPC (Sys + User),1.05
L2 Access (pti),145.61
L2 Miss (pti),16.65
L3 METRICS
Metric,System (Aggregated)
L3 Access,300
L3 Miss,90
L3 Miss %,30
L3 Hit %,70
Ave L3 Miss Latency (ns),124
"""
            )
            row = normalized_record(path, case, "cache", 1, 1)
        self.assertEqual(row["l3_access_per_invocation"], 100)
        self.assertEqual(row["l3_miss_per_invocation"], 30)
        self.assertEqual(row["l3_hit_per_invocation"], 70)

    @staticmethod
    def rows(pass_name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for round_number in (1, 2):
            for backend, split, time_ms, misses, hit_rate, dram, ai in (
                ("tpp", 1, 12.0, 120.0, 50.0, 240.0, 10.0),
                ("imbps", 4, 10.0, 80.0, 70.0, 160.0, 15.0),
                ("imbps", 8, 9.0, 60.0, 80.0, 120.0, 20.0),
                ("imbps", 16, 11.0, 70.0, 75.0, 140.0, 17.0),
            ):
                row: dict[str, object] = {
                    "pass": pass_name,
                    "round": round_number,
                    "model": "opt30b",
                    "batch": 16,
                    "sequence": 1920,
                    "backend": backend,
                    "split": split,
                    "median_ms": time_ms,
                    "cache_mib": 384,
                    "equation12_working_set_mib": 100,
                    "equation12_fits": backend == "imbps",
                    "equation13_strict_lower_bound": 6.2,
                    "equation13_strict_integer_candidate": 7,
                    "equation13_author_power_of_two_candidate": 8,
                }
                if pass_name == "cache":
                    row.update(
                        {
                            "ipc": 1.0,
                            "l2_access_pti": 1.0,
                            "l2_miss_pti": 1.0,
                            "l3_access_per_invocation": misses * 2,
                            "l3_miss_per_invocation": misses,
                            "l3_hit_per_invocation": misses,
                            "l3_miss_percent": 100 - hit_rate,
                            "l3_hit_percent": hit_rate,
                            "l3_miss_latency_ns": 100.0,
                        }
                    )
                else:
                    row.update(
                        {
                            "total_mem_bw_gbps": 1.0,
                            "read_mem_bw_gbps": 1.0,
                            "write_mem_bw_gbps": 1.0,
                            "dram_bytes_per_invocation": dram,
                            "dram_read_bytes_per_invocation": dram * 0.9,
                            "dram_write_bytes_per_invocation": dram * 0.1,
                            "measured_dram_ai_flops_per_byte": ai,
                        }
                    )
                rows.append(row)
        return rows

    def test_mechanism_check_identifies_common_best_k(self) -> None:
        cache = summarize_pass(self.rows("cache"), "cache")
        traffic = summarize_pass(self.rows("traffic"), "traffic")
        checks = mechanism_checks(cache, traffic)
        self.assertEqual(checks[0]["fastest_k_cache_pass"], 8)
        self.assertEqual(checks[0]["minimum_l3_miss_k"], 8)
        self.assertEqual(checks[0]["minimum_dram_byte_k"], 8)
        self.assertEqual(checks[0]["maximum_measured_dram_ai_k"], 8)
        self.assertTrue(checks[0]["cache_pass_rank_agreement"])
        self.assertTrue(checks[0]["traffic_pass_rank_agreement"])

    def test_cache_fit_claim_has_only_resident_imbps_cases(self) -> None:
        cases = experiment_cases("cache_fit_opt30b")
        self.assertEqual(len(cases), 24)
        self.assertTrue(
            all(case["equation12_fits"] for case in cases if case["backend"] == "imbps")
        )

    def test_server_equation_sweep_covers_power_of_two_candidates(self) -> None:
        cases = experiment_cases("server_equation_opt30b")
        imbps = [case for case in cases if case["backend"] == "imbps"]
        self.assertEqual(len(cases), 27)
        self.assertTrue(all(case["equation12_fits"] for case in imbps))
        candidates = {
            case["equation13_author_power_of_two_candidate"] for case in imbps
        }
        self.assertEqual(candidates, {2, 4, 8, 16, 32, 64})
        for sequence in {case["sequence"] for case in imbps}:
            shape = [case for case in imbps if case["sequence"] == sequence]
            self.assertEqual(
                min(case["split"] for case in shape),
                shape[0]["equation13_author_power_of_two_candidate"],
            )
        self.assertTrue(
            all(
                not case["equation12_fits"]
                for case in cases
                if case["backend"] == "tpp"
            )
        )

    def test_k4_k8_comparison_is_paired_by_round(self) -> None:
        comparisons = compare_k4_k8(self.rows("cache"), "cache")
        timing = next(row for row in comparisons if row["metric"] == "median_ms")
        self.assertEqual(timing["median_difference"], 1.0)
        self.assertTrue(timing["ci95_excludes_zero"])


if __name__ == "__main__":
    unittest.main()
