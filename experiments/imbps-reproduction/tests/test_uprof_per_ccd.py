from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.summarize_uprof_per_ccd import summarize


class UprofPerCCDSummaryTests(unittest.TestCase):
    def test_summarizes_ccd_l3_counters_per_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "thread-p2-k2-m128"
            case.mkdir()
            (case / "manifest.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            (case / "benchmark.json").write_text(
                json.dumps(
                    {
                        "claim": "synthetic",
                        "variant": {
                            "variant_id": "thread-p2-k2",
                            "mode": "per_ccd_threads",
                            "ccd_count": 2,
                            "total_splits": 2,
                        },
                        "records": [
                            {
                                "active_rows": 128,
                                "durations_ms": [4.0, 5.0],
                                "median_ms": 4.5,
                                "median_compute_and_sync_ms": 4.0,
                                "median_reduction_and_conversion_ms": 0.5,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (case / "uprof.csv").write_text(
                "CORE METRICS\n"
                "Metric,CCD 0,CCD 1\n"
                "IPC (Sys + User),1.0,2.0\n"
                "Utilization (%),90,50\n"
                "L2 Access (pti),100,120\n"
                "L2 Miss (pti),20,30\n"
                "L3 METRICS\n"
                "Metric,CCD 0,CCD 1\n"
                "L3 Access,100,300\n"
                "L3 Miss,20,60\n",
                encoding="utf-8",
            )
            long_rows, aggregate = summarize(root)
        self.assertEqual(len(long_rows), 2)
        self.assertEqual(aggregate[0]["l3_access_per_invocation"], 200)
        self.assertEqual(aggregate[0]["l3_miss_per_invocation"], 40)
        self.assertEqual(aggregate[0]["l3_hit_percent"], 80)
        self.assertEqual(aggregate[0]["ccd0_utilization_percent"], 90)
        self.assertEqual(aggregate[0]["other_ccd_median_utilization_percent"], 50)
        self.assertEqual(aggregate[0]["ipc_median_across_ccds"], 1.5)


if __name__ == "__main__":
    unittest.main()
