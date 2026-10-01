from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.summarize_performance import summarize_table_vi


class PerformanceSummaryTests(unittest.TestCase):
    def test_table_vi_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for batch in (128, 256, 512, 1024):
                for backend, split, framework, timing in (
                    ("vllm", 1, "vllm", 10.0),
                    ("imbps", 2, "pace", 9.0),
                ):
                    result_dir = (
                        root
                        / f"llama3.1-8b-b{batch}-{backend}-k{split}"
                        / "results"
                    )
                    result_dir.mkdir(parents=True)
                    payload = {
                        "benchmark_results": [
                            {"framework": framework, "metrics": {"average_ttft": timing}}
                        ]
                    }
                    (result_dir / "fake_results.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            rows = summarize_table_vi(root)
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(abs(row["measured_improvement_percent"] - 10.0) < 1e-12 for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
