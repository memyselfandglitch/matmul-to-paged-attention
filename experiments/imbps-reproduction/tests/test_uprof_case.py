from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from src.run_uprof_case import build_profiler_command, build_worker_command


class UprofCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = argparse.Namespace(
            uprof_bin=Path("/opt/AMDuProfPcm"),
            access_mode="perf",
            metrics="ipc,l2,l3",
            package=0,
            cpu_list="0-95",
            backend="imbps",
            split=4,
            batch=16,
            sequence=1920,
            hidden=7168,
            intermediate=28672,
            dtype="bf16",
            activation="relu",
            threads=96,
            warmups=3,
            iterations=3,
            setup_timeout_seconds=1800,
        )

    def test_profiler_is_package_scoped_and_signal_gated(self) -> None:
        worker = ["python", "worker.py"]
        command = build_profiler_command(self.args, worker, Path("out.csv"))
        self.assertIn("package=0", command)
        self.assertIn("--wait-for-signal", command)
        self.assertNotIn("-a", command)
        self.assertEqual(command[-2:], worker)

    def test_worker_is_pinned_and_has_readiness_gate(self) -> None:
        command = build_worker_command(
            self.args, Path("ready"), Path("start"), Path("result.json")
        )
        self.assertIn("--physcpubind=0-95", command)
        self.assertIn("--ready-file", command)
        self.assertIn("--start-file", command)
        self.assertIn("--warmups", command)


if __name__ == "__main__":
    unittest.main()
