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
            profiler_start_timeout_seconds=30,
        )

    def test_profiler_uses_only_pid_compatible_scope_options(self) -> None:
        command = build_profiler_command(self.args, 1234, Path("out.csv"))
        self.assertIn("-p", command)
        self.assertIn("1234", command)
        self.assertNotIn("--wait-for-signal", command)
        self.assertNotIn("-c", command)
        self.assertNotIn("-A", command)
        self.assertNotIn("-a", command)

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
