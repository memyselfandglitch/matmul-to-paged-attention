from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from src.run_uprof_case import (
    build_gate_command,
    build_profiler_command,
    build_worker_command,
)


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
            min_measurement_seconds=5.0,
            setup_timeout_seconds=1800,
            profiler_start_timeout_seconds=30,
            measurement_timeout_seconds=7200,
        )

    def test_profiler_wraps_gate_with_package_scoped_counters(self) -> None:
        gate = ["python", "gate.py"]
        command = build_profiler_command(self.args, gate, Path("out.csv"))
        self.assertIn("-c", command)
        self.assertIn("package=0", command)
        self.assertIn("--", command)
        self.assertEqual(command[-2:], gate)
        self.assertNotIn("-p", command)
        self.assertNotIn("-A", command)
        self.assertNotIn("-a", command)

    def test_gate_releases_worker_and_waits_for_output(self) -> None:
        command = build_gate_command(
            Path("start"), Path("done"), Path("timing.json"), 60
        )
        self.assertIn("--start-file", command)
        self.assertIn("--done-file", command)
        self.assertIn("--timing-file", command)
        self.assertIn("60", command)

    def test_worker_is_pinned_and_has_readiness_gate(self) -> None:
        command = build_worker_command(
            self.args,
            Path("ready"),
            Path("start"),
            Path("done"),
            Path("result.json"),
        )
        self.assertIn("--physcpubind=0-95", command)
        self.assertIn("--ready-file", command)
        self.assertIn("--start-file", command)
        self.assertIn("--done-file", command)
        self.assertIn("--warmups", command)


if __name__ == "__main__":
    unittest.main()
