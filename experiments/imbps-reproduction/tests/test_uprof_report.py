from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.uprof_report import (
    parse_uprof_component_report,
    parse_uprof_report,
    require_metric,
)


class UprofReportTests(unittest.TestCase):
    def test_parses_metric_sections(self) -> None:
        report_text = """\
header
CORE METRICS
Metric,System (Aggregated)
IPC (Sys + User),1.05

L3 METRICS
Metric,System (Aggregated)
L3 Access,35395735338.00
L3 Miss,10148567619.00

DF METRICS
Metric,System (Aggregated)
Total Mem Bw (GB/s),75.55
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            path.write_text(report_text, encoding="utf-8")
            report = parse_uprof_report(path)
        self.assertEqual(require_metric(report, "core", "IPC (Sys + User)"), 1.05)
        self.assertEqual(require_metric(report, "l3", "L3 Miss"), 10148567619)
        self.assertEqual(require_metric(report, "df", "Total Mem Bw (GB/s)"), 75.55)

    def test_missing_metric_fails_visibly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing uProf metric"):
            require_metric({}, "l3", "L3 Miss")

    def test_parses_ccd_component_columns(self) -> None:
        report_text = """\
L3 METRICS
Metric,CCD 0,CCD 1
L3 Access,100,200
L3 Miss,25,100
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            path.write_text(report_text, encoding="utf-8")
            report = parse_uprof_component_report(path)
        self.assertEqual(report["l3"]["L3 Access"]["CCD 1"], 200)
        self.assertEqual(report["l3"]["L3 Miss"]["CCD 0"], 25)


if __name__ == "__main__":
    unittest.main()
