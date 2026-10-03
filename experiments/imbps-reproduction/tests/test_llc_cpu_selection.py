from __future__ import annotations

import unittest

from src.select_llc_cpus import choose_groups, parse_cpu_list


class LlcCpuSelectionTests(unittest.TestCase):
    def test_parse_cpu_list(self) -> None:
        self.assertEqual(parse_cpu_list("0-3,8,10-11"), {0, 1, 2, 3, 8, 10, 11})

    def test_selects_complete_groups(self) -> None:
        groups = [tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 24))]
        self.assertEqual(choose_groups(groups, 16), groups[:2])

    def test_rejects_partial_group(self) -> None:
        groups = [tuple(range(0, 8)), tuple(range(8, 16))]
        with self.assertRaisesRegex(ValueError, "split LLC group"):
            choose_groups(groups, 12)


if __name__ == "__main__":
    unittest.main()
