from __future__ import annotations

import unittest

from src.select_llc_cpus import (
    choose_groups,
    groups_from_labels,
    parse_cpu_list,
    select_grouping,
)


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

    def test_groups_cpus_by_stable_topology_label(self) -> None:
        labels = {
            0: ("0", "7"),
            1: ("0", "7"),
            8: ("0", "9"),
            9: ("0", "9"),
        }
        self.assertEqual(groups_from_labels(labels), [(0, 1), (8, 9)])

    def test_exact_shape_skips_socket_wide_shared_cache_view(self) -> None:
        candidates = {
            "l3_cache_id": [tuple(range(16))],
            "l3_shared_cpu_list": [tuple(range(16))],
            "die_id": [tuple(range(0, 8)), tuple(range(8, 16))],
        }
        groups, source = select_grouping(
            candidates, expected_groups=2, expected_group_size=8
        )
        self.assertEqual(source, "die_id")
        self.assertEqual(groups, candidates["die_id"])

    def test_exact_shape_failure_reports_all_candidates(self) -> None:
        candidates = {
            "l3_cache_id": [tuple(range(16))],
            "cluster_id": [(cpu,) for cpu in range(16)],
        }
        with self.assertRaisesRegex(RuntimeError, "l3_cache_id=1 groups"):
            select_grouping(candidates, expected_groups=2, expected_group_size=8)


if __name__ == "__main__":
    unittest.main()
