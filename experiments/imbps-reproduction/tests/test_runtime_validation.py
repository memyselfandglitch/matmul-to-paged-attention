from __future__ import annotations

import unittest

from src.validate_runtime import mapped_allocator_libraries, selected_thp_mode


class RuntimeValidationTests(unittest.TestCase):
    def test_selected_thp_mode(self) -> None:
        self.assertEqual(selected_thp_mode("always [madvise] never"), "madvise")
        self.assertEqual(selected_thp_mode("[always] madvise never"), "always")
        self.assertIsNone(selected_thp_mode("always madvise never"))

    def test_allocator_mapping_deduplicates_paths(self) -> None:
        maps = """
7f00-7f01 r-xp 0 00:00 0 /opt/lib/libtcmalloc.so.4
7f01-7f02 r--p 0 00:00 0 /opt/lib/libtcmalloc.so.4
7f02-7f03 r-xp 0 00:00 0 /lib/libc.so.6
"""
        self.assertEqual(
            mapped_allocator_libraries(maps), ["/opt/lib/libtcmalloc.so.4"]
        )


if __name__ == "__main__":
    unittest.main()
