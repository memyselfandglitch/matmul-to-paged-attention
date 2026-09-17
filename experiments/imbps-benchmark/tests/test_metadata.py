import tempfile
import unittest
from pathlib import Path

from imbps_bench.metadata import collect_metadata


class MetadataTests(unittest.TestCase):
    def test_metadata_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metadata.json"
            metadata = collect_metadata(output, arguments={"test": True})
            self.assertTrue(output.exists())
            self.assertIn("torch", metadata)
            self.assertIn("process", metadata)
            self.assertEqual(metadata["arguments"], {"test": True})


if __name__ == "__main__":
    unittest.main()

