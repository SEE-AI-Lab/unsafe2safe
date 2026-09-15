import csv
import json
import tempfile
import unittest
from pathlib import Path

from vlm_captioning.collect_captions import collect_captions, merge_with_metadata


class CollectCaptionsTest(unittest.TestCase):
    def test_collects_nested_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "captions" / "class_a" / "image_001_caption.json"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({"caption": "A bicycle."}), encoding="utf-8")

            rows = collect_captions(root / "captions")

            self.assertEqual(rows, [{"file": "class_a/image_001.jpg", "caption": "A bicycle."}])

    def test_merges_only_matching_metadata_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "split"])
                writer.writeheader()
                writer.writerows([
                    {"file": "class_a/image_001.jpg", "split": "train"},
                    {"file": "class_b/image_002.jpg", "split": "val"},
                ])

            output = root / "merged.csv"
            merge_with_metadata([{"file": "class_a/image_001.jpg", "caption": "A bicycle."}], metadata, output)

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"file": "class_a/image_001.jpg", "split": "train", "caption": "A bicycle."}])


if __name__ == "__main__":
    unittest.main()
