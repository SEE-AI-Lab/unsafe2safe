import json
import tempfile
import unittest
from pathlib import Path

from pipeline.stage1.collect_captions import collect_captions


class CollectCaptionsTest(unittest.TestCase):
    def test_custom_output_column_and_structured_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caption_path = root / "class_a" / "000_caption.json"
            caption_path.parent.mkdir()
            caption_path.write_text(
                json.dumps(
                    {
                        "caption": "SECTION: PRIVACY_FLAG\nFALSE\n"
                        "SECTION: PRIVACY_PRESERVING_DESCRIPTION\nA bicycle."
                    }
                ),
                encoding="utf-8",
            )

            rows = collect_captions(root, parse_structured=True, output_column="edit_instruction")

        self.assertEqual(rows[0]["file"], "class_a/000.jpg")
        self.assertEqual(rows[0]["edit_instruction"], "SECTION: PRIVACY_FLAG\nFALSE\nSECTION: PRIVACY_PRESERVING_DESCRIPTION\nA bicycle.")
        self.assertFalse(rows[0]["PRIVACY_FLAG"])
        self.assertEqual(rows[0]["PUBLIC_CAPTION"], "A bicycle.")


if __name__ == "__main__":
    unittest.main()
