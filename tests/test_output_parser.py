import unittest

from vlm_captioning.output_parser import (
    extract_privacy_flag,
    parse_structured_output,
    split_sections,
)


class OutputParserTest(unittest.TestCase):
    def test_parses_released_prompt_headings_and_flag(self):
        response = """PRIVACY_FLAG: TRUE

SECTION: PRIVACY_REVIEW
- ITEM: A face

SECTION: FULL_VISUAL_DESCRIPTION
A person stands beside a bicycle.

SECTION: PRIVACY_PRESERVING_DESCRIPTION
A bicycle is beside an anonymous figure.
"""

        parsed = parse_structured_output(response)

        self.assertIs(parsed["PRIVACY_FLAG"], True)
        self.assertEqual(parsed["PRIVATE_CAPTION"], "A person stands beside a bicycle.")
        self.assertEqual(parsed["PUBLIC_CAPTION"], "A bicycle is beside an anonymous figure.")

    def test_parses_markdown_headings_from_paper_prompt(self):
        response = """### SECTION: PRIVACY_REVIEW
The review.
### SECTION: PRIVATE_CAPTION
The private scene.
### SECTION: PUBLIC_CAPTION
The public scene.
"""

        sections = split_sections(response)

        self.assertEqual(sections["PRIVACY_REVIEW"], "The review.")
        self.assertEqual(sections["PRIVATE_CAPTION"], "The private scene.")
        self.assertEqual(sections["PUBLIC_CAPTION"], "The public scene.")

    def test_missing_flag_returns_none(self):
        self.assertIsNone(extract_privacy_flag("No explicit decision."))


if __name__ == "__main__":
    unittest.main()
