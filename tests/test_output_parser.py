import unittest

from unsafe2safe.stage1.output_parser import parse_structured_output


class StructuredOutputParserTest(unittest.TestCase):
    def test_parses_flag_and_caption_aliases(self):
        parsed = parse_structured_output(
            """
            <think>internal reasoning</think>
            SECTION: PRIVACY_FLAG
            TRUE
            SECTION: FULL_VISUAL_DESCRIPTION
            A person stands beside a bicycle.
            SECTION: PRIVACY_PRESERVING_DESCRIPTION
            A bicycle beside a person.
            """
        )

        self.assertTrue(parsed["PRIVACY_FLAG"])
        self.assertEqual(parsed["PRIVATE_CAPTION"], "A person stands beside a bicycle.")
        self.assertEqual(parsed["PUBLIC_CAPTION"], "A bicycle beside a person.")

    def test_missing_sections_are_omitted(self):
        self.assertEqual(parse_structured_output("PRIVACY_FLAG: FALSE"), {"PRIVACY_FLAG": False})


if __name__ == "__main__":
    unittest.main()
