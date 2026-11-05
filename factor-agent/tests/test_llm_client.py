import unittest

from evoalpha_workflow.llm_client import _parse_json_object


class LLMClientTests(unittest.TestCase):
    def test_parses_plain_json(self):
        self.assertEqual(_parse_json_object('{"ok": true}'), {"ok": True})

    def test_parses_markdown_fenced_json(self):
        self.assertEqual(
            _parse_json_object('```json\n{"ok": true}\n```'),
            {"ok": True},
        )

    def test_extracts_json_from_surrounding_text(self):
        self.assertEqual(
            _parse_json_object('Result follows: {"ok": true} done.'),
            {"ok": True},
        )


if __name__ == "__main__":
    unittest.main()
