import json
import unittest
from types import SimpleNamespace

from evoalpha_workflow.llm_client import OpenAIJSONClient, _parse_json_object


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )


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

    def test_sends_provider_reasoning_and_output_limits(self):
        completions = FakeCompletions()
        client = OpenAIJSONClient(
            model="deepseek-v4-pro",
            client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            reasoning_effort="high",
            max_tokens=16_384,
        )

        result = client.complete("Return JSON.", json.dumps({"task": "test"}))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(completions.request["reasoning_effort"], "high")
        self.assertEqual(completions.request["max_tokens"], 16_384)
        self.assertEqual(completions.request["response_format"], {"type": "json_object"})
        self.assertEqual(client.request_count, 1)


if __name__ == "__main__":
    unittest.main()
