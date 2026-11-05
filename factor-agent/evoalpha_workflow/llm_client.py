"""OpenAI-compatible JSON client for the HKUST GenAI endpoint."""

from __future__ import annotations

import json
import os
from typing import Any


class OpenAIJSONClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        client: Any | None = None,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.temperature = temperature
        self.json_mode = json_mode
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required")
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv(
                    "OPENAI_BASE_URL",
                    "https://hkust.azure-api.net/hkust-genai/v1",
                ),
                default_headers={"api-key": api_key},
                timeout=120.0,
                max_retries=2,
            )
        self.client = client

    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.json_mode:
            request["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(
            **request,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return _parse_json_object(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be one JSON object")
    return payload
