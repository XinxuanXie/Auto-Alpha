"""JSON client for OpenAI-compatible model providers."""

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
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        default_base_url: str = "https://hkust.azure-api.net/hkust-genai/v1",
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        send_api_key_header: bool = True,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.temperature = temperature
        self.json_mode = json_mode
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.request_count = 0
        if client is None:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is required")
            from openai import OpenAI

            client_arguments = {
                "api_key": api_key,
                "base_url": os.getenv(base_url_env, default_base_url),
                "timeout": 180.0,
                "max_retries": 2,
            }
            if send_api_key_header:
                client_arguments["default_headers"] = {"api-key": api_key}
            client = OpenAI(
                **client_arguments,
            )
        self.client = client

    @classmethod
    def deepseek(
        cls,
        model: str,
        *,
        reasoning_effort: str,
        max_tokens: int = 16_384,
    ) -> "OpenAIJSONClient":
        return cls(
            model=model,
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url="https://api.deepseek.com",
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            send_api_key_header=False,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.request_count += 1
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
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
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
