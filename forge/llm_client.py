"""Thin provider-agnostic LLM client used across Forge.

Supports Groq (cloud) and Ollama (local) as backends, with a uniform
``complete`` (text) and ``complete_json`` (parsed JSON) interface.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


_JSON_INSTRUCTION = (
    "Respond ONLY with a valid JSON object. No markdown formatting, "
    "no code fences, no explanation before or after the JSON."
)

_MAX_JSON_ATTEMPTS = 3


def _strip_code_fences(text: str) -> str:
    """Remove any line that starts with triple backticks (```), preserving the rest."""
    cleaned_lines = [
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    ]
    return "\n".join(cleaned_lines).strip()


class LLMClient:
    """Uniform wrapper over Groq and Ollama chat completions."""

    def __init__(self, provider: str | None = None) -> None:
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "groq")
        self.provider = provider

        if provider == "groq":
            from groq import Groq

            self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
            self.model = "llama-3.3-70b-versatile"
        elif provider == "ollama":
            from ollama import Client

            self.client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
            self.model = "phi3:mini"
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                "Supported providers are 'groq' and 'ollama'."
            )

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Return the model's text completion for ``prompt``."""
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response["message"]["content"]
        except Exception as exc:
            logger.error(
                "LLMClient.complete failed (provider=%s, model=%s): %s",
                self.provider,
                self.model,
                exc,
            )
            raise

    def complete_json(self, prompt: str) -> dict[str, Any]:
        """Return a parsed JSON object from the model, retrying on parse failure."""
        full_prompt = f"{prompt}\n\n{_JSON_INSTRUCTION}"
        last_raw: str = ""
        last_error: Exception | None = None

        for attempt in range(1, _MAX_JSON_ATTEMPTS + 1):
            last_raw = self.complete(full_prompt)
            cleaned = _strip_code_fences(last_raw)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "LLMClient.complete_json attempt %d/%d failed to parse JSON: %s",
                    attempt,
                    _MAX_JSON_ATTEMPTS,
                    exc,
                )
                continue

            if not isinstance(parsed, dict):
                last_error = ValueError(
                    f"Expected JSON object (dict), got {type(parsed).__name__}"
                )
                logger.warning(
                    "LLMClient.complete_json attempt %d/%d returned non-object JSON",
                    attempt,
                    _MAX_JSON_ATTEMPTS,
                )
                continue

            return parsed

        raise ValueError(
            "LLM failed to return valid JSON after 3 attempts. "
            f"Last error: {last_error}. Raw response: {last_raw!r}"
        )
