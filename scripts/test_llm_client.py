"""Standalone verification for forge.llm_client.LLMClient.

Run from the project root with the project's virtualenv active:

    python scripts/test_llm_client.py

Set TEST_OLLAMA=1 to additionally exercise the Ollama backend (requires a
running local Ollama daemon with the configured model pulled).
"""

import os
import traceback

from forge.llm_client import LLMClient


def _run_provider_tests(provider: str) -> None:
    client = LLMClient(provider=provider)

    text = client.complete("Say exactly: hello from groq")
    print(f"[{provider}] complete() ->", text)

    obj = client.complete_json(
        "Return a JSON object with one key called status and value ok"
    )
    print(f"[{provider}] complete_json() ->", obj)


def main() -> None:
    try:
        _run_provider_tests("groq")
        print("Groq tests passed")
    except Exception:
        print("Groq tests FAILED:")
        traceback.print_exc()

    if os.environ.get("TEST_OLLAMA") == "1":
        try:
            _run_provider_tests("ollama")
            print("Ollama tests passed")
        except Exception:
            print("Ollama tests FAILED:")
            traceback.print_exc()
    else:
        print("Skipping Ollama test \u2014 set TEST_OLLAMA=1 to enable")


if __name__ == "__main__":
    main()
