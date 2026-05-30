"""Multi-turn consistency metric.

Checks whether the agent contradicts its own earlier statements across a
multi-turn run. Each later turn is compared against the **initial**
statements via a single LLM call; the first contradiction short-circuits
the whole score to ``0.0``.

Source selection (per the design spec): assistant-turn count from
``trajectory["conversation_history"]`` is compared against the number of
``llm_call`` step outputs, and whichever source has more entries is used.
A single ``LLMClient`` instance is constructed once per ``score()`` call
and reused across every contradiction check.

Failure semantics:

* Both sources have <2 entries (after assistant-only filtering) -> 1.0
  (single-turn agents cannot be inconsistent).
* Any successful evaluation reports ``contradiction_found = true`` -> 0.0
  (short-circuit; further checks are skipped).
* All evaluations succeed and none flag a contradiction -> 1.0.
* Some checks raise, others succeed: failed checks are logged at WARNING
  and skipped; the final score is decided by the successful ones.
* **Every** check raises -> 0.5 (we can't claim consistency we never
  measured; this is intentionally not 1.0).
"""

from __future__ import annotations

import logging
from typing import Optional

from forge.llm_client import LLMClient
from forge.metrics.base import BaseMetric


logger = logging.getLogger(__name__)


_TURN_CHAR_BUDGET = 500


_CONTRADICTION_CHECK_INSTRUCTIONS = (
    "You are checking for logical contradictions between an AI agent's "
    "statements across a conversation. Given the agent's initial "
    "statements and a later statement, determine if the later statement "
    "directly contradicts any claim in the initial statements. A "
    "contradiction is when the agent asserts X in one turn and explicitly "
    "asserts not-X or a mutually exclusive claim in a later turn. Vague "
    "inconsistencies or topic changes are not contradictions. Return a "
    'JSON object with key "contradiction_found" as a boolean and key '
    '"explanation" as a single sentence.'
)


class MultiTurnConsistencyMetric(BaseMetric):
    """0.0 if any later turn contradicts the first; 1.0 if none do; 0.5 if unknown."""

    METRIC_NAME = "multi_turn_consistency"

    def __init__(self, llm_provider: Optional[str] = None) -> None:
        super().__init__()
        self._provider = llm_provider

    @property
    def name(self) -> str:
        return self.METRIC_NAME

    def _get_llm(self) -> LLMClient:
        if self._provider is not None:
            return LLMClient(provider=self._provider)
        return LLMClient()

    def score(self, trajectory: dict) -> float:
        turns = self._select_turn_source(trajectory)
        if len(turns) < 2:
            return 1.0

        first_statements = self._truncate(turns[0])

        client = self._get_llm()

        successful_evals = 0
        for later_turn in turns[1:]:
            later_text = self._truncate(later_turn)
            prompt = (
                f"{_CONTRADICTION_CHECK_INSTRUCTIONS}\n"
                f"Initial statements: {first_statements}\n"
                f"Later statement: {later_text}"
            )
            try:
                result = client.complete_json(prompt)
            except Exception as exc:
                logger.warning(
                    "MultiTurnConsistencyMetric: contradiction check failed "
                    "for one turn: %s; treating as no-contradiction-found",
                    exc,
                )
                continue

            successful_evals += 1
            if self._coerce_contradiction(result):
                return 0.0

        if successful_evals == 0:
            return 0.5
        return 1.0

    # ------------------------------------------------------------------ helpers

    def _select_turn_source(self, trajectory: dict) -> list[str]:
        """Pick the richer of (assistant turns) vs (llm_call outputs).

        Returns the list of turn contents as strings. Assistant turns from
        ``conversation_history`` are preferred when they have at least as
        many entries as the ``llm_call`` step outputs (ties go to the
        conversation history, which is the more semantically faithful
        source when present).
        """
        conv = trajectory.get("conversation_history") or []
        assistant_contents: list[str] = []
        if isinstance(conv, list):
            for entry in conv:
                if not isinstance(entry, dict):
                    continue
                if entry.get("role") != "assistant":
                    continue
                content = entry.get("content", "")
                if content is None:
                    content = ""
                assistant_contents.append(
                    content if isinstance(content, str) else str(content)
                )

        llm_outputs: list[str] = []
        for step in trajectory.get("steps", []):
            if step.get("type") != "llm_call":
                continue
            out = step.get("output", "")
            if out is None:
                out = ""
            llm_outputs.append(out if isinstance(out, str) else str(out))

        if len(assistant_contents) >= len(llm_outputs):
            return assistant_contents
        return llm_outputs

    @staticmethod
    def _truncate(text) -> str:
        s = text if isinstance(text, str) else str(text)
        return s[:_TURN_CHAR_BUDGET]

    @staticmethod
    def _coerce_contradiction(result) -> bool:
        """Only an explicit boolean true (or string ``"true"/"yes"/"1"``) counts."""
        if not isinstance(result, dict):
            return False
        value = result.get("contradiction_found")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return False
