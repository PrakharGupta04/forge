"""LLM-judged hallucination metric (two-stage: extract claims, check grounding).

What this metric measures
-------------------------
The fraction of factual claims in the agent's final answer that are
supported by the available grounding context (tool outputs collected
from the trajectory plus any ``trajectory["retrieved_context"]`` chunks
emitted by a RAG pipeline).

Methodology
-----------
**Claim extraction (stage 1).** A single LLM-as-judge call with a
structured prompt (see ``_CLAIM_EXTRACTION_INSTRUCTIONS``) decomposes
the final answer into a JSON list of concise factual claims (≤20 words
each). Opinions, hedges, and meta-statements are excluded by design.

**Grounding check (stage 2).** For every extracted claim, an
LLM-as-judge call (see ``_GROUNDING_CHECK_INSTRUCTIONS``) decides
whether the assembled context contains evidence that supports the
claim. Each claim contributes one boolean; the score is
``supported / total``. A single :class:`LLMClient` instance is
constructed once per ``score()`` call and reused across both stages and
every per-claim check.

Known limitations
-----------------
* **The judge can itself hallucinate.** Both stages rely on an LLM, so
  errors compound: a stage-1 over-extraction can invent claims that
  were never made, and a stage-2 false positive can mark an
  unsupported claim as grounded.
* **Closed-world grounding only.** The grounding check considers
  *only* the provided context (tool outputs + ``retrieved_context``);
  it does not consult world knowledge. A factually-correct claim that
  happens to be absent from the context will be scored as
  ungrounded.
* **Cost scales with claim count.** Each claim costs one LLM call.
  Verbose answers with many claims can be expensive at scale.
* **Empty answer returns 1.0.** An agent that produced no answer
  cannot hallucinate, but this is *not* the same as having done its
  job — see the failure semantics below.

Failure semantics
-----------------
* **Empty final answer ->** ``1.0`` (vacuous: nothing to hallucinate).
  This can be misleading if the agent was *supposed* to produce an
  answer; combine with :class:`TaskCompletionMetric` to detect that
  case.
* **No grounding context available ->** ``0.5`` (neutral) with a
  WARNING log. Emitted when there are no tool outputs and no
  ``retrieved_context`` — we genuinely cannot decide either way.
* **No claims extracted ->** ``1.0`` (no claims, no hallucinations).
* **Whole-stage-1 LLM failure ->** ``0.5`` with a WARNING log.
* **Per-claim stage-2 failure ->** the affected claim is treated as
  ungrounded (``False``) and processing continues.
"""

from __future__ import annotations

import logging
from typing import Optional

from forge.llm_client import LLMClient
from forge.metrics.base import BaseMetric


logger = logging.getLogger(__name__)


_CONTEXT_CHAR_BUDGET = 3000


_CLAIM_EXTRACTION_INSTRUCTIONS = (
    "You are extracting factual claims from an AI agent's answer for "
    "hallucination analysis. Given the following answer, extract all "
    "specific factual claims made. A factual claim is a statement that "
    "asserts something that could be verified as true or false. Do not "
    "include opinions, qualifications, or meta-statements. Return a JSON "
    'object with a single key "claims" whose value is a list of strings. '
    "If the answer contains no verifiable factual claims, return an empty "
    "list. Keep each claim concise, under 20 words."
)


_GROUNDING_CHECK_INSTRUCTIONS = (
    "You are checking whether a factual claim is supported by provided "
    "context. Given the claim and context below, determine if the context "
    "contains information that supports or confirms this claim. Return a "
    'JSON object with key "grounded" as a boolean true if the context '
    "supports the claim, false if it contradicts the claim or the context "
    "does not contain relevant information."
)


class HallucinationMetric(BaseMetric):
    """Score = fraction of claims in the final answer that are grounded in context."""

    METRIC_NAME = "hallucination_score"

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
        # ---- Resolve final answer (fallback to last llm_call output) --------
        final_answer = trajectory.get("final_answer", "") or ""
        if not final_answer:
            for step in reversed(trajectory.get("steps", [])):
                if step.get("type") == "llm_call":
                    candidate = step.get("output", "") or ""
                    if candidate:
                        final_answer = candidate
                        break
        if not final_answer:
            return 1.0

        # ---- Assemble grounding context -------------------------------------
        context_string = self._build_context_string(trajectory)
        if not context_string:
            logger.warning(
                "HallucinationMetric: grounding context is unavailable "
                "(no tool outputs and no retrieved_context); returning neutral 0.5"
            )
            return 0.5

        # One client for the whole evaluation, per the design contract.
        client = self._get_llm()

        # ---- Stage 1: claim extraction --------------------------------------
        stage1_prompt = f"{_CLAIM_EXTRACTION_INSTRUCTIONS}\nAnswer: {final_answer}"
        try:
            stage1_result = client.complete_json(stage1_prompt)
        except Exception as exc:
            logger.warning(
                "HallucinationMetric stage 1 (claim extraction) failed: %s; "
                "returning neutral 0.5",
                exc,
            )
            return 0.5

        raw_claims = stage1_result.get("claims", []) if isinstance(stage1_result, dict) else []
        if not isinstance(raw_claims, list):
            raw_claims = []
        claims = [str(c) for c in raw_claims if c is not None and str(c).strip()]
        if not claims:
            return 1.0

        # ---- Stage 2: per-claim grounding check -----------------------------
        grounded_results: list[bool] = []
        for claim in claims:
            stage2_prompt = (
                f"{_GROUNDING_CHECK_INSTRUCTIONS}\n"
                f"Claim: {claim}\n"
                f"Context: {context_string}"
            )
            try:
                stage2_result = client.complete_json(stage2_prompt)
            except Exception as exc:
                logger.warning(
                    "HallucinationMetric stage 2 failed for claim %r: %s; "
                    "treating as ungrounded",
                    claim,
                    exc,
                )
                grounded_results.append(False)
                continue

            grounded_results.append(self._coerce_grounded(stage2_result))

        if not grounded_results:
            return 0.5

        raw_score = sum(1 for g in grounded_results if g) / len(grounded_results)
        return max(0.0, min(1.0, raw_score))

    # ------------------------------------------------------------------ helpers

    def _build_context_string(self, trajectory: dict) -> str:
        """Collect every available grounding source and join them, truncated.

        Tool outputs may be plain strings, dicts that wrap content under a
        ``result``/``text`` key, or arbitrary serializable objects. We try
        to preserve as much real grounding information as possible rather
        than dropping anything that isn't a bare string.
        """
        sources: list[str] = []

        for step in trajectory.get("steps", []):
            if step.get("type") != "tool_call":
                continue
            raw_out = step.get("tool_output")
            if raw_out is None or raw_out == "":
                raw_out = step.get("output")
            extracted = self._coerce_tool_output(raw_out)
            if extracted:
                sources.append(extracted)

        retrieved = trajectory.get("retrieved_context") or []
        if isinstance(retrieved, list):
            for chunk in retrieved:
                if chunk is None or chunk == "":
                    continue
                if isinstance(chunk, str):
                    sources.append(chunk)
                else:
                    sources.append(str(chunk))

        joined = "\n\n".join(sources)
        return joined[:_CONTEXT_CHAR_BUDGET]

    @staticmethod
    def _coerce_tool_output(raw) -> str:
        """Pull a usable text snippet from a tool's output, defensively."""
        if raw is None or raw == "":
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            for key in ("result", "text", "output", "content"):
                if key in raw and raw[key] not in (None, ""):
                    v = raw[key]
                    return v if isinstance(v, str) else str(v)
            return str(raw)
        return str(raw)

    @staticmethod
    def _coerce_grounded(stage2_result) -> bool:
        """Treat only an explicit boolean ``true`` (or string ``"true"``) as grounded."""
        if not isinstance(stage2_result, dict):
            return False
        value = stage2_result.get("grounded")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return False
