"""Reasoning-coherence metric — semantic topical consistency, not logic.

What this metric actually measures
----------------------------------
The mean cosine similarity between sentence-transformer embeddings
(``BAAI/bge-small-en-v1.5``) of consecutive ``llm_call`` step outputs in
the trajectory. High mean similarity is a *heuristic* indicator that the
agent stayed on topic across its reasoning chain; low mean similarity
indicates topic drift.

What this metric does NOT measure
---------------------------------
This is a heuristic measure of topical consistency, **not** a
ground-truth measure of logical coherence or reasoning quality. It
cannot detect:

* logical errors (valid premises, invalid inference);
* factual mistakes (two equally confident but contradictory statements
  on the same topic will both embed similarly and score *highly*);
* reasoning fallacies, circular arguments, or self-contradictions
  expressed in similar vocabulary.

For a more complete picture of reasoning quality combine this metric
with :class:`~forge.metrics.task_completion.TaskCompletionMetric`
(does the final answer actually solve the task?) and
:class:`~forge.metrics.hallucination.HallucinationMetric` (are the
claims grounded in the available evidence?).

Implementation notes
--------------------
The embedding model is loaded **at most once per process per model name**
via a class-level :attr:`_model_cache`. The first instantiation pays the
~2–4 s load; subsequent instantiations find the model in the cache
instantly. This matters for benchmark runs where ``MetricEngine``
creates a fresh ``ReasoningCoherenceMetric`` per task — without the
cache, a 50-task run pays 100–200 s of pure model-loading overhead plus
the memory churn of repeatedly loading and garbage-collecting the BGE
model.

The model is deliberately not loaded at import time (so importing the
module is cheap and side-effect-free) and not on every ``score`` call
(so the metric is fast enough to run inside the engine on a
per-trajectory basis). Tests that mock ``SentenceTransformer`` and want
a clean slate can call ``ReasoningCoherenceMetric._model_cache.clear()``
in setup.
"""

from __future__ import annotations

import numpy as np

from forge.metrics.base import BaseMetric
from forge.metrics.result import MetricResult


class ReasoningCoherenceMetric(BaseMetric):
    """Mean pairwise cosine similarity over consecutive LLM step embeddings."""

    METRIC_NAME = "reasoning_coherence"
    _MODEL_NAME = "BAAI/bge-small-en-v1.5"
    # Class-level cache so the SentenceTransformer model is loaded at most
    # once per process per model name. Keyed by model name to leave room
    # for future variants. Tests that mock SentenceTransformer can call
    # ``ReasoningCoherenceMetric._model_cache.clear()`` between runs.
    _model_cache: dict = {}

    def __init__(self) -> None:
        super().__init__()
        model_name = "BAAI/bge-small-en-v1.5"
        if model_name in ReasoningCoherenceMetric._model_cache:
            self._model = ReasoningCoherenceMetric._model_cache[model_name]
            return
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"ReasoningCoherenceMetric could not load embedding model "
                f"{model_name!r}: {exc}. A coherence metric without "
                f"its embedding model is non-functional."
            ) from exc
        ReasoningCoherenceMetric._model_cache[model_name] = model
        self._model = ReasoningCoherenceMetric._model_cache[model_name]
        _ = self._model.encode(["warmup"], convert_to_numpy=True)

    @property
    def name(self) -> str:
        return self.METRIC_NAME

    def score(self, trajectory: dict) -> float:
        steps = trajectory.get("steps", [])
        llm_steps = [s for s in steps if s.get("type") == "llm_call"]
        texts: list[str] = []
        for s in llm_steps:
            raw = s.get("output", "")
            if raw is None:
                text = ""
            elif isinstance(raw, dict):
                text = str(
                    raw.get("text") or raw.get("content") or raw.get("output") or ""
                )
            elif isinstance(raw, str):
                text = raw
            else:
                text = str(raw)
            text = text.strip()
            if len(text) >= 3:
                texts.append(text)

        if len(texts) < 2:
            return 1.0

        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        sims: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            sims.append(sim)

        if not sims:
            return 1.0

        return max(0.0, min(1.0, float(np.mean(sims))))

    def score_with_explanation(self, trajectory: dict) -> MetricResult:
        """Wrap :meth:`score` with a topical-consistency explanation.

        Counts the ``llm_call`` steps and emits a one-sentence
        explanation that is explicit about *topical* similarity (not
        logical coherence — see the class docstring). Metadata records
        the step count so consumers can flag the trivial single-step
        case where the score is 1.0 by definition rather than by
        evidence.
        """
        final_score = self.score(trajectory)
        steps = trajectory.get("steps", []) if isinstance(trajectory, dict) else []
        llm_step_count = sum(1 for s in steps if s.get("type") == "llm_call")

        if llm_step_count < 2:
            explanation = (
                f"Trivially coherent: only {llm_step_count} llm_call step(s); "
                "score defaults to 1.0 (fewer than 2 outputs to compare)"
            )
        else:
            explanation = (
                f"Mean cosine similarity {final_score:.3f} across "
                f"{llm_step_count} consecutive reasoning steps "
                "(topical consistency, not logical coherence)"
            )
        return MetricResult(
            score=final_score,
            explanation=explanation,
            metadata={"llm_step_count": llm_step_count},
            metric_name=self.name,
        )
