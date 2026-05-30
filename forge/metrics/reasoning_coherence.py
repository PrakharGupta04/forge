"""Reasoning-coherence metric.

Measures how semantically consistent the agent's chain of thought is by
encoding each ``llm_call`` step's output with a small sentence-transformer
model (``BAAI/bge-small-en-v1.5``) and averaging the cosine similarity
between consecutive embeddings.

The embedding model is loaded **exactly once** at instantiation. It is
deliberately not loaded at import time (so importing the module is cheap
and side-effect-free) and not on every ``score`` call (so the metric is
fast enough to run inside the engine on a per-trajectory basis).
"""

from __future__ import annotations

import numpy as np

from forge.metrics.base import BaseMetric


class ReasoningCoherenceMetric(BaseMetric):
    """Mean pairwise cosine similarity over consecutive LLM step embeddings."""

    _MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"ReasoningCoherenceMetric could not load embedding model "
                f"{self._MODEL_NAME!r}: {exc}. A coherence metric without "
                f"its embedding model is non-functional."
            ) from exc

    @property
    def name(self) -> str:
        return "reasoning_coherence"

    def score(self, trajectory: dict) -> float:
        texts: list[str] = []
        for step in trajectory.get("steps", []):
            if step.get("type") != "llm_call":
                continue
            out = step.get("output")
            if out is None or out == "":
                continue
            texts.append(out if isinstance(out, str) else str(out))

        # < 2 non-empty LLM outputs -> trivially coherent by design, not a
        # default fallback for missing data.
        if len(texts) < 2:
            return 1.0

        embeddings = self._model.encode(texts, convert_to_numpy=True)

        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                similarities.append(0.0)
                continue
            similarities.append(float(np.dot(a, b) / (na * nb)))

        mean_similarity = float(np.mean(similarities))
        return max(0.0, min(1.0, mean_similarity))
