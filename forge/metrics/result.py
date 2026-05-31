"""Structured result type for Forge metrics.

Going forward, every metric returns a :class:`MetricResult` from its new
``score_with_explanation`` method. The legacy ``score()`` method (a bare
``float``) and ``safe_score()`` (a ``(float, Optional[str])`` tuple)
remain in place for backward compatibility — see :mod:`forge.metrics.base`
— but the richer ``MetricResult`` shape is what the API and dashboards
will consume.

A ``MetricResult`` carries everything a UI or LLM-as-judge consumer needs
to interpret the score: the numeric score, a one-sentence human-readable
explanation, a metric-specific ``metadata`` bag for intermediate values
(LCS lengths, claim counts, etc.), and an explicit error channel so a
metric failure is never indistinguishable from a low-but-valid score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricResult:
    """A single metric's evaluation result.

    Attributes:
        score: Final clamped score in ``[0.0, 1.0]``.
        explanation: One-sentence, human-readable description of what the
            score means for the specific trajectory that was scored.
        metadata: Metric-specific intermediate values that contributed to
            the score (e.g. ``lcs_length``, ``golden_length``,
            ``name_score``, ``input_score`` for ``ToolCallFidelityMetric``).
            Use only JSON-serializable values.
        metric_name: The producing metric's stable identifier (matches
            ``BaseMetric.name``). Defaults to ``""`` so the dataclass can
            be constructed positionally with just the score+explanation
            during legacy fallbacks.
        had_error: ``True`` iff the score is a fallback emitted because
            the metric raised an exception. Always check this before
            treating a 0.0 as a meaningful low score.
        error_message: Stringified exception when ``had_error`` is true,
            otherwise ``None``.
    """

    score: float
    explanation: str
    metadata: dict = field(default_factory=dict)
    metric_name: str = ""
    had_error: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of every field."""
        return {
            "score": self.score,
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
            "metric_name": self.metric_name,
            "had_error": self.had_error,
            "error_message": self.error_message,
        }

    @classmethod
    def error_result(
        cls,
        metric_name: str,
        error_message: str,
        fallback_score: float = 0.0,
    ) -> "MetricResult":
        """Construct a ``MetricResult`` representing a metric failure.

        Used by :meth:`BaseMetric.safe_score_with_explanation` to capture
        any exception raised by ``score_with_explanation`` while still
        producing a structured result the engine can aggregate over.
        """
        return cls(
            score=fallback_score,
            explanation=f"Metric failed: {error_message}",
            metadata={},
            metric_name=metric_name,
            had_error=True,
            error_message=error_message,
        )
