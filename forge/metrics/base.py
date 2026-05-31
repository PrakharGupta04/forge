"""Abstract base class for Forge metrics.

Concrete metrics implement :meth:`BaseMetric.score`. The engine is the only
component allowed to catch exceptions; metric implementations themselves must
propagate any failure so the engine can record it.

There are now two engine-facing wrappers that perform that exception capture:

* :meth:`safe_score` — legacy ``(float, Optional[str])`` tuple interface,
  preserved verbatim for backward compatibility with every existing test
  and call site.
* :meth:`safe_score_with_explanation` — new primary interface that returns
  a :class:`~forge.metrics.result.MetricResult` carrying the score, a
  human-readable explanation, metric-specific metadata, and an explicit
  error channel. ``MetricEngine.run_all_with_explanations`` consumes this.

Subclasses can override :meth:`score_with_explanation` to surface richer
explanations and metadata; the default implementation falls back to
``score()`` and a generic ``"Score: 0.xxx"`` sentence so existing metrics
remain functional without code changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from forge.metrics.result import MetricResult


logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Base class every Forge metric inherits from.

    Subclasses must define :attr:`METRIC_NAME` (a class-level string) and
    :meth:`score`. The :attr:`name` instance property reads from
    ``METRIC_NAME`` so the engine can read metric names directly off the
    class — without instantiating it — during metadata operations such as
    ``MetricEngine.available_metrics()`` and ``MetricEngine.register()``.

    Every concrete ``score`` implementation MUST clamp its return value
    with ``max(0.0, min(1.0, raw_score))`` before returning, and MUST NOT
    swallow exceptions internally — exception handling is the engine's
    responsibility.
    """

    # Class-level identifier. Concrete metrics MUST set this.
    METRIC_NAME: ClassVar[str]

    @property
    @abstractmethod
    def name(self) -> str:
        """Short stable identifier used as the dict key in evaluation results."""

    @abstractmethod
    def score(self, trajectory: dict) -> float:
        """Compute the metric on ``trajectory`` and return a float in [0.0, 1.0]."""

    def score_with_explanation(self, trajectory: dict) -> MetricResult:
        """Compute the score and wrap it with a human-readable explanation.

        Default implementation: call :meth:`score` and wrap the float in
        a :class:`MetricResult` with a generic ``"Score: 0.xxx"``
        explanation and empty metadata. Concrete metrics SHOULD override
        this to return a domain-specific explanation and intermediate
        values (e.g. LCS length, claim counts) in ``metadata``.

        Exceptions raised here are caught by
        :meth:`safe_score_with_explanation`; this method itself must not
        swallow them.
        """
        value = self.score(trajectory)
        return MetricResult(
            score=value,
            explanation=f"Score: {value:.3f}",
            metadata={},
            metric_name=self.name,
        )

    def batch_score(self, trajectories: list) -> list:
        """Score a list of trajectories. Does not catch exceptions."""
        return [self.score(t) for t in trajectories]

    def safe_score(self, trajectory: dict) -> tuple[float, Optional[str]]:
        """Score ``trajectory`` and translate failures into ``(0.0, reason)``.

        On success returns ``(score, None)``. On any exception, logs at
        ERROR with the metric name, the trajectory id (if present), and
        the exception message, then returns ``(0.0, str(exception))``.

        Now routes through :meth:`safe_score_with_explanation` so a
        subclass that overrides ``score_with_explanation`` benefits from
        the same error handling, but the ``(float, Optional[str])``
        return shape is unchanged.
        """
        result = self.safe_score_with_explanation(trajectory)
        return (result.score, result.error_message)

    def safe_score_with_explanation(self, trajectory: dict) -> MetricResult:
        """Run :meth:`score_with_explanation` with exception capture.

        On success returns the metric's own :class:`MetricResult`. On any
        exception, logs at ERROR with the metric name and trajectory id
        and returns :meth:`MetricResult.error_result` with
        ``score=0.0``.
        """
        try:
            return self.score_with_explanation(trajectory)
        except Exception as exc:
            traj_id = (
                trajectory.get("trajectory_id")
                if isinstance(trajectory, dict)
                else None
            )
            logger.error(
                "Metric %r failed on trajectory_id=%s: %s",
                self.name,
                traj_id,
                exc,
            )
            return MetricResult.error_result(self.name, str(exc))
