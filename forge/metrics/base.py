"""Abstract base class for Forge metrics.

Concrete metrics implement :meth:`BaseMetric.score`. The engine is the only
component allowed to catch exceptions; metric implementations themselves must
propagate any failure so the engine can record it. ``safe_score`` is the
engine-facing wrapper that performs that exception capture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Optional


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

    def batch_score(self, trajectories: list) -> list:
        """Score a list of trajectories. Does not catch exceptions."""
        return [self.score(t) for t in trajectories]

    def safe_score(self, trajectory: dict) -> tuple[float, Optional[str]]:
        """Score ``trajectory`` and translate failures into ``(0.0, reason)``.

        On success returns ``(score, None)``. On any exception, logs at ERROR
        with the metric name, the trajectory id (if present), and the
        exception message, then returns ``(0.0, str(exception))``.
        """
        try:
            return (self.score(trajectory), None)
        except Exception as exc:
            traj_id = (
                trajectory.get("trajectory_id") if isinstance(trajectory, dict) else None
            )
            logger.error(
                "Metric %r failed on trajectory_id=%s: %s",
                self.name,
                traj_id,
                exc,
            )
            return (0.0, str(exc))
