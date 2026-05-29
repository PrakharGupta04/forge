"""Engine that runs all registered Forge metrics on a single trajectory.

The engine is the only component allowed to catch metric exceptions; it does
so via :meth:`BaseMetric.safe_score` and reports them via the optional
``_metric_errors`` and ``_has_failures`` keys in the result dict.
"""

from __future__ import annotations

import logging
from typing import Optional

from forge.metrics.base import BaseMetric


logger = logging.getLogger(__name__)


class MetricEngine:
    """Run all (or a subset of) registered metrics on a trajectory."""

    ALL_METRICS: list[type[BaseMetric]] = []  # populated after all metrics are implemented

    def __init__(self, metric_names: Optional[list[str]] = None) -> None:
        self.metric_names = metric_names

    def run_all(self, trajectory: dict) -> dict:
        """Run every selected metric on ``trajectory`` and return the results.

        Result dict shape::

            {
                "<metric_name>": float,        # one entry per metric
                ...,
                "composite_score": float,       # arithmetic mean of metric scores
                "_metric_errors": {<name>: <reason>},  # only present on failure
                "_has_failures": True,           # only present on failure
            }

        Empty registry short-circuits to ``{"composite_score": 0.0}``.
        """
        if not self.ALL_METRICS:
            logger.warning(
                "MetricEngine.run_all called with no registered metrics; "
                "returning composite_score=0.0"
            )
            return {"composite_score": 0.0}

        selected = self._select_metric_classes()

        results: dict = {}
        errors: dict[str, str] = {}

        for metric_cls in selected:
            metric = metric_cls()
            score, error = metric.safe_score(trajectory)
            results[metric.name] = score
            if error is not None:
                errors[metric.name] = error

        scores = [v for k, v in results.items() if not k.startswith("_")]
        results["composite_score"] = (sum(scores) / len(scores)) if scores else 0.0

        if errors:
            results["_metric_errors"] = errors
            results["_has_failures"] = True

        return results

    def available_metrics(self) -> list[str]:
        """Return the names of every registered metric (regardless of filter)."""
        return [cls().name for cls in self.ALL_METRICS]

    @classmethod
    def register(cls, metric_class: type[BaseMetric]) -> None:
        """Register ``metric_class`` if no existing metric shares its name."""
        new_name = metric_class().name
        existing_names = {existing().name for existing in cls.ALL_METRICS}
        if new_name not in existing_names:
            cls.ALL_METRICS.append(metric_class)

    def _select_metric_classes(self) -> list[type[BaseMetric]]:
        """Resolve ``self.metric_names`` to a list of metric classes.

        If ``metric_names`` is ``None``, returns every registered metric.
        Unknown names are silently ignored (the registry is authoritative).
        """
        if self.metric_names is None:
            return list(self.ALL_METRICS)
        wanted = set(self.metric_names)
        return [cls for cls in self.ALL_METRICS if cls().name in wanted]
