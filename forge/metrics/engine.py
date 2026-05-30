"""Engine that runs all registered Forge metrics on a single trajectory.

The engine is the only component allowed to catch metric exceptions; it does
so via :meth:`BaseMetric.safe_score` and reports them via the optional
``_metric_errors`` and ``_has_failures`` keys in the result dict.
"""

from __future__ import annotations

import logging
from typing import Optional

from forge.metrics.base import BaseMetric
from forge.metrics.consistency import MultiTurnConsistencyMetric
from forge.metrics.hallucination import HallucinationMetric
from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric
from forge.metrics.recovery_rate import RecoveryRateMetric
from forge.metrics.step_efficiency import StepEfficiencyMetric
from forge.metrics.task_completion import TaskCompletionMetric
from forge.metrics.tool_call_fidelity import ToolCallFidelityMetric


logger = logging.getLogger(__name__)


class MetricEngine:
    """Run all (or a subset of) registered metrics on a trajectory.

    The seven-metric registry below is the canonical Forge metric suite.
    Order is significant for the result dict's insertion order and for the
    composite-score weighting (currently all weights are equal: the
    arithmetic mean).
    """

    ALL_METRICS: list[type[BaseMetric]] = [
        TaskCompletionMetric,
        ToolCallFidelityMetric,
        StepEfficiencyMetric,
        ReasoningCoherenceMetric,
        HallucinationMetric,
        RecoveryRateMetric,
        MultiTurnConsistencyMetric,
    ]

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
        """Return the names of every registered metric (regardless of filter).

        Reads ``cls.METRIC_NAME`` directly so this metadata operation never
        instantiates a metric — heavyweight ``__init__`` work (e.g. the
        SentenceTransformer load in ``ReasoningCoherenceMetric``) stays off
        the name-lookup path.
        """
        return [cls.METRIC_NAME for cls in self.ALL_METRICS]

    @classmethod
    def register(cls, metric_class: type[BaseMetric]) -> None:
        """Register ``metric_class`` if no existing metric shares its name.

        Reads ``METRIC_NAME`` off the class — no instantiation here either.
        """
        new_name = metric_class.METRIC_NAME
        existing_names = {existing.METRIC_NAME for existing in cls.ALL_METRICS}
        if new_name not in existing_names:
            cls.ALL_METRICS.append(metric_class)

    def _select_metric_classes(self) -> list[type[BaseMetric]]:
        """Resolve ``self.metric_names`` to a list of metric classes.

        If ``metric_names`` is ``None``, returns every registered metric.
        Unknown names are silently ignored (the registry is authoritative).
        Name lookup uses ``cls.METRIC_NAME`` so filtering never triggers a
        metric's ``__init__``.
        """
        if self.metric_names is None:
            return list(self.ALL_METRICS)
        wanted = set(self.metric_names)
        return [cls for cls in self.ALL_METRICS if cls.METRIC_NAME in wanted]
