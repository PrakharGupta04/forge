"""Engine that runs all registered Forge metrics on a single trajectory.

The engine is the only component allowed to catch metric exceptions; it
does so via :meth:`BaseMetric.safe_score_with_explanation` and reports
them via the optional ``_metric_errors`` and ``_has_failures`` keys in
the result dict.

Two public scoring methods coexist:

* :meth:`run_all` — preserved legacy interface that returns
  ``{metric_name: float, "composite_score": float, ...}``. The composite
  is now a *weighted* sum, but with the default
  :class:`~forge.metrics.config.WeightingConfig.equal_weights` it equals
  the arithmetic mean of the seven metric scores (the prior behaviour).
* :meth:`run_all_with_explanations` — new richer interface that returns
  the full :class:`~forge.metrics.result.MetricResult` dict per metric,
  used by the upcoming ``/evaluate`` API endpoint.
"""

from __future__ import annotations

import logging
from typing import Optional

from forge.metrics.base import BaseMetric
from forge.metrics.config import WeightingConfig
from forge.metrics.consistency import MultiTurnConsistencyMetric
from forge.metrics.hallucination import HallucinationMetric
from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric
from forge.metrics.recovery_rate import RecoveryRateMetric
from forge.metrics.result import MetricResult
from forge.metrics.step_efficiency import StepEfficiencyMetric
from forge.metrics.task_completion import TaskCompletionMetric
from forge.metrics.tool_call_fidelity import ToolCallFidelityMetric


logger = logging.getLogger(__name__)


class MetricEngine:
    """Run all (or a subset of) registered metrics on a trajectory.

    The seven-metric registry below is the canonical Forge metric suite.
    Order is significant for the result dict's insertion order and for
    the composite-score weighting. The default
    :class:`WeightingConfig.equal_weights` preserves the historical
    arithmetic-mean composite exactly.
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

    def __init__(
        self,
        metric_names: Optional[list[str]] = None,
        weighting_config: Optional[WeightingConfig] = None,
    ) -> None:
        self.metric_names = metric_names
        self.weighting_config = (
            weighting_config if weighting_config is not None
            else WeightingConfig.equal_weights()
        )

    # ----------------------------------------------------------------- public

    def run_all(self, trajectory: dict) -> dict:
        """Run every selected metric on ``trajectory`` and return the results.

        Result dict shape (unchanged from the legacy interface)::

            {
                "<metric_name>": float,        # one entry per metric
                ...,
                "composite_score": float,       # weighted composite
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

        per_metric = self._run_metrics(trajectory)

        results: dict = {}
        errors: dict[str, str] = {}
        for name, result in per_metric.items():
            results[name] = result.score
            if result.had_error and result.error_message is not None:
                errors[name] = result.error_message

        results["composite_score"] = self._weighted_composite(per_metric)

        if errors:
            results["_metric_errors"] = errors
            results["_has_failures"] = True

        return results

    def run_all_with_explanations(self, trajectory: dict) -> dict:
        """Like :meth:`run_all` but every metric key maps to the full
        ``MetricResult.to_dict()`` (score + explanation + metadata +
        error fields). ``composite_score`` is the weighted composite
        float, same as :meth:`run_all`.

        Used by the API ``/evaluate`` endpoint to surface explanations
        and per-metric metadata to clients.
        """
        if not self.ALL_METRICS:
            logger.warning(
                "MetricEngine.run_all_with_explanations called with no "
                "registered metrics; returning composite_score=0.0"
            )
            return {"composite_score": 0.0}

        per_metric = self._run_metrics(trajectory)

        results: dict = {name: r.to_dict() for name, r in per_metric.items()}
        results["composite_score"] = self._weighted_composite(per_metric)
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

    # ----------------------------------------------------------------- internal

    def _run_metrics(self, trajectory: dict) -> dict[str, MetricResult]:
        """Instantiate and run every selected metric; return name -> result."""
        selected = self._select_metric_classes()
        per_metric: dict[str, MetricResult] = {}
        for metric_cls in selected:
            metric = metric_cls()
            per_metric[metric.name] = metric.safe_score_with_explanation(trajectory)
        return per_metric

    def _weighted_composite(self, per_metric: dict[str, MetricResult]) -> float:
        """Composite = Σ (normalized_weight[name] · score[name]) restricted
        to metrics that both produced a score and appear in the weights
        config.

        With ``WeightingConfig.equal_weights()`` and all seven canonical
        metrics scored, the normalized weights are each ``1/7`` and the
        contributing weight total is ``1.0``, so this collapses to the
        arithmetic mean — exactly preserving the historical composite
        and every existing composite-score test.

        Metrics that are registered but not present in the weights dict
        (e.g. custom registrations without a corresponding weight) do not
        contribute. Returns ``0.0`` if no contributing metric exists.
        """
        normalized = self.weighting_config.normalized_weights()
        contributing = {
            name: w for name, w in normalized.items() if name in per_metric
        }
        total_weight = sum(contributing.values())
        if not contributing or total_weight <= 0:
            return 0.0
        return sum(
            w * per_metric[name].score for name, w in contributing.items()
        ) / total_weight

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
