"""Unit tests that close out the MetricEngine: all seven metrics registered,
composite calculation, partial-failure handling, dedup, and result shape.

Every test runs offline. A module-wide autouse fixture mocks
``sentence_transformers.SentenceTransformer`` so instantiating
``ReasoningCoherenceMetric`` never tries to download the BGE model. Tests
that depend on specific metric outputs patch each metric's ``score``
method directly with ``patch.object`` rather than mocking the LLM client
or model internals, which keeps the tests independent of any one metric's
implementation details.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forge.metrics.consistency import MultiTurnConsistencyMetric
from forge.metrics.engine import MetricEngine
from forge.metrics.hallucination import HallucinationMetric
from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric
from forge.metrics.recovery_rate import RecoveryRateMetric
from forge.metrics.step_efficiency import StepEfficiencyMetric
from forge.metrics.task_completion import TaskCompletionMetric
from forge.metrics.tool_call_fidelity import ToolCallFidelityMetric


ALL_SEVEN: list[type] = [
    TaskCompletionMetric,
    ToolCallFidelityMetric,
    StepEfficiencyMetric,
    ReasoningCoherenceMetric,
    HallucinationMetric,
    RecoveryRateMetric,
    MultiTurnConsistencyMetric,
]

ALL_SEVEN_NAMES: list[str] = [
    "task_completion",
    "tool_call_fidelity",
    "step_efficiency",
    "reasoning_coherence",
    "hallucination_score",
    "recovery_rate",
    "multi_turn_consistency",
]


@pytest.fixture(autouse=True)
def _mock_sentence_transformer():
    """Prevent ReasoningCoherenceMetric() from loading the real BGE model.

    The metric does ``from sentence_transformers import SentenceTransformer``
    inside ``__init__``; patching the attribute on the package makes the
    subsequent re-import resolve to the mock.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        # Two orthogonal 2-d vectors — anything callable will get *some*
        # numpy array back if a test happens not to patch score().
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_cls.return_value = mock_model
        yield mock_cls


def _minimal_trajectory() -> dict:
    return {
        "trajectory_id": "00000000-0000-0000-0000-000000000001",
        "task": "demo",
        "agent_id": "test",
        "final_answer": "an answer",
        "steps": [
            {
                "step_index": 0,
                "type": "llm_call",
                "input": "?",
                "output": "an answer",
            }
        ],
    }


def _patch_scores(scores_by_metric: dict[type, float | Exception]) -> ExitStack:
    """Stack of ``patch.object(metric, 'score', ...)`` for a batch of metrics."""
    stack = ExitStack()
    for cls, val in scores_by_metric.items():
        if isinstance(val, Exception):
            stack.enter_context(patch.object(cls, "score", side_effect=val))
        else:
            stack.enter_context(patch.object(cls, "score", return_value=val))
    return stack


# --------------------------------------------------------------------------- tests


@pytest.mark.unit
def test_all_seven_metrics_registered():
    names = MetricEngine().available_metrics()
    assert len(names) == 7, f"expected 7 registered metrics, got {len(names)}: {names}"
    for expected in ALL_SEVEN_NAMES:
        assert expected in names, f"missing {expected!r} from registry: {names}"


@pytest.mark.unit
def test_composite_score_is_mean_of_seven():
    scores = {
        TaskCompletionMetric: 0.10,
        ToolCallFidelityMetric: 0.20,
        StepEfficiencyMetric: 0.30,
        ReasoningCoherenceMetric: 0.40,
        HallucinationMetric: 0.50,
        RecoveryRateMetric: 0.60,
        MultiTurnConsistencyMetric: 0.70,
    }
    with _patch_scores(scores):
        result = MetricEngine().run_all(_minimal_trajectory())

    expected = sum(scores.values()) / 7
    assert round(result["composite_score"], 3) == round(expected, 3), (
        f"composite_score={result['composite_score']} != expected mean {expected}"
    )
    # Every individual metric score also lands in the result dict.
    for cls, val in scores.items():
        assert result[cls().name] == pytest.approx(val)


@pytest.mark.unit
def test_partial_failure_composite_still_computed():
    failing = [TaskCompletionMetric, HallucinationMetric, MultiTurnConsistencyMetric]
    succeeding = [
        ToolCallFidelityMetric,
        StepEfficiencyMetric,
        ReasoningCoherenceMetric,
        RecoveryRateMetric,
    ]

    patches: dict[type, float | Exception] = {}
    for cls in failing:
        patches[cls] = RuntimeError(f"{cls.__name__} simulated failure")
    for cls in succeeding:
        patches[cls] = 0.8

    with _patch_scores(patches):
        result = MetricEngine().run_all(_minimal_trajectory())

    # Failed metrics contribute 0.0, succeeding contribute 0.8.
    expected = (3 * 0.0 + 4 * 0.8) / 7
    assert round(result["composite_score"], 3) == round(expected, 3)
    assert result.get("_has_failures") is True

    errors = result.get("_metric_errors", {})
    assert len(errors) == 3, f"expected 3 error entries, got {len(errors)}: {errors}"
    for cls in failing:
        name = cls().name
        assert name in errors, f"failed metric {name!r} not recorded in _metric_errors"


@pytest.mark.unit
def test_engine_register_deduplication():
    """Deduplicate by name; restore registry via try/finally per the
    isolation pattern established in test_metrics_unit.py."""
    original = list(MetricEngine.ALL_METRICS)
    try:
        before = MetricEngine().available_metrics()
        assert len(before) == 7

        MetricEngine.register(TaskCompletionMetric)
        MetricEngine.register(TaskCompletionMetric)

        after = MetricEngine().available_metrics()
        assert len(after) == 7, f"register() did not dedup: {after}"
        assert len(after) == len(set(after)), f"duplicate name in registry: {after}"
        assert set(after) == set(before)
    finally:
        MetricEngine.ALL_METRICS[:] = original


@pytest.mark.unit
def test_run_all_returns_all_keys():
    with _patch_scores({cls: 0.5 for cls in ALL_SEVEN}):
        result = MetricEngine().run_all(_minimal_trajectory())

    for name in ALL_SEVEN_NAMES:
        assert name in result, f"missing metric key {name!r} in {sorted(result)}"
    assert "composite_score" in result
    # All seven succeed -> no diagnostic keys should be present.
    assert "_metric_errors" not in result
    assert "_has_failures" not in result
