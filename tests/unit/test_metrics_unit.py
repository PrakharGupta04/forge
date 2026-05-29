"""Unit tests for ToolCallFidelityMetric, StepEfficiencyMetric, and the
BaseMetric / MetricEngine failure plumbing as it touches both metrics.

All tests in this module are pure-Python and make no external API calls,
network calls, or filesystem I/O. They are marked ``unit`` and can be run
with::

    pytest -m unit
"""

from __future__ import annotations

import pytest

from forge.metrics.engine import MetricEngine
from forge.metrics.step_efficiency import StepEfficiencyMetric
from forge.metrics.tool_call_fidelity import ToolCallFidelityMetric


def _tool_step(idx: int, tool_name: str, tool_input: str, output: str = "ok") -> dict:
    """Helper: build a tool_call step that satisfies Trajectory.validate()."""
    return {
        "step_index": idx,
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "input": tool_input,
        "output": output,
        "tool_output": output,
    }


# ---------- ToolCallFidelityMetric --------------------------------------------


@pytest.mark.unit
def test_tool_call_fidelity_perfect_match():
    actual = [
        _tool_step(0, "search", "capital of Japan"),
        _tool_step(1, "calculator", "1 + 1"),
    ]
    golden = [
        _tool_step(0, "search", "capital of Japan"),
        _tool_step(1, "calculator", "1 + 1"),
    ]
    traj = {
        "task": "demo",
        "steps": actual,
        "metadata": {"golden_trajectory": {"steps": golden}},
    }
    score = ToolCallFidelityMetric().score(traj)
    assert score == pytest.approx(1.0)


@pytest.mark.unit
def test_tool_call_fidelity_no_golden():
    traj = {"task": "demo", "steps": [_tool_step(0, "search", "x")], "metadata": {}}
    with pytest.raises(ValueError, match="No golden_trajectory in metadata"):
        ToolCallFidelityMetric().score(traj)


@pytest.mark.unit
def test_tool_call_fidelity_partial_match():
    actual = [
        _tool_step(0, "search", "Tokyo population"),
        _tool_step(1, "calculator", "13 * 1000"),
        _tool_step(2, "writer", "draft report"),
    ]
    golden = [
        _tool_step(0, "search", "Tokyo population"),
        _tool_step(1, "database", "lookup metro stats"),
    ]
    traj = {
        "task": "demo",
        "steps": actual,
        "metadata": {"golden_trajectory": {"steps": golden}},
    }
    score = ToolCallFidelityMetric().score(traj)
    assert 0.0 < score < 0.8


@pytest.mark.unit
def test_tool_call_fidelity_lcs_alignment_skips_intermediate():
    """Regression test for Issue 2: matched pairs must come from the LCS
    backtrace, not sequential ``zip``-style pairing. With
    ``actual=[A, B, C]`` and ``golden=[A, C]``, the input similarity for the
    second matched pair must compare ``actual[2]`` against ``golden[1]`` —
    not ``actual[1]``.
    """
    actual = [
        _tool_step(0, "A", "alpha tokens"),
        _tool_step(1, "B", "completely unrelated text"),
        _tool_step(2, "C", "gamma tokens"),
    ]
    golden = [
        _tool_step(0, "A", "alpha tokens"),
        _tool_step(1, "C", "gamma tokens"),
    ]
    metric = ToolCallFidelityMetric()
    pairs = metric._lcs_alignment(actual, golden)
    assert pairs == [(0, 0), (2, 1)]

    traj = {
        "task": "demo",
        "steps": actual,
        "metadata": {"golden_trajectory": {"steps": golden}},
    }
    assert metric.score(traj) == pytest.approx(1.0)


# ---------- StepEfficiencyMetric ----------------------------------------------


@pytest.mark.unit
def test_step_efficiency_exact_minimum():
    traj = {
        "steps": [
            _tool_step(0, "search", "q"),
            _tool_step(1, "calculator", "1+1"),
            _tool_step(2, "writer", "x"),
        ],
        "metadata": {"minimum_steps": 3},
    }
    assert StepEfficiencyMetric().score(traj) == pytest.approx(1.0)


@pytest.mark.unit
def test_step_efficiency_double_steps():
    traj = {
        "steps": [_tool_step(i, "search", f"q{i}") for i in range(6)],
        "metadata": {"minimum_steps": 3},
    }
    assert StepEfficiencyMetric().score(traj) == pytest.approx(0.5)


@pytest.mark.unit
def test_step_efficiency_no_metadata():
    traj = {
        "steps": [
            _tool_step(0, "search", "q"),
            _tool_step(1, "calculator", "1+1"),
        ],
    }
    score = StepEfficiencyMetric().score(traj)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------- Failure plumbing --------------------------------------------------


@pytest.mark.unit
def test_safe_score_returns_zero_on_failure():
    metric = ToolCallFidelityMetric()
    traj = {"task": "demo", "steps": [_tool_step(0, "search", "x")]}
    score, error = metric.safe_score(traj)
    assert score == 0.0
    assert isinstance(error, str) and error


@pytest.mark.unit
def test_engine_marks_failures():
    """Per Issue 3, isolate the registry mutation in a try/finally so that
    earlier or later tests do not see a clobbered ``ALL_METRICS``."""
    original = list(MetricEngine.ALL_METRICS)
    try:
        MetricEngine.ALL_METRICS.clear()
        MetricEngine.register(ToolCallFidelityMetric)

        engine = MetricEngine()
        result = engine.run_all(
            {
                "trajectory_id": "00000000-0000-0000-0000-000000000001",
                "task": "demo",
                "steps": [_tool_step(0, "search", "x")],
            }
        )

        assert result.get("_has_failures") is True
        assert "tool_call_fidelity" in result.get("_metric_errors", {})
        assert result["composite_score"] == 0.0
    finally:
        MetricEngine.ALL_METRICS[:] = original
