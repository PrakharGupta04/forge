"""Unit tests for RecoveryRateMetric.

Pure structural metric — no mocking needed, no network, no embeddings.
"""

from __future__ import annotations

import pytest

from forge.metrics.recovery_rate import RecoveryRateMetric


def _tool_step(
    idx: int,
    tool_name: str,
    tool_input: str,
    output: str = "ok",
    error: str | None = None,
) -> dict:
    """Tool-call step shaped like what ForgeTracer produces."""
    return {
        "step_index": idx,
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "input": tool_input,
        "output": "" if error else output,
        "tool_output": "" if error else output,
        "duration_ms": 10,
        "tokens": 0,
        "error": error,
    }


def _llm_step(idx: int, output: str) -> dict:
    return {
        "step_index": idx,
        "type": "llm_call",
        "input": "(prompt elided)",
        "output": output,
    }


@pytest.mark.unit
def test_no_failures_returns_one():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            _tool_step(0, "web_search", "Tokyo"),
            _tool_step(1, "calculator", "1+1"),
            _llm_step(2, "All sources gathered, ready to answer."),
        ]
    }
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_failure_followed_by_different_tool_is_recovery():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            _tool_step(0, "web_search", "Tokyo population", error="API timeout"),
            _tool_step(1, "calculator", "1+1"),
        ]
    }
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_failure_followed_by_same_tool_same_input_is_not_recovery():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            _tool_step(0, "web_search", "Tokyo population", error="timeout"),
            _tool_step(1, "web_search", "Tokyo population"),
        ]
    }
    assert metric.score(traj) == 0.0


@pytest.mark.unit
def test_failure_at_last_step_is_not_recovery():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            _tool_step(0, "web_search", "warm-up", output="warm-up result"),
            _tool_step(1, "web_search", "second query", error="500 server error"),
        ]
    }
    assert metric.score(traj) == 0.0


@pytest.mark.unit
def test_failure_followed_by_llm_acknowledging_error_is_recovery():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            _tool_step(0, "web_search", "Tokyo population", error="bad request"),
            _llm_step(1, "The search failed. I'll reason from prior knowledge."),
        ]
    }
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_partial_recovery():
    metric = RecoveryRateMetric()
    traj = {
        "steps": [
            # Failure 1 -> recovery (different tool)
            _tool_step(0, "web_search", "query A", error="timeout"),
            _tool_step(1, "calculator", "1+1"),
            # Failure 2 -> NOT a recovery (same tool, same input)
            _tool_step(2, "calculator", "2+2", error="div by zero"),
            _tool_step(3, "calculator", "2+2"),
        ]
    }
    assert metric.score(traj) == pytest.approx(0.5)


@pytest.mark.unit
def test_failure_followed_by_same_tool_different_input_is_recovery():
    metric = RecoveryRateMetric()
    failed_input = "x"  # length 1
    next_input = "a much longer and more elaborated query for Tokyo"  # length ~50

    # Sanity-check that the lengths actually clear the 20% threshold.
    diff = abs(len(failed_input) - len(next_input)) / max(
        len(failed_input), len(next_input), 1
    )
    assert diff > 0.2

    traj = {
        "steps": [
            _tool_step(0, "web_search", failed_input, error="no results"),
            _tool_step(1, "web_search", next_input),
        ]
    }
    assert metric.score(traj) == 1.0
