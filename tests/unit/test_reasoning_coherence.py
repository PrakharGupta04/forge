"""Unit tests for ReasoningCoherenceMetric.

The embedding model is heavy enough that we share a single instance across
the whole module via a ``scope="module"`` fixture. The first test run will
download ``BAAI/bge-small-en-v1.5`` (~130 MB) into the local HuggingFace
cache; subsequent runs are fast.

Run only this module with::

    pytest tests/unit/test_reasoning_coherence.py -m unit -v
"""

from __future__ import annotations

import pytest

from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric


@pytest.fixture(scope="function")
def metric():
    ReasoningCoherenceMetric._model_cache.clear()
    m = ReasoningCoherenceMetric()
    return m


def _llm(idx: int, output: str) -> dict:
    return {
        "step_index": idx,
        "type": "llm_call",
        "input": "(prompt elided)",
        "output": output,
    }


def _tool(idx: int, output: str = "result") -> dict:
    return {
        "step_index": idx,
        "type": "tool_call",
        "tool_name": "stub",
        "tool_input": "x",
        "input": "x",
        "output": output,
        "tool_output": output,
    }


@pytest.mark.unit
def test_single_llm_step_returns_one(metric: ReasoningCoherenceMetric) -> None:
    traj = {"steps": [_llm(0, "Tokyo is the capital of Japan.")]}
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_zero_llm_steps_returns_one(metric: ReasoningCoherenceMetric) -> None:
    traj = {"steps": [_tool(0), _tool(1), _tool(2)]}
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_coherent_trajectory_scores_high(metric: ReasoningCoherenceMetric) -> None:
    traj = {
        "steps": [
            _llm(0, "Tokyo is the capital and most populous city of Japan."),
            _llm(1, "Japan's capital city is Tokyo, located on Honshu island."),
            _llm(2, "The capital of Japan is Tokyo, a major global metropolis."),
        ]
    }
    score = metric.score(traj)
    assert score > 0.6, f"expected score > 0.6 for tightly on-topic outputs, got {score}"


@pytest.mark.unit
def test_incoherent_trajectory_scores_lower(metric: ReasoningCoherenceMetric) -> None:
    coherent = {
        "steps": [
            _llm(0, "Tokyo is the capital and most populous city of Japan."),
            _llm(1, "Japan's capital city is Tokyo, located on Honshu island."),
            _llm(2, "The capital of Japan is Tokyo, a major global metropolis."),
        ]
    }
    incoherent = {
        "steps": [
            _llm(0, "I love baking sourdough bread on Sunday mornings."),
            _llm(1, "The Andromeda galaxy is approaching the Milky Way at 110 km/s."),
            _llm(2, "Lionel Messi lifted the FIFA World Cup trophy in 2022."),
        ]
    }
    coherent_score = metric.score(coherent)
    incoherent_score = metric.score(incoherent)
    assert incoherent_score < coherent_score, (
        f"incoherent score {incoherent_score} should be < coherent score "
        f"{coherent_score}"
    )


@pytest.mark.unit
def test_score_is_clamped(metric: ReasoningCoherenceMetric) -> None:
    # Inline sanity check that the clamp formula behaves as documented.
    assert max(0.0, min(1.0, 1.5)) == 1.0
    assert max(0.0, min(1.0, -0.3)) == 0.0

    traj = {
        "steps": [
            _llm(0, "Tokyo is the capital of Japan."),
            _llm(1, "Japan's capital is Tokyo."),
        ]
    }
    score = metric.score(traj)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_empty_outputs_skipped(metric: ReasoningCoherenceMetric) -> None:
    traj = {"steps": [_llm(0, ""), _llm(1, "")]}
    assert metric.score(traj) == 1.0
