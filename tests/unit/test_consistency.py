"""Unit tests for MultiTurnConsistencyMetric.

Every test patches ``forge.metrics.consistency.LLMClient`` so no real
Groq/Ollama calls are made.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from forge.metrics.consistency import MultiTurnConsistencyMetric


def _llm_step(idx: int, output: str) -> dict:
    return {
        "step_index": idx,
        "type": "llm_call",
        "input": "(prompt elided)",
        "output": output,
    }


def _conv(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@pytest.mark.unit
def test_single_turn_returns_one():
    metric = MultiTurnConsistencyMetric()
    traj = {"steps": [_llm_step(0, "single answer")]}
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_no_conversation_history_single_step_returns_one():
    metric = MultiTurnConsistencyMetric()
    traj = {"conversation_history": None, "steps": [_llm_step(0, "lone answer")]}
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_no_contradiction_returns_one():
    with patch("forge.metrics.consistency.LLMClient") as MockLLM:
        MockLLM.return_value.complete_json.return_value = {
            "contradiction_found": False,
            "explanation": "no contradiction",
        }
        metric = MultiTurnConsistencyMetric()
        traj = {
            "conversation_history": [
                _conv("user", "What is the capital of Japan?"),
                _conv("assistant", "Tokyo is the capital of Japan."),
                _conv("user", "Are you sure?"),
                _conv("assistant", "Yes, Tokyo is correct."),
            ]
        }
        assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_contradiction_found_returns_zero():
    with patch("forge.metrics.consistency.LLMClient") as MockLLM:
        MockLLM.return_value.complete_json.return_value = {
            "contradiction_found": True,
            "explanation": "agent said Tokyo, then Osaka",
        }
        metric = MultiTurnConsistencyMetric()
        traj = {
            "conversation_history": [
                _conv("user", "Capital?"),
                _conv("assistant", "Tokyo is the capital."),
                _conv("user", "Again?"),
                _conv("assistant", "Osaka is the capital."),
            ]
        }
        assert metric.score(traj) == 0.0


@pytest.mark.unit
def test_contradiction_short_circuits():
    """A contradiction on the very first check must skip all remaining turns."""
    with patch("forge.metrics.consistency.LLMClient") as MockLLM:
        MockLLM.return_value.complete_json.return_value = {
            "contradiction_found": True,
            "explanation": "X vs not-X",
        }
        metric = MultiTurnConsistencyMetric()
        traj = {
            "conversation_history": [
                _conv("assistant", "turn 1 (initial)"),
                _conv("assistant", "turn 2 (would contradict)"),
                _conv("assistant", "turn 3 (must not be checked)"),
                _conv("assistant", "turn 4 (must not be checked)"),
            ]
        }
        score = metric.score(traj)
        assert score == 0.0
        assert MockLLM.return_value.complete_json.call_count == 1


@pytest.mark.unit
def test_all_calls_fail_returns_half():
    with patch("forge.metrics.consistency.LLMClient") as MockLLM:
        MockLLM.return_value.complete_json.side_effect = RuntimeError("LLM boom")
        metric = MultiTurnConsistencyMetric()
        traj = {
            "conversation_history": [
                _conv("assistant", "turn 1"),
                _conv("assistant", "turn 2"),
                _conv("assistant", "turn 3"),
            ]
        }
        assert metric.score(traj) == 0.5


@pytest.mark.unit
def test_uses_conversation_history_over_steps_when_longer():
    """With 4 assistant turns vs 2 llm_call outputs, conversation_history wins."""
    with patch("forge.metrics.consistency.LLMClient") as MockLLM:
        MockLLM.return_value.complete_json.return_value = {
            "contradiction_found": False,
            "explanation": "ok",
        }
        metric = MultiTurnConsistencyMetric()

        CONV_INITIAL = "CONV_FIRST_marker_xyz_123"
        STEPS_INITIAL = "STEPS_FIRST_marker_xyz_456"
        traj = {
            "conversation_history": [
                _conv("user", "u1"),
                _conv("assistant", CONV_INITIAL),
                _conv("user", "u2"),
                _conv("assistant", "CONV_TURN_2"),
                _conv("user", "u3"),
                _conv("assistant", "CONV_TURN_3"),
                _conv("user", "u4"),
                _conv("assistant", "CONV_TURN_4"),
            ],
            "steps": [
                _llm_step(0, STEPS_INITIAL),
                _llm_step(1, "STEPS_TURN_2"),
            ],
        }
        metric.score(traj)

        calls = MockLLM.return_value.complete_json.call_args_list
        # 4 assistant turns -> 3 contradiction checks (turn 2/3/4 vs turn 1).
        assert len(calls) == 3, f"expected 3 checks, got {len(calls)}"
        for call in calls:
            prompt = call.args[0]
            assert CONV_INITIAL in prompt, (
                f"every prompt must use the conversation_history first turn "
                f"({CONV_INITIAL!r}) as initial statements:\n{prompt!r}"
            )
            assert STEPS_INITIAL not in prompt, (
                "must NOT use the llm_call step content when assistant turns are richer"
            )
