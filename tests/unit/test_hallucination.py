"""Unit tests for HallucinationMetric.

Every test patches ``forge.metrics.hallucination.LLMClient`` so no real
Groq/Ollama calls are made and the suite stays hermetic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from forge.metrics.hallucination import HallucinationMetric


def _llm_step(idx: int, output: str) -> dict:
    return {
        "step_index": idx,
        "type": "llm_call",
        "input": "(prompt elided)",
        "output": output,
    }


def _tool_step(idx: int, output, tool_name: str = "search", tool_input: str = "q") -> dict:
    return {
        "step_index": idx,
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "input": tool_input,
        "output": output,
        "tool_output": output,
    }


@pytest.mark.unit
def test_empty_final_answer_returns_one():
    metric = HallucinationMetric()
    traj = {"final_answer": None, "steps": []}
    assert metric.score(traj) == 1.0


@pytest.mark.unit
def test_no_grounding_context_returns_half():
    """final_answer present, but no tool outputs and no retrieved_context -> 0.5."""
    with patch("forge.metrics.hallucination.LLMClient") as MockClient:
        metric = HallucinationMetric()
        traj = {
            "final_answer": "Tokyo is the capital of Japan",
            "steps": [_llm_step(0, "Tokyo is the capital of Japan")],
        }
        assert metric.score(traj) == 0.5
        # The LLM must not be invoked when there's nothing to ground against.
        MockClient.return_value.complete_json.assert_not_called()


@pytest.mark.unit
def test_all_claims_grounded_returns_one():
    with patch("forge.metrics.hallucination.LLMClient") as MockClient:
        client = MockClient.return_value
        client.complete_json.side_effect = [
            {"claims": ["Paris is the capital of France"]},
            {"grounded": True},
        ]
        metric = HallucinationMetric()
        traj = {
            "final_answer": "Paris is the capital of France",
            "steps": [_tool_step(0, "Paris is the capital and most populous city of France")],
        }
        assert metric.score(traj) == 1.0
        assert client.complete_json.call_count == 2


@pytest.mark.unit
def test_no_claims_extracted_returns_one():
    with patch("forge.metrics.hallucination.LLMClient") as MockClient:
        client = MockClient.return_value
        client.complete_json.return_value = {"claims": []}
        metric = HallucinationMetric()
        traj = {
            "final_answer": "Hmm, interesting.",
            "steps": [_tool_step(0, "some grounding context")],
        }
        assert metric.score(traj) == 1.0
        assert client.complete_json.call_count == 1, (
            f"stage 2 must be skipped when no claims; got "
            f"{client.complete_json.call_count} calls"
        )


@pytest.mark.unit
def test_half_claims_grounded_returns_half():
    with patch("forge.metrics.hallucination.LLMClient") as MockClient:
        client = MockClient.return_value
        client.complete_json.side_effect = [
            {"claims": ["Claim A", "Claim B"]},
            {"grounded": True},
            {"grounded": False},
        ]
        metric = HallucinationMetric()
        traj = {
            "final_answer": "Answer asserting A and B.",
            "steps": [_tool_step(0, "context info that supports A but not B")],
        }
        assert metric.score(traj) == pytest.approx(0.5)
        assert client.complete_json.call_count == 3


@pytest.mark.unit
def test_grounding_uses_tool_outputs():
    """The unique tool_output string must appear verbatim in the stage-2 prompt."""
    sentinel = "UNIQUE_GROUNDING_STRING_xyz_abc_42"
    with patch("forge.metrics.hallucination.LLMClient") as MockClient:
        client = MockClient.return_value
        client.complete_json.side_effect = [
            {"claims": ["Some factual claim"]},
            {"grounded": True},
        ]
        metric = HallucinationMetric()
        traj = {
            "final_answer": "An answer that should be checked for hallucinations.",
            "steps": [_tool_step(0, sentinel)],
        }
        metric.score(traj)

        assert client.complete_json.call_count == 2
        stage2_prompt = client.complete_json.call_args_list[1].args[0]
        assert sentinel in stage2_prompt, (
            f"stage-2 prompt should include the tool output as context, "
            f"but {sentinel!r} not found in:\n{stage2_prompt!r}"
        )
