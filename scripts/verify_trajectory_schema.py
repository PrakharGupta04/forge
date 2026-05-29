"""Standalone verification for the Trajectory schema extension.

Confirms three things:

1. A Trajectory with ``conversation_history``, ``retrieved_context``, and
   ``error_summary`` populated survives a ``to_dict() -> from_dict()``
   roundtrip with identical values.
2. An old-style trajectory dict that predates these fields still loads
   cleanly via ``from_dict``, with each new field defaulting to ``None``.
3. ``validate()`` accepts both shapes without raising.

Run with::

    python scripts/verify_trajectory_schema.py
"""

from __future__ import annotations

from forge.capture.trajectory import Trajectory


def main() -> None:
    # --- 1) New-style trajectory: all three new fields populated -------------
    convo = [
        {"role": "user", "content": "Hi, what's the capital of Japan?"},
        {"role": "assistant", "content": "Tokyo."},
        {"role": "user", "content": "And its population?"},
        {"role": "assistant", "content": "About 13.96 million in the metro core."},
    ]
    context = [
        "Tokyo is the capital and most populous city of Japan.",
        "As of 2024, its metropolitan population is roughly 13.96 million.",
    ]
    summary = "Run completed with one tool call that returned stale cached data."

    original = Trajectory(
        task="Population of Tokyo?",
        ground_truth="13.96 million",
        final_answer="About 13.96 million",
        conversation_history=convo,
        retrieved_context=context,
        error_summary=summary,
    )

    as_dict = original.to_dict()
    assert "conversation_history" in as_dict
    assert "retrieved_context" in as_dict
    assert "error_summary" in as_dict

    restored = Trajectory.from_dict(as_dict)
    assert restored.conversation_history == convo, (
        f"conversation_history did not roundtrip:\n  before={convo!r}\n  after={restored.conversation_history!r}"
    )
    assert restored.retrieved_context == context, (
        f"retrieved_context did not roundtrip:\n  before={context!r}\n  after={restored.retrieved_context!r}"
    )
    assert restored.error_summary == summary, (
        f"error_summary did not roundtrip:\n  before={summary!r}\n  after={restored.error_summary!r}"
    )

    # --- 2) Old-style trajectory: none of the new fields present -------------
    legacy_dict = {
        "trajectory_id": "deadbeef-dead-beef-dead-beefdeadbeef",
        "task": "Capital of Japan?",
        "agent_id": "legacy_agent",
        "ground_truth": "Tokyo",
        "final_answer": "Tokyo",
        "steps": [
            {
                "step_index": 0,
                "type": "llm_call",
                "input": "Capital of Japan?",
                "output": "Tokyo.",
            }
        ],
        "timestamp": "2025-01-01T00:00:00",
        "total_duration_ms": 42,
        "total_tokens": 7,
        "metadata": None,
    }
    legacy = Trajectory.from_dict(legacy_dict)
    assert legacy.conversation_history is None, legacy.conversation_history
    assert legacy.retrieved_context is None, legacy.retrieved_context
    assert legacy.error_summary is None, legacy.error_summary

    # --- 3) Validation passes on both shapes ---------------------------------
    original.validate()
    restored.validate()
    legacy.validate()

    print("Trajectory schema update verified")


if __name__ == "__main__":
    main()
