"""Standalone verification for forge.capture.trajectory.Trajectory.

Run from the project root with the project's virtualenv active:

    python scripts/test_trajectory.py
"""

import json

from forge.capture.trajectory import Trajectory


def main() -> None:
    traj = Trajectory(
        task="What is the capital of France?",
        agent_id="test_agent",
    )

    traj.steps.append(
        {
            "step_index": 0,
            "type": "llm_call",
            "input": {"prompt": "What is the capital of France?"},
            "output": {"text": "I should look this up using a search tool."},
            "duration_ms": 120,
            "tokens": 24,
        }
    )
    traj.steps.append(
        {
            "step_index": 1,
            "type": "tool_call",
            "input": {"tool": "web_search", "query": "capital of France"},
            "output": {"result": "Paris"},
            "duration_ms": 340,
            "tokens": 0,
        }
    )

    traj.validate()
    print("validate() passed")

    as_dict = traj.to_dict()
    print(json.dumps(as_dict, indent=2))

    restored = Trajectory.from_dict(as_dict)
    assert restored.trajectory_id == traj.trajectory_id, (
        f"trajectory_id mismatch: {restored.trajectory_id!r} != {traj.trajectory_id!r}"
    )
    print("from_dict() roundtrip passed")

    bad = Trajectory(task="")
    try:
        bad.validate()
    except ValueError:
        print("validate() correctly caught empty task")
    else:
        raise AssertionError("validate() did not raise ValueError for empty task")


if __name__ == "__main__":
    main()
