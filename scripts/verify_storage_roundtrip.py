"""Manual verification of the trajectory + evaluation storage roundtrip.

End-to-end flow:

    Trajectory -> to_dict -> Database.save_trajectory
                 -> Database.load_trajectory -> MetricEngine.run_all
                 -> Database.save_evaluation

Note: the synthetic trajectory deliberately does not include
``metadata.golden_trajectory``. ``ToolCallFidelityMetric`` is therefore
expected to fail; ``MetricEngine.safe_score`` should convert that failure
into ``(0.0, reason)``, surface it via ``_metric_errors`` /
``_has_failures``, and still produce a numeric ``composite_score``. This
script asserts that contract explicitly.

Requires ``DATABASE_URL`` (Postgres) and ``GROQ_API_KEY`` (for the
LLM-judged ``TaskCompletionMetric``) in ``.env`` and a running Postgres
with ``schema.sql`` applied. Run with::

    python scripts/verify_storage_roundtrip.py
"""

from __future__ import annotations

from dotenv import load_dotenv

from forge.capture.trajectory import Trajectory
from forge.metrics.engine import MetricEngine
from forge.server.db import Database


def main() -> None:
    load_dotenv()

    db = Database()
    if not db.health_check():
        raise RuntimeError("Database health check failed (SELECT 1)")
    print("DB connection healthy")

    traj = Trajectory(
        task="Storage roundtrip test",
        agent_id="roundtrip_test",
        ground_truth="test",
        final_answer="test answer",
    )
    traj.steps.append(
        {
            "step_index": 0,
            "type": "llm_call",
            "input": "Storage roundtrip test",
            "output": "I should answer concisely.",
            "duration_ms": 50,
            "tokens": 12,
        }
    )
    traj.steps.append(
        {
            "step_index": 1,
            "type": "tool_call",
            "tool_name": "echo",
            "tool_input": "test",
            "input": "test",
            "output": "test answer",
            "tool_output": "test answer",
            "duration_ms": 25,
            "tokens": 0,
            "error": None,
        }
    )

    trajectory_dict = traj.to_dict()

    new_uuid = db.save_trajectory(trajectory_dict, "roundtrip_test")
    print(f"Saved trajectory with id: {new_uuid}")

    loaded = db.load_trajectory(new_uuid)
    assert loaded["trajectory_id"] == trajectory_dict["trajectory_id"], (
        f"trajectory_id mismatch: stored={trajectory_dict['trajectory_id']!r} "
        f"loaded={loaded['trajectory_id']!r}"
    )
    assert loaded["task"] == trajectory_dict["task"], (
        f"task mismatch: stored={trajectory_dict['task']!r} loaded={loaded['task']!r}"
    )
    assert len(loaded["steps"]) == 2, f"expected 2 steps, got {len(loaded['steps'])}"
    print("Storage roundtrip assertions passed")

    engine = MetricEngine()
    scores = engine.run_all(loaded)
    print("\nMetricEngine scores:")
    for name, value in scores.items():
        print(f"  {name:>20} : {value}")

    composite = scores["composite_score"]
    assert isinstance(composite, float), (
        f"composite_score must be a float, got {type(composite).__name__}"
    )

    # Per constraint 3: ToolCallFidelityMetric is expected to fail because
    # the synthetic trajectory has no metadata.golden_trajectory. The engine
    # must absorb that failure cleanly and still yield a composite.
    assert scores.get("_has_failures") is True, (
        "_has_failures should be True since tool_call_fidelity has no golden trajectory"
    )
    assert "tool_call_fidelity" in scores.get("_metric_errors", {}), (
        "_metric_errors should record tool_call_fidelity's failure reason"
    )
    print("\nExpected failure path verified:")
    print(f"  tool_call_fidelity error -> {scores['_metric_errors']['tool_call_fidelity']}")
    print(f"  composite_score still produced -> {composite}")
    print("MetricEngine consumed stored trajectory successfully")

    eval_uuid = db.save_evaluation(new_uuid, scores)
    print(f"Evaluation saved to DB (id: {eval_uuid})")

    db.close()
    print("\n=== Storage roundtrip complete ===")


if __name__ == "__main__":
    main()
