"""Standalone verification for forge.metrics.task_completion.TaskCompletionMetric.

Runs the LLM-judged metric against three small hand-crafted trajectories
that bracket the expected scoring scale, then exercises ``safe_score``
once to confirm the engine-facing wrapper.

Usage:

    python scripts/test_task_completion.py

Requires GROQ_API_KEY in the environment (loaded from .env). Exactly four
LLM calls are made: one ``score()`` per trajectory plus one ``safe_score()``.
"""

from __future__ import annotations

from dotenv import load_dotenv

from forge.metrics.task_completion import TaskCompletionMetric


TRAJECTORIES = [
    {
        "label": "correct",
        "expected": ("above", 0.7),
        "trajectory": {
            "trajectory_id": "11111111-1111-1111-1111-111111111111",
            "task": "What is the capital of Japan?",
            "ground_truth": "Tokyo",
            "final_answer": "The capital of Japan is Tokyo",
            "steps": [],
        },
    },
    {
        "label": "refusal",
        "expected": ("below", 0.4),
        "trajectory": {
            "trajectory_id": "22222222-2222-2222-2222-222222222222",
            "task": "What is the capital of Japan?",
            "ground_truth": "Tokyo",
            "final_answer": "I don't know",
            "steps": [],
        },
    },
    {
        "label": "wrong",
        "expected": ("below", 0.5),
        "trajectory": {
            "trajectory_id": "33333333-3333-3333-3333-333333333333",
            "task": "What is the capital of Japan?",
            "ground_truth": "Tokyo",
            "final_answer": "The capital of Japan is Osaka",
            "steps": [],
        },
    },
]


def _check(label: str, score: float, direction: str, threshold: float) -> bool:
    if direction == "above":
        ok = score > threshold
        cmp = ">"
    else:
        ok = score < threshold
        cmp = "<"
    status = "PASS" if ok else "WARN"
    print(f"  expected {cmp} {threshold}: {status}")
    if not ok:
        print(f"  (LLM grade for '{label}' fell outside the expected band)")
    return ok


def main() -> None:
    load_dotenv()

    metric = TaskCompletionMetric()
    scores: dict[str, float] = {}

    for case in TRAJECTORIES:
        traj = case["trajectory"]
        label = case["label"]
        direction, threshold = case["expected"]
        print(f"\n[{label}]")
        print(f"  task   : {traj['task']}")
        print(f"  answer : {traj['final_answer']}")
        score = metric.score(traj)
        scores[label] = score
        print(f"  score  : {score:.3f}")
        _check(label, score, direction, threshold)

    print("\n[safe_score check on 'correct']")
    safe_result = metric.safe_score(TRAJECTORIES[0]["trajectory"])
    print(f"  safe_score returned: {safe_result}")
    assert isinstance(safe_result, tuple) and len(safe_result) == 2, safe_result
    assert isinstance(safe_result[0], float), safe_result
    assert safe_result[1] is None, f"unexpected error reason: {safe_result[1]!r}"
    assert 0.0 <= safe_result[0] <= 1.0, safe_result

    print("\nTaskCompletionMetric tests passed")


if __name__ == "__main__":
    main()
