"""Manual verification for BenchmarkRunner orchestration.

Uses a plain mock agent (no LangChain, no real LLM, no DB) and patches
out the only heavy/billable dependencies the metric engine pulls in:

* ``sentence_transformers.SentenceTransformer`` — so
  ``ReasoningCoherenceMetric()`` does not download or load the BGE model.
* ``forge.metrics.task_completion.LLMClient`` — so the LLM-judged
  ``TaskCompletionMetric`` does not spend Groq credits.

The remaining metrics (``tool_call_fidelity``, ``step_efficiency``,
``recovery_rate``, ``multi_turn_consistency``) are pure-structural and
need no patching, and ``HallucinationMetric`` short-circuits to 0.5 when
there is no grounding context (the mock agent has no tool steps), so it
also makes no LLM call.

Run with::

    python scripts/verify_benchmark_runner.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from forge.benchmark.runner import BenchmarkRunner


def mock_agent(task: str) -> str:
    return "This is a mock answer for: " + task[:50]


REQUIRED_TOP_KEYS = (
    "agent_id",
    "domain",
    "total_tasks",
    "completed_tasks",
    "failed_tasks",
    "per_task_results",
    "aggregate_scores",
)

REQUIRED_PER_TASK_KEYS = (
    "task_id",
    "domain",
    "task",
    "ground_truth",
    "final_answer",
    "scores",
)


def main() -> None:
    with patch("sentence_transformers.SentenceTransformer") as mock_st, patch(
        "forge.metrics.task_completion.LLMClient"
    ) as mock_tc_client:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_st.return_value = mock_model

        mock_tc_client.return_value.complete_json.return_value = {
            "score": 0.7,
            "reasoning": "mock evaluator",
        }

        runner = BenchmarkRunner(
            agent_fn=mock_agent,
            agent_id="mock_agent_v1",
            db=None,
        )
        result = runner.run(domain="factual_research", max_tasks=3)

    print(f"agent_id        : {result['agent_id']}")
    print(f"domain          : {result['domain']}")
    print(f"total_tasks     : {result['total_tasks']}")
    print(f"completed_tasks : {result['completed_tasks']}")
    print(f"failed_tasks    : {result['failed_tasks']}")
    print(f"per_task_results: {len(result['per_task_results'])} entries")
    print(f"aggregate_scores:")
    for name, value in result["aggregate_scores"].items():
        print(f"  {name:>24} : {value:.3f}")

    missing_top = [k for k in REQUIRED_TOP_KEYS if k not in result]
    assert not missing_top, f"result is missing top-level keys: {missing_top}"

    assert result["total_tasks"] == 3, f"total_tasks should be 3, got {result['total_tasks']}"
    assert result["completed_tasks"] == 3, (
        f"completed_tasks should be 3, got {result['completed_tasks']}"
    )
    assert result["failed_tasks"] == 0, (
        f"failed_tasks should be 0, got {result['failed_tasks']}"
    )
    assert len(result["per_task_results"]) == 3, (
        f"expected 3 per_task_results, got {len(result['per_task_results'])}"
    )

    for i, tr in enumerate(result["per_task_results"]):
        missing = [k for k in REQUIRED_PER_TASK_KEYS if k not in tr]
        assert not missing, f"per_task_results[{i}] missing keys: {missing}"
        assert "error" not in tr, (
            f"per_task_results[{i}] unexpectedly carries an error: {tr.get('error')}"
        )
        assert tr["final_answer"].startswith("This is a mock answer for: "), (
            f"per_task_results[{i}].final_answer not populated by runner: "
            f"{tr['final_answer']!r}"
        )
        assert "composite_score" in tr["scores"]

    # Sanity-check aggregation: every aggregate value is a plain float in [0, 1].
    for name, val in result["aggregate_scores"].items():
        assert isinstance(val, float), f"aggregate {name!r} not float: {type(val).__name__}"
        assert 0.0 <= val <= 1.0, f"aggregate {name!r} out of [0,1]: {val}"
    # Engine meta keys must not leak into aggregates.
    for meta_key in ("_metric_errors", "_has_failures"):
        assert meta_key not in result["aggregate_scores"], (
            f"engine meta key {meta_key!r} leaked into aggregate_scores"
        )

    print("\nBenchmarkRunner verification passed")


if __name__ == "__main__":
    main()
