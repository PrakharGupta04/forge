"""Manual verification for BenchmarkRunner orchestration.

Two checks are run end-to-end:

1. ``_verify_mock_agent`` — uses a plain mock agent (no LangChain, no real
   LLM, no DB) and patches the only heavy/billable dependencies the metric
   engine pulls in:

   * ``sentence_transformers.SentenceTransformer`` — so
     ``ReasoningCoherenceMetric()`` does not download or load the BGE model.
   * ``forge.metrics.task_completion.LLMClient`` — so the LLM-judged
     ``TaskCompletionMetric`` does not spend Groq credits.

   The remaining metrics (``tool_call_fidelity``, ``step_efficiency``,
   ``recovery_rate``, ``multi_turn_consistency``) are pure-structural and
   need no patching, and ``HallucinationMetric`` short-circuits to 0.5 when
   there is no grounding context (the mock agent has no tool steps), so it
   also makes no LLM call.

2. ``_verify_langchain_executor`` — the regression test for the native
   LangChain dispatch fix. A real ``ChatGroq`` + ``DuckDuckGoSearchRun``
   ``AgentExecutor`` is passed **directly** as ``agent_fn`` (no wrapper
   function). The runner is expected to detect the executor via
   ``isinstance`` and plumb the per-task ``ForgeTracer`` into its callback
   chain at invocation time, so the captured trajectory must contain at
   least one step. The check uses a ``_CapturingMetricEngine`` injected
   via the existing ``metric_engine`` constructor argument to observe the
   trajectory dict the runner produces; this avoids changing the runner's
   public result format and also short-circuits the seven real metrics so
   the verifier does not pay for the LLM judge or the BGE model load.
   The LangChain check is skipped (not failed) when ``GROQ_API_KEY`` is
   not present, so the mock check remains useful in offline environments.

Run with::

    python scripts/verify_benchmark_runner.py
"""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
from dotenv import load_dotenv

from forge.benchmark.runner import BenchmarkRunner
from forge.metrics.engine import MetricEngine


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


_FALLBACK_REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class _CapturingMetricEngine(MetricEngine):
    """Drop-in :class:`MetricEngine` that records the trajectory dict it scores.

    Used by :func:`_verify_langchain_executor` to inspect what the runner
    actually hands to the engine after the agent runs. Returns trivial
    bounded scores so the runner's downstream logic (aggregation, error
    handling) keeps working without paying for real metric scoring.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured_trajectories: list[dict] = []

    def run_all(self, trajectory: dict) -> dict:  # type: ignore[override]
        self.captured_trajectories.append(trajectory)
        scores = {name: 0.0 for name in self.available_metrics()}
        scores["composite_score"] = 0.0
        return scores


def _build_prompt() -> Any:
    """Try LangChain Hub; fall back to the canonical inline ReAct prompt."""
    from langchain_classic import hub
    from langchain_core.prompts import PromptTemplate

    try:
        return hub.pull("hwchase17/react")
    except Exception as exc:
        print(
            f"  WARN: hub.pull('hwchase17/react') failed "
            f"({type(exc).__name__}: {exc}); falling back to inline ReAct prompt"
        )
        return PromptTemplate.from_template(_FALLBACK_REACT_PROMPT)


def _verify_mock_agent() -> None:
    print("=== [1/2] Mock-agent orchestration check ===")
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

    for name, val in result["aggregate_scores"].items():
        assert isinstance(val, float), f"aggregate {name!r} not float: {type(val).__name__}"
        assert 0.0 <= val <= 1.0, f"aggregate {name!r} out of [0,1]: {val}"
    for meta_key in ("_metric_errors", "_has_failures"):
        assert meta_key not in result["aggregate_scores"], (
            f"engine meta key {meta_key!r} leaked into aggregate_scores"
        )

    print("\nMock-agent BenchmarkRunner verification passed\n")


def _verify_langchain_executor() -> Optional[bool]:
    """Verify native AgentExecutor dispatch produces a non-empty trajectory.

    Returns ``True`` on success, ``None`` if skipped (e.g., no Groq API
    key). Raises ``AssertionError`` on a real regression.
    """
    print("=== [2/2] Native LangChain AgentExecutor dispatch check ===")

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print(
            "  SKIPPED: GROQ_API_KEY not set; the LangChain-dispatch check "
            "requires a live ChatGroq call. Set GROQ_API_KEY in .env to run it."
        )
        return None

    # Imports are local so the offline mock check above does not pay the
    # import-time cost of LangChain providers when the second check is
    # skipped.
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_groq import ChatGroq

    print("  Building ChatGroq + DuckDuckGo ReAct executor...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    search = DuckDuckGoSearchRun()
    tools = [search]
    prompt = _build_prompt()
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    capturing_engine = _CapturingMetricEngine()
    runner = BenchmarkRunner(
        agent_fn=executor,
        agent_id="groq_react_v1",
        metric_engine=capturing_engine,
        db=None,
    )

    print("  Running 1 factual_research task with the executor passed directly...")
    result = runner.run(domain="factual_research", max_tasks=1)

    assert result["total_tasks"] == 1, (
        f"expected exactly 1 task, got total_tasks={result['total_tasks']}"
    )
    assert len(result["per_task_results"]) == 1, (
        f"expected 1 per_task_results entry, got {len(result['per_task_results'])}"
    )

    per_task = result["per_task_results"][0]
    if "error" in per_task:
        # An agent or network failure here is unrelated to the dispatch
        # bug we are verifying, but it does invalidate the check — surface
        # it loudly rather than silently passing.
        raise AssertionError(
            f"agent raised before any steps could be captured: {per_task['error']!r}"
        )

    assert len(capturing_engine.captured_trajectories) == 1, (
        f"_CapturingMetricEngine expected exactly 1 trajectory, got "
        f"{len(capturing_engine.captured_trajectories)}"
    )
    trajectory = capturing_engine.captured_trajectories[0]
    steps = trajectory.get("steps") or []
    step_count = len(steps)

    print(f"  task_id      : {per_task.get('task_id')}")
    print(f"  task         : {(per_task.get('task') or '')[:80]}")
    print(f"  final_answer : {per_task.get('final_answer')!r}")
    print(f"  step count   : {step_count}")
    if step_count:
        step_types = [s.get("step_type") or s.get("type") for s in steps]
        print(f"  step types   : {step_types}")

    assert step_count > 0, (
        "native LangChain dispatch regression: the tracer captured zero "
        "steps when the AgentExecutor was passed directly. This is the "
        "exact failure mode the runner fix is supposed to prevent — the "
        "per-task ForgeTracer must be plumbed into the executor's callback "
        "chain via config={'callbacks': [tracer]} at invocation time."
    )

    print("\nNative LangChain executor trajectory capture verified")
    return True


def main() -> None:
    _verify_mock_agent()
    langchain_outcome = _verify_langchain_executor()
    print()
    print("BenchmarkRunner verification passed")
    if langchain_outcome is None:
        print("(LangChain-dispatch check was skipped; set GROQ_API_KEY to enable it.)")


if __name__ == "__main__":
    main()
