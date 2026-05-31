"""Week 3 end-to-end integration test.

Exercises the full Forge Week-3 pipeline against a **real** Groq-backed
LangChain ReAct agent on 3 ``factual_research`` benchmark tasks::

    BenchmarkLoader -> BenchmarkRunner -> AgentExecutor (ChatGroq + DuckDuckGo)
                                       -> ForgeTracer (per-task, injected
                                          into executor's callbacks chain)
                                       -> MetricEngine (7 metrics + composite)
                                       -> aggregate_scores

Two architectural choices that make this test exercise the real
production path (rather than an easier mock-style path):

1. **The ``AgentExecutor`` is passed directly to ``BenchmarkRunner``** —
   no wrapper function. ``BenchmarkRunner._invoke_agent`` now dispatches
   on ``isinstance(agent_fn, AgentExecutor)`` and calls
   ``agent_fn.invoke({"input": task}, config={"callbacks": [tracer]})``,
   so the per-task ``ForgeTracer`` is plumbed into the executor's
   callback chain at invocation time and observes every LLM and tool
   event. A wrapper function would silently fall back to the
   plain-callable branch and produce empty trajectories, defeating the
   point of the integration test.

2. **The DuckDuckGo tool is constructed with ``name="web_search"``** —
   the canonical vocabulary used by every ``factual_research`` benchmark
   task's ``golden_trajectory.steps[*].tool_name``. LangChain's default
   name is ``"duckduckgo_search"``, which would cause
   ``ToolCallFidelityMetric`` to score 0.0 on every task because its
   LCS-over-tool-names alignment would never match. Overriding the name
   at construction is the right place to bind the abstract benchmark
   vocabulary to a concrete LangChain tool implementation — this keeps
   the benchmark JSON framework-agnostic and the runner free of
   tool-alias bookkeeping.

Database persistence is intentionally skipped (``db=None``) so the test
does not require a running Postgres. The LangChain import surface
mirrors ``scripts/test_integration_week2.py`` (``langchain_classic`` for
the legacy AgentExecutor/ReAct/hub APIs, ``langchain_core.prompts`` for
``PromptTemplate``); the hub pull is wrapped in a try/except so the
script keeps working when LangChain Hub is unreachable or requires
``dangerously_pull_public_prompt=True``.

Run with::

    python scripts/test_integration_week3.py

Requires ``GROQ_API_KEY`` in ``.env`` and outbound network access to the
Groq API and to DuckDuckGo (and optionally to LangChain Hub).

Calibration note (not a correctness issue): ``StepEfficiencyMetric``
divides ``metadata.minimum_steps`` (which the benchmark JSON sets to
the number of golden tool calls) by ``len(trajectory.steps)`` (which
counts every LLM + tool step a ReAct agent emits). A correct agent
that takes the minimum number of tool calls will still score below 1.0
because of the interleaved LLM "thought" steps. The metric still ranks
fewer-step agents higher; only the absolute number is conservative.
This is a calibration choice, not a structural defect, so it isn't
asserted against here.
"""

from __future__ import annotations

import sys
import time
import traceback

from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from forge.benchmark.runner import BenchmarkRunner


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


# Canonical Week-3 metric registry. Keep in sync with
# forge/metrics/engine.py::MetricEngine.ALL_METRICS.
EXPECTED_METRIC_NAMES: tuple[str, ...] = (
    "task_completion",
    "tool_call_fidelity",
    "step_efficiency",
    "reasoning_coherence",
    "hallucination_score",
    "recovery_rate",
    "multi_turn_consistency",
)


def _build_prompt():
    """Try LangChain Hub; fall back to the canonical inline ReAct prompt."""
    try:
        return hub.pull("hwchase17/react")
    except Exception as exc:
        print(
            f"  WARN: hub.pull('hwchase17/react') failed "
            f"({type(exc).__name__}: {exc}); falling back to inline ReAct prompt"
        )
        return PromptTemplate.from_template(_FALLBACK_REACT_PROMPT)


def _format_score(scores: dict, name: str) -> str:
    """Render a single score as a fixed-width string, or ``MISSING`` if absent."""
    if name not in scores:
        return "MISSING"
    val = scores[name]
    if isinstance(val, float):
        return f"{val:.3f}"
    return repr(val)


def main() -> None:
    print("=== Forge Week 3 Integration Test ===")

    load_dotenv()

    # ----- [1/5] Agent setup ---------------------------------------------------
    print("\n[1/5] Setting up live ChatGroq + DuckDuckGo ReAct agent...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # Override the tool name to the benchmark's canonical vocabulary
    # (``web_search``); see module docstring for the rationale.
    search = DuckDuckGoSearchRun(name="web_search")
    tools = [search]
    prompt = _build_prompt()
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
    )

    # ----- [2/5] Benchmark run -------------------------------------------------
    # Pass the AgentExecutor directly — BenchmarkRunner._invoke_agent
    # uses isinstance(self.agent_fn, AgentExecutor) to dispatch and will
    # plumb the per-task ForgeTracer into config={"callbacks": [tracer]}.
    # Wrapping in a function would silently fall back to the plain-callable
    # branch and produce empty trajectories.
    print("\n[2/5] Running BenchmarkRunner on 3 factual_research tasks...")
    runner = BenchmarkRunner(
        agent_fn=executor,
        agent_id="groq_react_v1",
        db=None,
    )

    t0 = time.perf_counter()
    result = runner.run_quick(n=3, domain="factual_research")
    elapsed = time.perf_counter() - t0

    # ----- [3/5] Per-task summary ---------------------------------------------
    print("\n[3/5] Per-task results")
    print("-" * 78)
    for tr in result["per_task_results"]:
        task_id = tr.get("task_id", "<unknown>")
        task_str = (tr.get("task") or "")[:80]
        final_answer = tr.get("final_answer") or ""
        scores = tr.get("scores") or {}

        print(f"[{task_id}] {task_str}")
        print(f"  final_answer : {final_answer!r}")
        print("  scores:")
        for name in EXPECTED_METRIC_NAMES:
            print(f"    {name:>24} : {_format_score(scores, name)}")
        print(f"    {'composite_score':>24} : {_format_score(scores, 'composite_score')}")
        if "error" in tr:
            print(f"  ERROR: {tr['error']}")
        if scores.get("_has_failures"):
            print(f"  metric failures: {scores.get('_metric_errors')}")
        print()

    # ----- [4/5] Aggregate + timing -------------------------------------------
    print("[4/5] Aggregate scores")
    print("-" * 78)
    aggregate = result.get("aggregate_scores") or {}
    for name in EXPECTED_METRIC_NAMES:
        print(f"  {name:>24} : {_format_score(aggregate, name)}")
    print(f"  {'composite_score':>24} : {_format_score(aggregate, 'composite_score')}")

    print()
    print(f"  total_tasks     : {result.get('total_tasks')}")
    print(f"  completed_tasks : {result.get('completed_tasks')}")
    print(f"  failed_tasks    : {result.get('failed_tasks')}")
    print(f"  elapsed         : {elapsed:.2f}s (perf_counter)")

    # ----- [5/5] Assertions ----------------------------------------------------
    print("\n[5/5] Running assertions...")
    try:
        # Top-level shape
        assert result.get("total_tasks") == 3, (
            f"total_tasks must be 3, got {result.get('total_tasks')!r}"
        )
        per_task = result.get("per_task_results")
        assert isinstance(per_task, list) and len(per_task) == 3, (
            f"per_task_results must be a list of 3 entries, "
            f"got type={type(per_task).__name__} "
            f"len={len(per_task) if hasattr(per_task, '__len__') else 'n/a'}"
        )

        # Per-task scores: every metric name + composite_score present and bounded
        for i, tr in enumerate(per_task):
            scores = tr.get("scores")
            assert isinstance(scores, dict), (
                f"per_task_results[{i}].scores must be a dict, "
                f"got {type(scores).__name__}"
            )
            for name in EXPECTED_METRIC_NAMES:
                assert name in scores, (
                    f"per_task_results[{i}].scores missing metric {name!r}; "
                    f"keys={sorted(scores.keys())}"
                )
            assert "composite_score" in scores, (
                f"per_task_results[{i}].scores missing 'composite_score'; "
                f"keys={sorted(scores.keys())}"
            )

            # Bounds: skip engine-meta keys (_metric_errors dict, _has_failures bool)
            # which carry non-float values by design.
            for name, val in scores.items():
                if name.startswith("_"):
                    continue
                assert isinstance(val, float), (
                    f"per_task_results[{i}].scores[{name!r}] is "
                    f"{type(val).__name__}, expected float"
                )
                assert 0.0 <= val <= 1.0, (
                    f"per_task_results[{i}].scores[{name!r}]={val} "
                    f"outside [0.0, 1.0]"
                )

        # Aggregate: all 7 metric names present, exactly 8 keys total
        # (the seven metrics plus composite_score, nothing else).
        assert isinstance(aggregate, dict), (
            f"aggregate_scores must be a dict, got {type(aggregate).__name__}"
        )
        for name in EXPECTED_METRIC_NAMES:
            assert name in aggregate, (
                f"aggregate_scores missing metric {name!r}; "
                f"keys={sorted(aggregate.keys())}"
            )
        assert "composite_score" in aggregate, (
            f"aggregate_scores missing 'composite_score'; "
            f"keys={sorted(aggregate.keys())}"
        )

        expected_keys = set(EXPECTED_METRIC_NAMES) | {"composite_score"}
        actual_keys = set(aggregate.keys())
        assert actual_keys == expected_keys, (
            f"aggregate_scores keys must be exactly the 7 metric names plus "
            f"'composite_score'; missing={sorted(expected_keys - actual_keys)} "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

        # Aggregate values must also be bounded floats.
        for name, val in aggregate.items():
            assert isinstance(val, float), (
                f"aggregate_scores[{name!r}] is {type(val).__name__}, expected float"
            )
            assert 0.0 <= val <= 1.0, (
                f"aggregate_scores[{name!r}]={val} outside [0.0, 1.0]"
            )
    except AssertionError as exc:
        print(f"\nASSERTION FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Week 3 Integration Test PASSED ===")
    print(
        f"Ran 3 factual_research tasks with a live Groq agent in "
        f"{elapsed:.2f}s; all 7 metrics + composite_score produced bounded "
        f"floats and aggregated cleanly."
    )


if __name__ == "__main__":
    main()
