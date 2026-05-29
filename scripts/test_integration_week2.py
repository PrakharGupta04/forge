"""Week 2 end-to-end integration test.

Drives the full Forge pipeline that Week 2 produced:

    ChatGroq -> AgentExecutor -> ForgeTracer -> Trajectory dict
                                          -> MetricEngine.run_all -> scores

Uses the same LangChain import surface that ``scripts/test_capture.py``
already proved working on this project's installed langchain 1.x stack
(``langchain_classic`` for the legacy AgentExecutor/ReAct/hub APIs;
``langchain_core.prompts`` for ``PromptTemplate``). The hub pull is
wrapped in a try/except so the script keeps working even when LangChain
Hub requires ``dangerously_pull_public_prompt=True`` or is unreachable.

Run with::

    python scripts/test_integration_week2.py

Requires ``GROQ_API_KEY`` in ``.env`` and outbound network access to the
Groq API and to DuckDuckGo (and optionally to LangChain Hub).
"""

from __future__ import annotations

import sys
import traceback

from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from forge.capture.langchain_tracer import ForgeTracer
from forge.metrics.engine import MetricEngine


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


def _build_prompt():
    """Try LangChain Hub; fall back to the canonical inline ReAct prompt."""
    try:
        return hub.pull("hwchase17/react")
    except Exception as exc:
        print(
            f"  WARN: hub.pull('hwchase17/react') failed ({type(exc).__name__}: {exc}); "
            "falling back to inline ReAct prompt"
        )
        return PromptTemplate.from_template(_FALLBACK_REACT_PROMPT)


def main() -> None:
    print("=== Forge Week 2 Integration Test ===")

    load_dotenv()

    # ----- [1/4] Agent setup ---------------------------------------------------
    print("\n[1/4] Setting up LangChain agent...")
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
        max_iterations=4,
    )

    # ----- [2/4] Agent run with tracer ----------------------------------------
    print("\n[2/4] Running agent with ForgeTracer...")
    tracer = ForgeTracer(
        task="What programming language was created by Guido van Rossum?",
        ground_truth="Python",
        agent_id="integration_test_agent",
    )
    result = executor.invoke(
        {"input": "What programming language was created by Guido van Rossum?"},
        config={"callbacks": [tracer]},
    )
    trajectory_dict = tracer.get_trajectory_dict()

    final_answer = trajectory_dict.get("final_answer") or result.get("output", "")
    print(f"  steps captured : {len(trajectory_dict['steps'])}")
    print(f"  final_answer   : {final_answer}")

    # ----- [3/4] Metric evaluation --------------------------------------------
    print("\n[3/4] Running MetricEngine on captured trajectory...")
    engine = MetricEngine()
    scores = engine.run_all(trajectory_dict)
    for name, value in scores.items():
        if name.startswith("_"):
            continue
        if isinstance(value, float):
            print(f"  {name:>22} : {value:.3f}")
        else:
            print(f"  {name:>22} : {value}")

    # ----- [4/4] Assertions ----------------------------------------------------
    print("\n[4/4] Running assertions...")
    try:
        assert trajectory_dict.get("trajectory_id"), (
            "trajectory_id must be a non-empty string"
        )
        assert len(trajectory_dict["steps"]) > 0, "no steps captured"
        assert "task_completion" in scores, "task_completion score missing"
        assert "tool_call_fidelity" in scores, "tool_call_fidelity score missing"
        assert "step_efficiency" in scores, "step_efficiency score missing"
        assert "composite_score" in scores, "composite_score missing"
        assert trajectory_dict["agent_id"] == "integration_test_agent", (
            f"agent_id mismatch: {trajectory_dict['agent_id']!r}"
        )

        # Only the real metric outputs + composite_score are bounded floats.
        # Diagnostic keys like _metric_errors (dict) and _has_failures (bool)
        # carry non-float values by design and are ignored here.
        for name, value in scores.items():
            if name.startswith("_"):
                continue
            assert isinstance(value, float), (
                f"{name} is {type(value).__name__}, expected float"
            )
            assert 0.0 <= value <= 1.0, f"{name}={value} outside [0.0, 1.0]"
    except AssertionError as exc:
        print(f"\nASSERTION FAILED: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== ALL ASSERTIONS PASSED ===")
    print("Week 2 integration test complete. Pipeline is working end-to-end.")


if __name__ == "__main__":
    main()
