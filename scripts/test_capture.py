"""Standalone verification for forge.capture.langchain_tracer.ForgeTracer.

Runs a real ReAct agent (Groq LLM + DuckDuckGo search tool) against a
simple factual question and confirms the tracer captures a well-formed
trajectory.

Usage:

    python scripts/test_capture.py

Requires GROQ_API_KEY in the environment (loaded from .env) and network
access for the Groq API and DuckDuckGo search. The ReAct prompt is kept
local to this script so verification has no dependency on LangChain Hub.
"""

from __future__ import annotations

from dotenv import load_dotenv

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from forge.capture.langchain_tracer import ForgeTracer


# Local copy of the canonical hwchase17/react prompt. Kept inline to avoid
# any runtime dependency on LangChain Hub (which now requires
# `dangerously_pull_public_prompt=True` and an API round-trip).
_REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

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


def main() -> None:
    load_dotenv()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    search = DuckDuckGoSearchRun()
    tools = [search]

    prompt = PromptTemplate.from_template(_REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    tracer = ForgeTracer(
        task="What is the capital of Japan?",
        ground_truth="Tokyo",
    )

    result = executor.invoke(
        {"input": "What is the capital of Japan?"},
        config={"callbacks": [tracer]},
    )

    traj = tracer.get_trajectory_dict()

    print(f"steps recorded: {len(traj['steps'])}")
    if traj.get("final_answer"):
        print(f"final_answer  : {traj['final_answer']}")
    else:
        print(f"agent result  : {result.get('output', result)}")
    print(f"total_tokens  : {traj['total_tokens']}")

    assert len(traj["steps"]) >= 1, "tracer captured zero steps"
    required = ("step_index", "type", "input", "output")
    for i, step in enumerate(traj["steps"]):
        missing = [k for k in required if k not in step]
        assert not missing, f"step {i} missing keys {missing}: {step}"
    assert (
        isinstance(traj["trajectory_id"], str) and traj["trajectory_id"]
    ), "trajectory_id must be a non-empty string"

    print("ForgeTracer verification passed")


if __name__ == "__main__":
    main()
