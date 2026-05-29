"""Manual verification script for the new deterministic metrics.

This is **not** a pytest test file. It is named ``verify_*.py`` (not
``test_*.py``) and lives in ``scripts/`` (outside ``testpaths = tests``) so
pytest discovery cannot pick it up. Run it directly:

    python scripts/verify_fidelity_efficiency.py

The script exercises ToolCallFidelityMetric and StepEfficiencyMetric on
hand-crafted trajectories. Both metrics are deterministic and make no API
calls, so this script runs offline and is safe to re-run cheaply.
"""

from __future__ import annotations

from forge.metrics.step_efficiency import StepEfficiencyMetric
from forge.metrics.tool_call_fidelity import ToolCallFidelityMetric


def _tool_step(idx: int, tool_name: str, tool_input: str, output: str = "ok") -> dict:
    return {
        "step_index": idx,
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "input": tool_input,
        "output": output,
        "tool_output": output,
    }


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def demo_tool_call_fidelity() -> None:
    metric = ToolCallFidelityMetric()

    _section("ToolCallFidelity: perfect match")
    golden = [_tool_step(0, "search", "Tokyo"), _tool_step(1, "calculator", "13e6")]
    actual = [_tool_step(0, "search", "Tokyo"), _tool_step(1, "calculator", "13e6")]
    traj = {
        "task": "Population of Tokyo",
        "steps": actual,
        "metadata": {"golden_trajectory": {"steps": golden}},
    }
    print(f"  score = {metric.score(traj):.3f}  (expected 1.000)")

    _section("ToolCallFidelity: partial match (1 of 2 names align)")
    golden = [_tool_step(0, "search", "Tokyo population"),
              _tool_step(1, "database", "lookup metro stats")]
    actual = [_tool_step(0, "search", "Tokyo population"),
              _tool_step(1, "calculator", "13 * 1000000"),
              _tool_step(2, "writer", "draft")]
    traj = {
        "task": "Population of Tokyo",
        "steps": actual,
        "metadata": {"golden_trajectory": {"steps": golden}},
    }
    print(f"  score = {metric.score(traj):.3f}  (expected ~0.65)")

    _section("ToolCallFidelity: LCS skip — [A,B,C] vs [A,C]")
    actual = [_tool_step(0, "A", "alpha tokens"),
              _tool_step(1, "B", "noise text"),
              _tool_step(2, "C", "gamma tokens")]
    golden = [_tool_step(0, "A", "alpha tokens"),
              _tool_step(1, "C", "gamma tokens")]
    pairs = metric._lcs_alignment(actual, golden)
    print(f"  LCS pairs = {pairs}  (expected [(0, 0), (2, 1)])")
    traj = {"task": "x", "steps": actual,
            "metadata": {"golden_trajectory": {"steps": golden}}}
    print(f"  score = {metric.score(traj):.3f}  (expected 1.000)")

    _section("ToolCallFidelity: missing golden -> raises")
    try:
        metric.score({"task": "x", "steps": actual, "metadata": {}})
    except ValueError as exc:
        print(f"  raised ValueError as expected: {exc}")
    else:
        print("  UNEXPECTED: no exception raised")


def demo_step_efficiency() -> None:
    metric = StepEfficiencyMetric()

    _section("StepEfficiency: actual == minimum (3 == 3)")
    traj = {
        "steps": [_tool_step(i, f"t{i}", "x") for i in range(3)],
        "metadata": {"minimum_steps": 3},
    }
    print(f"  score = {metric.score(traj):.3f}  (expected 1.000)")

    _section("StepEfficiency: actual == 2 * minimum (6 vs 3)")
    traj = {
        "steps": [_tool_step(i, "search", "x") for i in range(6)],
        "metadata": {"minimum_steps": 3},
    }
    print(f"  score = {metric.score(traj):.3f}  (expected 0.500)")

    _section("StepEfficiency: no metadata -> estimated minimum")
    traj = {
        "steps": [
            _tool_step(0, "search", "q"),
            _tool_step(1, "calculator", "1+1"),
            _tool_step(2, "writer", "draft"),
            _tool_step(3, "writer", "redraft"),
        ],
    }
    print(f"  score = {metric.score(traj):.3f}  (3 unique tools + 1 = 4 estimated min, 4 actual)")

    _section("StepEfficiency: zero steps -> 0.0")
    print(f"  score = {metric.score({'steps': []}):.3f}  (expected 0.000)")


def main() -> None:
    demo_tool_call_fidelity()
    demo_step_efficiency()
    print()
    print("Manual verification complete")


if __name__ == "__main__":
    main()
