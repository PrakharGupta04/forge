"""Step-efficiency metric.

Rewards agents that solve the task in close to the minimum required number
of steps. ``minimum_steps`` may be supplied via
``trajectory["metadata"]["minimum_steps"]``; otherwise it is estimated from
the number of distinct tools the agent invoked plus one (for the final
reasoning step).

Final score follows the Forge convention of an explicit
``max(0.0, min(1.0, ...))`` clamp, even when the formula appears safe.
"""

from __future__ import annotations

from forge.metrics.base import BaseMetric


class StepEfficiencyMetric(BaseMetric):
    """Score = minimum_steps / actual_steps (capped at 1.0, clamped to [0, 1])."""

    @property
    def name(self) -> str:
        return "step_efficiency"

    def score(self, trajectory: dict) -> float:
        steps = trajectory.get("steps", [])
        actual_steps = len(steps)
        if actual_steps == 0:
            return max(0.0, min(1.0, 0.0))

        metadata = trajectory.get("metadata") or {}
        minimum_steps = metadata.get("minimum_steps")
        if minimum_steps is None:
            tool_names = {
                s.get("tool_name")
                for s in steps
                if s.get("type") == "tool_call" and s.get("tool_name")
            }
            minimum_steps = max(1, len(tool_names) + 1)

        if actual_steps <= minimum_steps:
            raw = 1.0
        else:
            raw = minimum_steps / actual_steps

        return max(0.0, min(1.0, raw))
