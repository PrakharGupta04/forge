"""Recovery-rate metric — fraction of tool failures the agent recovered from.

A failure is any ``tool_call`` step whose ``error`` field is a non-empty
string. A recovery is detected by **structural heuristics only** on the
step that immediately follows the failure:

* A different tool was called (any non-matching ``tool_name``).
* The same tool was called with a meaningfully different input (string
  lengths differ by more than 20%).
* The agent's next ``llm_call`` output contains a failure-acknowledgement
  keyword (case-insensitive): ``error``, ``failed``, ``unable``, ``try``,
  ``alternative``, ``instead``, ``different``.

No LLM calls are made; this keeps the metric cheap and constant-time per
step so it can run on large benchmark runs without cost or rate-limit
concerns. Trajectories with zero failures return 1.0 by deliberate
design choice — agents that never face failures shouldn't be penalised.
"""

from __future__ import annotations

from forge.metrics.base import BaseMetric


_RECOVERY_KEYWORDS: tuple[str, ...] = (
    "error",
    "failed",
    "unable",
    "try",
    "alternative",
    "instead",
    "different",
)

_INPUT_DIFFERENCE_THRESHOLD = 0.2


class RecoveryRateMetric(BaseMetric):
    """recoveries / failures, with empty-failures collapsing to 1.0."""

    METRIC_NAME = "recovery_rate"

    @property
    def name(self) -> str:
        return self.METRIC_NAME

    def score(self, trajectory: dict) -> float:
        steps = trajectory.get("steps", [])

        failed_indices: list[int] = []
        for i, step in enumerate(steps):
            if step.get("type") != "tool_call":
                continue
            err = step.get("error")
            if err is None:
                continue
            if not isinstance(err, str) or not err:
                continue
            failed_indices.append(i)

        if not failed_indices:
            return 1.0

        recoveries = 0
        for fi in failed_indices:
            next_i = fi + 1
            if next_i >= len(steps):
                continue
            if self._is_recovery(steps[fi], steps[next_i]):
                recoveries += 1

        return max(0.0, min(1.0, recoveries / len(failed_indices)))

    def _is_recovery(self, failed_step: dict, next_step: dict) -> bool:
        next_type = next_step.get("type")

        if next_type == "tool_call":
            failed_tool = failed_step.get("tool_name")
            next_tool = next_step.get("tool_name")
            if failed_tool != next_tool:
                return True
            # Same tool — recovery iff input length changed meaningfully.
            failed_input = self._step_input(failed_step)
            next_input = self._step_input(next_step)
            denom = max(len(failed_input), len(next_input), 1)
            diff_ratio = abs(len(failed_input) - len(next_input)) / denom
            return diff_ratio > _INPUT_DIFFERENCE_THRESHOLD

        if next_type == "llm_call":
            raw_out = next_step.get("output", "")
            text = raw_out if isinstance(raw_out, str) else str(raw_out)
            lower = text.lower()
            return any(kw in lower for kw in _RECOVERY_KEYWORDS)

        return False

    @staticmethod
    def _step_input(step: dict) -> str:
        """Resolve a step's input as a string, preferring tool_input over input."""
        raw = step.get("tool_input", step.get("input", ""))
        return raw if isinstance(raw, str) else str(raw)
