"""LLM-judged task-completion metric.

Asks the configured Forge LLM to grade how well the agent's final answer
satisfies the original task, optionally with a ground-truth reference.

Per the BaseMetric/MetricEngine contract this metric:

* clamps its return value to ``[0.0, 1.0]``;
* never catches LLM exceptions — they propagate so ``MetricEngine.safe_score``
  can convert them into ``(0.0, reason)`` for the engine result dict.
"""

from __future__ import annotations

from typing import Optional

from forge.llm_client import LLMClient
from forge.metrics.base import BaseMetric


_EVALUATOR_INSTRUCTIONS = (
    "You are an objective evaluator assessing whether an AI agent completed "
    "its assigned task. You must respond with a JSON object containing exactly "
    'two keys: "score" as a float between 0.0 and 1.0, and "reasoning" as a '
    "single sentence explanation. Use this scale: 0.0 means the agent "
    "completely failed to address the task or produced a nonsensical answer, "
    "0.3 means the agent attempted the task but the answer is substantially "
    "wrong or incomplete, 0.5 means the agent partially completed the task "
    "with some correct elements, 0.7 means the agent mostly completed the "
    "task with minor gaps or inaccuracies, 1.0 means the agent fully and "
    "correctly completed the task."
)


class TaskCompletionMetric(BaseMetric):
    """Grade an agent's final answer against the task (and optional ground truth)."""

    def __init__(self, llm_provider: Optional[str] = None) -> None:
        super().__init__()
        self._provider = llm_provider

    @property
    def name(self) -> str:
        return "task_completion"

    def score(self, trajectory: dict) -> float:
        task = trajectory.get("task", "")
        final_answer = trajectory.get("final_answer", "")

        if not final_answer:
            for step in reversed(trajectory.get("steps", [])):
                if step.get("type") == "llm_call":
                    final_answer = step.get("output", "")
                    break
            if not final_answer:
                return 0.0

        ground_truth = trajectory.get("ground_truth", "Not provided")
        if ground_truth is None:
            ground_truth = "Not provided"

        prompt = (
            f"{_EVALUATOR_INSTRUCTIONS}\n\n"
            f"Task: {task}\n"
            f"Agent Answer: {final_answer}\n"
            f"Ground Truth: {ground_truth}\n\n"
            "Respond ONLY with the JSON object, nothing else."
        )

        client = (
            LLMClient(provider=self._provider)
            if self._provider is not None
            else LLMClient()
        )
        result = client.complete_json(prompt)
        return max(0.0, min(1.0, float(result["score"])))
