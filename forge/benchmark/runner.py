"""Benchmark runner — orchestrates agent execution, tracing, evaluation, and storage.

The runner ties together the Week 3 components:

* :class:`BenchmarkLoader` supplies tasks from ``data/benchmark``.
* :class:`ForgeTracer` captures each agent run as a Forge ``Trajectory``.
* :class:`MetricEngine` scores the captured trajectory across all 7 metrics.
* Optional :class:`Database` persists the trajectory and evaluation.

The runner uses **native LangChain dispatch** for ``AgentExecutor``
instances: when ``agent_fn`` is an executor, the runner calls
``agent_fn.invoke({"input": task}, config={"callbacks": [tracer]})``
directly so the per-task :class:`ForgeTracer` is plumbed into the
executor's callback chain **at invocation time** and actually observes
every LLM/tool event fired inside the executor. (Construction-time
callbacks would force one executor per task, which defeats the
benchmark runner's whole point.) Any non-``AgentExecutor`` agent is
treated as a plain callable and invoked with the task string directly;
its return value is recorded as the trajectory's ``final_answer``, but
no internal steps are captured because plain callables cannot accept
a callback handler.

A critical orchestration step is the **metadata injection** into the
captured trajectory before evaluation: ``minimum_steps`` and
``golden_trajectory`` from the task JSON are merged into
``trajectory["metadata"]`` so ``ToolCallFidelityMetric`` and
``StepEfficiencyMetric`` receive the data they need.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Callable, Optional

from forge.benchmark.loader import BenchmarkLoader
from forge.capture.langchain_tracer import ForgeTracer
from forge.metrics.engine import MetricEngine
from forge.server.db import Database


# Native dispatch on LangChain ``AgentExecutor`` requires the class to be
# importable for an isinstance check. The 1.x public surface re-homed the
# executor under ``langchain_classic.agents``; we try the canonical
# ``langchain.agents`` path first (forward-compatible with anything that
# re-exports it there) and fall back to ``langchain_classic.agents`` so
# the isinstance branch actually fires on this project's installed stack.
# If neither path is available, ``AgentExecutor`` is set to ``None`` and
# every agent is treated as a plain callable — preserving the runner's
# usability in LangChain-free environments.
try:
    from langchain.agents import AgentExecutor  # type: ignore[import-not-found]
except ImportError:
    try:
        from langchain_classic.agents import (  # type: ignore[import-not-found]
            AgentExecutor,
        )
    except ImportError:
        AgentExecutor = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run an agent against the Forge benchmark suite, score it, optionally persist."""

    def __init__(
        self,
        agent_fn: Callable,
        metric_engine: Optional[MetricEngine] = None,
        db: Optional[Database] = None,
        agent_id: str = "default_agent",
    ) -> None:
        self.agent_fn = agent_fn
        self.metric_engine = (
            metric_engine if metric_engine is not None else MetricEngine()
        )
        self.db = db
        self.agent_id = agent_id

    # ----------------------------------------------------------------- public

    def run(
        self,
        domain: Optional[str] = None,
        max_tasks: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run the agent across the selected benchmark slice and return a result dict."""
        loader = BenchmarkLoader()
        tasks = loader.load(domain=domain)
        if max_tasks is not None:
            tasks = tasks[:max_tasks]

        per_task_results: list[dict] = []
        for task in tasks:
            per_task_results.append(self._run_one(task))

        completed_tasks = sum(1 for r in per_task_results if "error" not in r)
        failed_tasks = len(per_task_results) - completed_tasks
        aggregate_scores = self._aggregate(per_task_results)

        return {
            "agent_id": self.agent_id,
            "domain": domain,
            "total_tasks": len(per_task_results),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "per_task_results": per_task_results,
            "aggregate_scores": aggregate_scores,
        }

    def run_quick(self, n: int = 5, domain: str = "factual_research") -> dict[str, Any]:
        """Convenience: a short-run sanity check (defaults to 5 factual_research tasks)."""
        return self.run(domain=domain, max_tasks=n)

    # ----------------------------------------------------------------- internal

    def _run_one(self, task: dict) -> dict[str, Any]:
        task_id = task.get("task_id", "<unknown>")
        tracer = ForgeTracer(
            task=task["task"],
            ground_truth=task.get("ground_truth"),
            agent_id=self.agent_id,
        )

        try:
            final_answer = self._invoke_agent(task["task"], tracer)
        except Exception as exc:
            logger.error(
                "BenchmarkRunner: agent failed on task %s: %s", task_id, exc
            )
            zero_scores: dict[str, float] = {
                name: 0.0 for name in self.metric_engine.available_metrics()
            }
            zero_scores["composite_score"] = 0.0
            return {
                "task_id": task_id,
                "domain": task.get("domain"),
                "task": task["task"],
                "ground_truth": task.get("ground_truth"),
                "final_answer": "",
                "scores": zero_scores,
                "error": str(exc),
            }

        # Plain callables emit no tracer events, so the tracer never set
        # final_answer. Populate it BEFORE serializing so downstream metrics
        # (task_completion, hallucination) see the agent's real output.
        if not tracer.trajectory.final_answer and final_answer:
            tracer.trajectory.final_answer = final_answer

        trajectory_dict = tracer.get_trajectory_dict()

        # Inject task metadata so the metrics that depend on it can run:
        #   * ToolCallFidelityMetric needs metadata.golden_trajectory
        #   * StepEfficiencyMetric  needs metadata.minimum_steps
        metadata = trajectory_dict.get("metadata") or {}
        metadata.update(
            {
                "task_id": task_id,
                "domain": task.get("domain"),
                "difficulty": task.get("difficulty"),
                "minimum_steps": task.get("minimum_steps"),
                "golden_trajectory": task.get("golden_trajectory"),
            }
        )
        trajectory_dict["metadata"] = metadata

        # Belt and suspenders: also normalize final_answer at the dict level.
        if not trajectory_dict.get("final_answer") and final_answer:
            trajectory_dict["final_answer"] = final_answer

        scores = self.metric_engine.run_all(trajectory_dict)

        result: dict[str, Any] = {
            "task_id": task_id,
            "domain": task.get("domain"),
            "task": task["task"],
            "ground_truth": task.get("ground_truth"),
            "final_answer": trajectory_dict.get("final_answer", "") or "",
            "scores": scores,
        }

        if self.db is not None:
            try:
                trajectory_id = self.db.save_trajectory(trajectory_dict, self.agent_id)
                result["trajectory_id"] = trajectory_id
                self.db.save_evaluation(trajectory_id, scores)
            except Exception as exc:
                logger.error(
                    "BenchmarkRunner: DB persistence failed for task %s: %s",
                    task_id,
                    exc,
                )

        return result

    def _invoke_agent(self, task: str, tracer: ForgeTracer) -> str:
        """Dispatch on agent type; return the final answer as a string.

        Native LangChain branch: when ``self.agent_fn`` is an
        :class:`AgentExecutor`, invoke it with
        ``config={"callbacks": [tracer]}`` so the per-task tracer is
        plumbed into the executor's callback chain and observes every
        LLM and tool event. This is the only way the tracer constructed
        inside :meth:`_run_one` can see what happens inside the
        executor.

        Plain-callable branch: simply call ``self.agent_fn(task)`` —
        plain callables cannot accept a callback handler, so their
        trajectories will carry only ``final_answer`` (no captured
        steps). The ``AgentExecutor is None`` guard preserves this
        branch as the default when LangChain is not installed.
        """
        if AgentExecutor is not None and isinstance(self.agent_fn, AgentExecutor):
            result = self.agent_fn.invoke(
                {"input": task}, config={"callbacks": [tracer]}
            )
            if isinstance(result, dict):
                return result.get("output", "") or ""
            if result is None:
                return ""
            return result if isinstance(result, str) else str(result)

        result = self.agent_fn(task)
        if result is None:
            return ""
        return result if isinstance(result, str) else str(result)

    @staticmethod
    def _aggregate(per_task_results: list[dict]) -> dict[str, float]:
        """Mean each numeric metric (and composite_score) across non-error tasks."""
        good = [r for r in per_task_results if "error" not in r]
        if not good:
            return {}

        per_metric: dict[str, list[float]] = {}
        for r in good:
            for k, v in (r.get("scores") or {}).items():
                # Skip engine-meta keys (_metric_errors dict, _has_failures bool).
                if k.startswith("_"):
                    continue
                # Booleans are int subclasses in Python; exclude explicitly.
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    per_metric.setdefault(k, []).append(float(v))

        return {k: statistics.fmean(vs) for k, vs in per_metric.items() if vs}
