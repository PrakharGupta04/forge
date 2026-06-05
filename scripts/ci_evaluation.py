"""Deterministic CI benchmark for Forge — structural metrics only, no LLM calls.

Runs a 3-task slice of the ``factual_research`` benchmark with a mock agent
and scores it using ONLY the three structural metrics that require no LLM
calls, no embedding models, and no external services:

* ``step_efficiency``    — compares step count to the task's minimum_steps
* ``tool_call_fidelity`` — LCS/Jaccard match against the golden trajectory
* ``recovery_rate``      — structural error-recovery accounting

Because none of the LLM/embedding metrics (task_completion, hallucination,
reasoning_coherence, multi_turn_consistency) are instantiated, this script
never touches Groq, Ollama, sentence-transformers, Redis, or the network.
It is reproducible, free, and fast (well under 30 seconds).

Regression gating: the script compares the current composite against a
committed ``ci_results/baseline.json``. A drop of more than 0.1 fails CI
(exit 1). If no baseline exists, the current run is saved as the baseline.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from forge.benchmark.runner import BenchmarkRunner
from forge.metrics.config import WeightingConfig  # noqa: F401 (documented dependency)
from forge.metrics.engine import MetricEngine

# The only three metrics evaluated in CI — all structural, all deterministic.
CI_METRICS = ["step_efficiency", "tool_call_fidelity", "recovery_rate"]

# How far the composite may fall below baseline before CI fails.
REGRESSION_THRESHOLD = 0.1

RESULTS_DIR = Path("ci_results")
RESULTS_FILE = RESULTS_DIR / "eval_results.json"
BASELINE_FILE = RESULTS_DIR / "baseline.json"

BENCHMARK_DOMAIN = "factual_research"
MAX_TASKS = 3


class CIMetricEngine(MetricEngine):
    """MetricEngine restricted to the three structural CI metrics.

    Overrides :meth:`run_all_with_explanations` (the single source of truth
    that :meth:`MetricEngine.run_all` delegates to) so only the CI metrics
    are instantiated and scored. The composite is the plain mean of the
    three CI metric scores — deterministic and independent of the global
    weighting config. ``available_metrics`` is also narrowed so the
    runner's error path builds a consistent zero-score dict.
    """

    def available_metrics(self) -> list[str]:
        return list(CI_METRICS)

    def run_all_with_explanations(self, trajectory: dict) -> dict:
        selected = [
            cls for cls in self.ALL_METRICS if cls.METRIC_NAME in CI_METRICS
        ]
        results: dict = {}
        scores: list[float] = []
        for metric_cls in selected:
            metric = metric_cls()
            result = metric.safe_score_with_explanation(trajectory)
            results[metric.name] = result.to_dict()
            scores.append(result.score)

        results["composite_score"] = (
            sum(scores) / len(scores) if scores else 0.0
        )
        return results


def mock_agent(task: str) -> str:
    """Deterministic mock agent — no external calls."""
    return f"CI mock answer: {task[:60]}"


def _print_summary(results: dict) -> None:
    aggregate = results.get("aggregate_scores", {}) or {}
    print("=" * 56)
    print("Forge CI Evaluation — structural metrics only (no LLM)")
    print("=" * 56)
    print(f"Agent:      {results.get('agent_id')}")
    print(f"Benchmark:  {results.get('benchmark')}")
    print(
        f"Tasks:      {results.get('completed_tasks')}"
        f"/{results.get('total_tasks')} completed"
    )
    print("-" * 56)
    print(f"{'Metric':<24}{'Score':>10}")
    print("-" * 56)
    for key in CI_METRICS:
        value = aggregate.get(key)
        shown = f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"
        print(f"{key:<24}{shown:>10}")
    composite = aggregate.get("composite_score")
    composite_shown = (
        f"{composite * 100:.1f}%" if isinstance(composite, (int, float)) else "n/a"
    )
    print("-" * 56)
    print(f"{'composite_score':<24}{composite_shown:>10}")
    print("=" * 56)


def _composite_of(payload: dict) -> float:
    """Extract composite_score from a results payload, defaulting to 0.0."""
    aggregate = payload.get("aggregate_scores", {}) or {}
    value = aggregate.get("composite_score")
    return float(value) if isinstance(value, (int, float)) else 0.0


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(
        agent_fn=mock_agent,
        metric_engine=CIMetricEngine(),
        db=None,
        agent_id="ci_agent",
    )
    run_result = runner.run(domain=BENCHMARK_DOMAIN, max_tasks=MAX_TASKS)

    results = {
        "agent_id": run_result.get("agent_id", "ci_agent"),
        "benchmark": BENCHMARK_DOMAIN,
        "total_tasks": run_result.get("total_tasks", 0),
        "completed_tasks": run_result.get("completed_tasks", 0),
        "aggregate_scores": run_result.get("aggregate_scores", {}) or {},
        "ci_metrics_only": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_summary(results)

    current_composite = _composite_of(results)

    # --- Regression gate -------------------------------------------------
    if not BASELINE_FILE.exists():
        with BASELINE_FILE.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline established (composite: {current_composite:.4f})")
        return 0

    try:
        with BASELINE_FILE.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"\nBaseline unreadable ({exc}); skipping regression check")
        return 0

    if not (baseline.get("ci_metrics_only") is True and results["ci_metrics_only"] is True):
        print("Baseline incompatible (different metric set), skipping regression check")
        return 0

    baseline_composite = _composite_of(baseline)
    delta = current_composite - baseline_composite

    print(f"\nBaseline composite: {baseline_composite:.4f}")
    print(f"Current composite:  {current_composite:.4f}")
    print(f"Delta:              {delta:+.4f}")

    if delta < -REGRESSION_THRESHOLD:
        print("REGRESSION DETECTED")
        print(
            f"Composite dropped {abs(delta):.4f} "
            f"(threshold {REGRESSION_THRESHOLD}); failing CI."
        )
        return 1

    print("No regression detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
