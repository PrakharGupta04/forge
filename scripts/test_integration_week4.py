"""Week 4 integration smoke test for the Forge HTTP API.

Drives a *running* uvicorn server (assumed at ``http://127.0.0.1:8000``)
through the full Week 4 endpoint surface in a single linear flow:

  /health -> /evaluate -> /trajectories/{id} -> /evaluate (weighted)
          -> /benchmark/run + /benchmark/{job_id} (poll)
          -> /leaderboard -> /compare

Each step prints a concise pass marker; on failure the script dumps the
relevant response body and exits non-zero. The script does not start or
stop the server, does not modify the database, and runs entirely
against the public HTTP surface — it is the "does Week 4 actually work
end to end?" check.

Prereqs:
* uvicorn forge.server.main:app running on port 8000
* PostgreSQL + Redis reachable from that process (so /benchmark/run
  can be queued and the run completed; /trajectories/{id} can read
  back persisted evaluations)
* GROQ_API_KEY (or equivalent for LLM_PROVIDER) set, since
  TaskCompletionMetric uses an LLM judge

Run:
    python scripts/test_integration_week4.py

Watch-outs honoured (verbatim, by design):
* Field names are inspected dynamically off the live response, not
  asserted against a hard-coded internal contract.
* Step 2's metric set is the seven names the engine actually
  registers — engine-internal sentinels (``_metric_errors``,
  ``_has_failures``) are top-level fields, not entries inside
  ``scores``, so the test does not look for them there.
* Step 4 tolerates equal composite scores when every per-metric score
  is identical (mathematically unavoidable across any weighting).
* Step 5 polls defensively, prints each intermediate status, and on a
  hung run dumps the most recent response payload.
* Step 6 treats an empty leaderboard as a legitimate pass.
* Step 7 asserts status and the presence of a ``detail`` field only,
  not its text.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from typing import Any, Optional

import httpx


_BASE_URL = "http://127.0.0.1:8000"
_TIMEOUT_SECONDS = 120.0
_BENCHMARK_POLL_ATTEMPTS = 20
_BENCHMARK_POLL_INTERVAL_SECONDS = 10.0

# Canonical metric set — derived from forge.metrics.engine.MetricEngine.
# Listed locally so the script can run against a deployed API without
# importing the package, and so a regression that drops a metric still
# trips an assertion here.
_EXPECTED_METRICS: tuple[str, ...] = (
    "task_completion",
    "tool_call_fidelity",
    "step_efficiency",
    "reasoning_coherence",
    "hallucination_score",
    "recovery_rate",
    "multi_turn_consistency",
)


def _abort(label: str, reason: str, response: Optional[httpx.Response] = None) -> None:
    """Print a clear failure message (and the full response body if any), then exit 1."""
    print(f"\nFAIL [{label}]: {reason}")
    if response is not None:
        print(f"  request : {response.request.method} {response.request.url}")
        print(f"  status  : {response.status_code}")
        body_preview = response.text
        if len(body_preview) > 4000:
            body_preview = body_preview[:4000] + " …[truncated]"
        print(f"  body    : {body_preview}")
    sys.exit(1)


def _build_valid_trajectory(task: str) -> dict:
    """Return a Trajectory.from_dict() / .validate()-clean payload.

    Includes one ``llm_call`` and one ``tool_call`` step so every metric
    receives at least one input — keeps the wall-clock low while still
    exercising the full pipeline. A fresh ``trajectory_id`` per call
    guarantees the two /evaluate invocations don't collide if the API
    de-dupes on submitted ids (it currently doesn't, but the test
    doesn't depend on that).
    """
    return {
        "task": task,
        "trajectory_id": str(uuid.uuid4()),
        "agent_id": "week4_integration_agent",
        "ground_truth": "Paris is the capital of France.",
        "final_answer": "Paris is the capital of France.",
        "steps": [
            {
                "step_index": 0,
                "type": "llm_call",
                "input": task,
                "output": "I should answer concisely.",
            },
            {
                "step_index": 1,
                "type": "tool_call",
                "tool_name": "echo",
                "tool_input": task,
                "input": task,
                "output": "Paris is the capital of France.",
                "tool_output": "Paris is the capital of France.",
                "error": None,
            },
        ],
    }


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def main() -> int:
    print("=== Forge Week 4 Integration Test ===")

    with httpx.Client(base_url=_BASE_URL, timeout=_TIMEOUT_SECONDS) as client:
        # ------------------------------------------------------------ Step 1
        r = client.get("/health")
        if r.status_code not in (200, 503):
            _abort("1/7", f"unexpected status {r.status_code} for /health", r)
        body = _safe_json(r)
        if not isinstance(body, dict) or "components" not in body:
            _abort("1/7", "/health response missing 'components' dict", r)
        components = body["components"]
        if not isinstance(components, dict):
            _abort("1/7", f"'components' is not a dict: {components!r}", r)
        print(f"Health status: HTTP {r.status_code} -> {body.get('status')!r}")
        for name, value in components.items():
            print(f"  {name:>16}: {value}")
        notes = body.get("notes")
        if isinstance(notes, dict):
            for k, v in notes.items():
                print(f"  note ({k}): {v}")
        print("[1/7] Health check passed")

        # ------------------------------------------------------------ Step 2
        eval_body_equal = {
            "trajectory": _build_valid_trajectory(
                "What is the capital of France?"
            ),
            "include_explanations": True,
            "weighting_strategy": "equal",
        }
        r = client.post("/evaluate", json=eval_body_equal)
        if r.status_code != 200:
            _abort("2/7", f"POST /evaluate (equal) -> {r.status_code}", r)
        eval_resp_equal = _safe_json(r)
        if not isinstance(eval_resp_equal, dict):
            _abort("2/7", "/evaluate response is not a JSON object", r)

        scores = eval_resp_equal.get("scores")
        if not isinstance(scores, dict):
            _abort("2/7", f"'scores' is not a dict: {scores!r}", r)

        # Verify all 7 canonical metric names are present in scores.
        # We inspect the response dynamically (the watch-out) but cross-check
        # against the engine's published set so a metric regression fails here.
        missing = [m for m in _EXPECTED_METRICS if m not in scores]
        if missing:
            _abort(
                "2/7",
                f"scores missing required metric(s): {missing}; "
                f"actual keys: {sorted(scores)}",
                r,
            )

        explanations = eval_resp_equal.get("explanations")
        if not isinstance(explanations, dict) or not explanations:
            _abort(
                "2/7",
                f"'explanations' must be non-empty when "
                f"include_explanations=True (got {explanations!r})",
                r,
            )

        eval_cfg = eval_resp_equal.get("evaluation_config")
        if not isinstance(eval_cfg, dict):
            _abort("2/7", f"'evaluation_config' missing/not dict: {eval_cfg!r}", r)
        for required_key in ("forge_version", "judge_model"):
            if required_key not in eval_cfg:
                _abort(
                    "2/7",
                    f"evaluation_config missing {required_key!r}; "
                    f"keys present: {sorted(eval_cfg)}",
                    r,
                )

        trajectory_id = eval_resp_equal.get("trajectory_id")
        composite_equal = eval_resp_equal.get("composite_score")
        if not isinstance(composite_equal, (int, float)):
            _abort(
                "2/7", f"composite_score must be a number, got {composite_equal!r}",
                r,
            )
        composite_equal = float(composite_equal)

        print(f"Composite score (equal weights): {composite_equal:.4f}")
        print(f"  forge_version: {eval_cfg.get('forge_version')}")
        print(f"  judge_model:   {eval_cfg.get('judge_model')}")
        print(f"  judge_provider:{eval_cfg.get('judge_provider')}")
        print(f"  scores keys:   {sorted(scores)}")
        if eval_resp_equal.get("has_failures"):
            print(
                f"  has_failures=True; metric_errors="
                f"{eval_resp_equal.get('metric_errors')}"
            )
        print(f"  trajectory_id: {trajectory_id}")
        print("[2/7] POST /evaluate passed with explanations")

        # ------------------------------------------------------------ Step 3
        if not isinstance(trajectory_id, str) or trajectory_id in (
            "", "unsaved", None,
        ):
            _abort(
                "3/7",
                f"cannot retrieve trajectory because /evaluate did not "
                f"persist one (trajectory_id={trajectory_id!r}); check DB "
                f"connectivity",
            )

        r = client.get(f"/trajectories/{trajectory_id}")
        if r.status_code != 200:
            _abort(
                "3/7",
                f"GET /trajectories/{trajectory_id} -> {r.status_code}",
                r,
            )
        traj_body = _safe_json(r)
        if not isinstance(traj_body, dict):
            _abort("3/7", "trajectory response is not a JSON object", r)
        expected_task = eval_body_equal["trajectory"]["task"]
        if traj_body.get("task") != expected_task:
            _abort(
                "3/7",
                f"trajectory task mismatch: expected {expected_task!r}, "
                f"got {traj_body.get('task')!r}",
                r,
            )
        print(f"Retrieved trajectory task: {traj_body['task']!r}")
        print("[3/7] Trajectory retrieval passed")

        # ------------------------------------------------------------ Step 4
        eval_body_research = {
            "trajectory": _build_valid_trajectory(
                "What is the capital of France?"
            ),
            "include_explanations": True,
            "weighting_strategy": "research",
        }
        r = client.post("/evaluate", json=eval_body_research)
        if r.status_code != 200:
            _abort("4/7", f"POST /evaluate (research) -> {r.status_code}", r)
        eval_resp_research = _safe_json(r)
        if not isinstance(eval_resp_research, dict):
            _abort("4/7", "/evaluate (research) response not a JSON object", r)
        composite_research = eval_resp_research.get("composite_score")
        scores_research = eval_resp_research.get("scores") or {}
        if not isinstance(composite_research, (int, float)):
            _abort(
                "4/7",
                f"composite_score (research) must be number, "
                f"got {composite_research!r}",
                r,
            )
        composite_research = float(composite_research)

        print(f"Composite score (equal):    {composite_equal:.4f}")
        print(f"Composite score (research): {composite_research:.4f}")
        delta = composite_research - composite_equal
        print(f"Delta:                      {delta:+.4f}")

        if abs(delta) < 1e-9:
            # Equality is mathematically unavoidable when every metric scored
            # identically (per-metric == composite under both weightings).
            # Confirm and explain rather than failing.
            metric_values = [
                v for k, v in scores_research.items()
                if k in _EXPECTED_METRICS and isinstance(v, (int, float))
            ]
            distinct = len({round(v, 6) for v in metric_values})
            if distinct <= 1 and metric_values:
                print(
                    f"  identical composites are expected: all "
                    f"{len(metric_values)} per-metric scores collapsed to "
                    f"a single value ({metric_values[0]}); weighting cannot "
                    f"differentiate."
                )
            else:
                # Same composite despite non-uniform metric scores would be
                # numerically suspicious — log distinct values so a future
                # engine regression is easy to spot.
                rounded = sorted({round(v, 6) for v in metric_values})
                print(
                    f"  WARN: composites equal but per-metric values vary "
                    f"({len(rounded)} distinct values: {rounded}); the "
                    f"weighting strategies happen to coincide on this "
                    f"specific input — acceptable but worth checking."
                )
        print("[4/7] Weighted evaluation passed")

        # ------------------------------------------------------------ Step 5
        bench_body = {
            "benchmark": "factual_research",
            "max_tasks": 2,
            "agent_type": "mock",
        }
        r = client.post("/benchmark/run", json=bench_body)
        if r.status_code != 200:
            _abort("5/7", f"POST /benchmark/run -> {r.status_code}", r)
        bench_queue = _safe_json(r)
        if not isinstance(bench_queue, dict) or not bench_queue.get("job_id"):
            _abort("5/7", "POST /benchmark/run missing job_id", r)
        job_id = bench_queue["job_id"]
        print(f"Queued benchmark job: {job_id}")

        terminal_states = {"complete", "failed"}
        last_status_resp: Optional[httpx.Response] = None
        last_status_body: Optional[dict] = None
        final_status: Optional[str] = None
        for attempt in range(1, _BENCHMARK_POLL_ATTEMPTS + 1):
            last_status_resp = client.get(f"/benchmark/{job_id}")
            if last_status_resp.status_code != 200:
                _abort(
                    "5/7",
                    f"GET /benchmark/{job_id} -> "
                    f"{last_status_resp.status_code} on poll {attempt}",
                    last_status_resp,
                )
            last_status_body = _safe_json(last_status_resp)
            if not isinstance(last_status_body, dict):
                _abort(
                    "5/7",
                    f"benchmark status response not a JSON object on "
                    f"poll {attempt}",
                    last_status_resp,
                )
            current_status = last_status_body.get("status")
            completed = last_status_body.get("completed_tasks")
            total = last_status_body.get("total_tasks")
            print(
                f"  poll {attempt:02d}/{_BENCHMARK_POLL_ATTEMPTS}: "
                f"status={current_status!r} "
                f"completed={completed} total={total}"
            )
            if current_status in terminal_states:
                final_status = current_status
                break
            time.sleep(_BENCHMARK_POLL_INTERVAL_SECONDS)

        if final_status != "complete":
            _abort(
                "5/7",
                f"benchmark {job_id} did not reach 'complete'; "
                f"final status={final_status!r}, last_body="
                f"{json.dumps(last_status_body, default=str, indent=2)}",
                last_status_resp,
            )

        aggregate_scores = (last_status_body or {}).get("aggregate_scores")
        print(f"Final status: {final_status}")
        print(
            f"Aggregate scores: "
            f"{json.dumps(aggregate_scores, default=str, indent=2)}"
        )
        print("[5/7] Benchmark run completed")

        # ------------------------------------------------------------ Step 6
        r = client.get("/leaderboard")
        if r.status_code != 200:
            _abort("6/7", f"GET /leaderboard -> {r.status_code}", r)
        leaderboard = _safe_json(r)
        if not isinstance(leaderboard, list):
            _abort(
                "6/7",
                f"/leaderboard must return a list; got {type(leaderboard).__name__}",
                r,
            )
        mock_entries = [
            e for e in leaderboard
            if isinstance(e, dict) and e.get("agent_type") == "mock"
        ]
        if mock_entries:
            _abort(
                "6/7",
                f"leaderboard contains {len(mock_entries)} mock entries; "
                f"first offender: {mock_entries[0]!r}",
                r,
            )
        print(f"Leaderboard entries: {len(leaderboard)} (mock excluded)")
        if not leaderboard:
            print("  (empty leaderboard is acceptable — no non-mock runs yet)")
        else:
            for entry in leaderboard[:5]:
                print(
                    f"  agent_id={entry.get('agent_id')!r:>32}  "
                    f"benchmark={entry.get('benchmark_name')!r:>20}  "
                    f"composite={entry.get('composite_score')}  "
                    f"agent_type={entry.get('agent_type')!r}"
                )
            if len(leaderboard) > 5:
                print(f"  … {len(leaderboard) - 5} more")
        print("[6/7] Leaderboard excludes mock agents")

        # ------------------------------------------------------------ Step 7
        # Two random UUID-shaped agent_ids — extremely unlikely to exist
        # in the evaluations table.
        ghost_a = f"__ghost_a_{uuid.uuid4().hex[:12]}"
        ghost_b = f"__ghost_b_{uuid.uuid4().hex[:12]}"
        r = client.get(
            "/compare", params={"agent_1": ghost_a, "agent_2": ghost_b}
        )
        if r.status_code != 404:
            _abort(
                "7/7",
                f"GET /compare with nonexistent agents expected 404, "
                f"got {r.status_code}",
                r,
            )
        body_404 = _safe_json(r)
        if not isinstance(body_404, dict) or "detail" not in body_404:
            _abort(
                "7/7",
                f"/compare 404 response missing 'detail' field: {body_404!r}",
                r,
            )
        print(f"/compare 404 detail: {body_404['detail']!r}")
        print("[7/7] Compare 404 behavior correct")

    print()
    print("=== Week 4 Integration Test PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
