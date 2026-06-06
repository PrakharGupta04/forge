"""Week 5 integration test — validates the API, the CI script, and the README.

This script exercises the Week 5 deliverables end-to-end against a running
Forge stack (API on http://localhost:8000 with PostgreSQL reachable). It does
NOT test frontend rendering — browser automation is explicitly out of scope;
it only verifies that the dashboard *builds*.

Steps:
    1. API health
    2. Evaluate a trajectory via the API
    3. Retrieve the stored trajectory
    4. Run the deterministic CI evaluation script
    5. README quality checks
    6. Dashboard production build

Run from anywhere; paths are resolved relative to the repository root.
Requires the API to be running: ``uvicorn forge.server.main:app``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx

API_BASE = "http://localhost:8000"
REPO_ROOT = Path(__file__).resolve().parent.parent
CI_RESULTS = REPO_ROOT / "ci_results" / "eval_results.json"
README = REPO_ROOT / "README.md"
DASHBOARD = REPO_ROOT / "dashboard"
DASHBOARD_INDEX = DASHBOARD / "dist" / "index.html"


def _sample_trajectory() -> dict:
    return {
        "trajectory_id": str(uuid.uuid4()),
        "task": "What is the capital of Japan?",
        "agent_id": "week5_itest_agent",
        "ground_truth": "Tokyo",
        "final_answer": "The capital of Japan is Tokyo.",
        "steps": [
            {
                "step_index": 0,
                "type": "llm_call",
                "input": "What is the capital of Japan?",
                "output": "I should search for the capital of Japan.",
                "duration_ms": 100,
                "tokens": 15,
            },
            {
                "step_index": 1,
                "type": "tool_call",
                "tool_name": "web_search",
                "input": "capital of Japan",
                "output": "Tokyo is the capital of Japan.",
                "duration_ms": 400,
                "tokens": 0,
            },
            {
                "step_index": 2,
                "type": "llm_call",
                "input": "Search returned: Tokyo is the capital of Japan.",
                "output": "The capital of Japan is Tokyo.",
                "duration_ms": 200,
                "tokens": 20,
            },
        ],
        "total_duration_ms": 700,
        "total_tokens": 35,
        "metadata": {},
    }


def step1_health() -> None:
    resp = httpx.get(f"{API_BASE}/health", timeout=30)
    assert resp.status_code == 200, f"health status {resp.status_code}"
    body = resp.json()
    assert "components" in body, "health response missing 'components'"
    for comp in ("postgresql", "redis", "embedding_model"):
        assert comp in body["components"], f"health missing component {comp}"
    print("[1/6] API health PASSED")


def step2_evaluate() -> str:
    payload = {
        "trajectory": _sample_trajectory(),
        "include_explanations": True,
        "weighting_strategy": "equal",
    }
    resp = httpx.post(f"{API_BASE}/evaluate", json=payload, timeout=180)
    assert resp.status_code == 200, f"evaluate status {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("scores"), "evaluate response has no scores"
    assert body.get("explanations"), "explanations requested but missing/empty"
    trajectory_id = body.get("trajectory_id")
    assert trajectory_id and trajectory_id != "unsaved", (
        f"trajectory not persisted (trajectory_id={trajectory_id!r}); "
        "is PostgreSQL reachable?"
    )
    print(
        f"[2/6] Evaluate via API PASSED "
        f"(composite={body.get('composite_score'):.3f}, "
        f"{len(body['scores'])} metrics, "
        f"{len(body['explanations'])} explanations)"
    )
    return trajectory_id


def step3_retrieve(trajectory_id: str) -> None:
    resp = httpx.get(f"{API_BASE}/trajectories/{trajectory_id}", timeout=30)
    assert resp.status_code == 200, (
        f"trajectory retrieval status {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body.get("task"), "retrieved trajectory missing 'task'"
    print("[3/6] Trajectory retrieval PASSED")


def step4_ci_script() -> None:
    env = dict(os.environ)  # DATABASE_URL (and the rest) passed through
    proc = subprocess.run(
        [sys.executable, "scripts/ci_evaluation.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, f"ci_evaluation.py exited {proc.returncode}"
    assert CI_RESULTS.exists(), f"missing {CI_RESULTS}"
    with CI_RESULTS.open("r", encoding="utf-8") as f:
        results = json.load(f)
    assert results.get("ci_metrics_only") is True, "ci_metrics_only is not True"
    print("[4/6] CI script PASSED")


def step5_readme() -> None:
    assert README.exists(), f"missing {README}"
    text = README.read_text(encoding="utf-8")
    lower = text.lower()
    assert len(text) > 5000, f"README too short ({len(text)} chars)"
    assert "mermaid" in lower, "README missing mermaid diagram"
    assert "## Metrics" in text, "README missing '## Metrics' section"
    assert "Cohen" in text, "README missing Cohen (validation section)"
    assert "TBD" in text, "README missing TBD (honest validation)"
    assert "limitation" in lower, "README missing limitation (honest metric docs)"
    print("[5/6] README quality PASSED")


def step6_dashboard_build() -> None:
    proc = subprocess.run(
        ["npm", "run", "build"],
        cwd=DASHBOARD,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, f"npm run build exited {proc.returncode}"
    assert DASHBOARD_INDEX.exists(), f"missing {DASHBOARD_INDEX}"
    print("[6/6] Dashboard build PASSED")


def main() -> int:
    print("=== Forge Week 5 Integration Test ===")
    step1_health()
    trajectory_id = step2_evaluate()
    step3_retrieve(trajectory_id)
    step4_ci_script()
    step5_readme()
    step6_dashboard_build()
    print("=== Week 5 Integration Test PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
