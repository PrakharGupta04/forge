"""End-to-end integration tests for the Forge HTTP API.

These tests drive the real ASGI app through ``httpx.AsyncClient`` over
``ASGITransport`` so the full request/response pipeline — including the
lifespan, rate limiter, body-size middleware, dependency injection, and
the Database/Redis singletons — is exercised exactly as it would be in
production. They are skipped at the module level unless
``FORGE_INTEGRATION_TESTS=1`` so the regular ``pytest tests/unit/`` run
stays fast, offline, and free of external-service requirements.

Prerequisites (when ``FORGE_INTEGRATION_TESTS=1``):

* A running PostgreSQL with ``schema.sql`` applied and
  ``migrations/001_add_evaluation_provenance.sql`` applied, reachable
  via ``DATABASE_URL``.
* A running Redis reachable via ``REDIS_URL``.
* ``GROQ_API_KEY`` (or the equivalent ``LLM_PROVIDER`` credentials),
  because :class:`forge.metrics.task_completion.TaskCompletionMetric`
  is an LLM-as-judge metric and ``/evaluate`` will invoke it.

Endpoint coverage notes
-----------------------
Several tests target endpoints that are part of the API's planned
surface but are not implemented yet (``/leaderboard``,
``/trajectories/{id}``, ``/compare``). For those:

* Tests that only assert ``404`` or "not ``500``" pass today *because*
  FastAPI's default routing returns ``404`` for unknown paths, and they
  continue to pass once the real endpoints are implemented correctly.
* Tests that require a ``200``/``[]`` from a not-yet-implemented
  endpoint are marked ``@pytest.mark.xfail(strict=False)``: they
  document the expected contract today, flip to ``XPASS`` once the
  endpoint lands, and never silently break the suite.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport


# Module-level gate. Apply on collection so importing this file in the
# normal unit run is a no-op aside from the skip notice.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FORGE_INTEGRATION_TESTS") != "1",
        reason=(
            "set FORGE_INTEGRATION_TESTS=1 (and provide live Postgres, "
            "Redis, GROQ_API_KEY) to run API integration tests"
        ),
    ),
]


# Importing the app pulls metrics + engine + db, all of which are cheap.
# BenchmarkRunner (and its langchain import chain) is lazy-imported
# inside the background task, so collection time stays low even when the
# integration gate is off.
from forge.server.main import app, limiter  # noqa: E402


# --------------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def app_ctx():
    """Run the ASGI lifespan once per test; yield the live app.

    ``Router.lifespan_context`` is the canonical Starlette API for
    running the application's startup hooks programmatically (it's what
    the ASGI server invokes on real boot). Doing it here ensures
    ``app.state.db`` and ``app.state.redis`` are populated before any
    request is dispatched, and that the corresponding shutdown runs
    when the test finishes — closing the pool and the Redis client
    cleanly between tests.
    """
    async with app.router.lifespan_context(app):
        yield app


@pytest_asyncio.fixture
async def client(app_ctx):
    """An ``AsyncClient`` bound to the live app via ``ASGITransport``."""
    transport = ASGITransport(app=app_ctx)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def db(app_ctx):
    """The live :class:`~forge.server.db.Database` from the lifespan.

    Fails the test with a clear message if PostgreSQL was unreachable
    at startup — integration mode requires a real DB, and a silent
    ``None`` would lead to confusing ``AttributeError`` deep in the
    test body.
    """
    if app_ctx.state.db is None:
        pytest.fail(
            "PostgreSQL is unavailable; integration tests require a "
            "live DB reachable via DATABASE_URL"
        )
    return app_ctx.state.db


# ------------------------------------------------------- helpers / payloads


def _valid_trajectory(task: str = "Integration test task") -> dict:
    """A minimal trajectory that passes :meth:`Trajectory.validate`.

    Includes one ``llm_call`` and one ``tool_call`` so the
    reasoning/tool-fidelity/recovery metrics have something to chew on
    without driving the test wall-clock through the roof. Each call
    gets a fresh ``trajectory_id`` so concurrent calls (e.g. the rate
    limit test) don't collide on the UUID.
    """
    return {
        "task": task,
        "trajectory_id": str(uuid.uuid4()),
        "agent_id": "integration_test_agent",
        "ground_truth": "test",
        "final_answer": "Test answer",
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
                "output": "Test answer",
                "tool_output": "Test answer",
                "error": None,
            },
        ],
    }


# ------------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_health_returns_components(client):
    """GET /health returns the documented component map.

    Both 200 (everything healthy) and 503 (one or more degraded) are
    valid status codes — the embedding-model probe in particular is
    ``False`` until the first ``/evaluate`` call lazy-loads the BGE
    model.
    """
    r = await client.get("/health")
    assert r.status_code in (200, 503), r.text
    body = r.json()
    assert "components" in body
    for key in ("postgresql", "redis", "embedding_model"):
        assert key in body["components"], f"missing component: {key}"
        assert isinstance(body["components"][key], bool)


@pytest.mark.asyncio
async def test_evaluate_returns_explanations(client):
    """POST /evaluate returns per-metric explanations + provenance.

    With ``include_explanations: True`` the response must include a
    populated ``explanations`` map (one entry per registered metric)
    and an ``evaluation_config`` provenance block carrying at minimum
    the ``forge_version`` field.
    """
    r = await client.post(
        "/evaluate",
        json={
            "trajectory": _valid_trajectory(),
            "include_explanations": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["explanations"], (
        "explanations dict must be non-empty when include_explanations=True"
    )
    assert "forge_version" in body["evaluation_config"]


@pytest.mark.asyncio
async def test_evaluate_invalid_trajectory_returns_422(client):
    """A trajectory missing required fields must return 422, not 500."""
    r = await client.post(
        "/evaluate",
        json={"trajectory": {"missing": "required_fields"}},
    )
    assert r.status_code == 422, r.text


@pytest.mark.asyncio
async def test_evaluate_stores_provenance(client, db):
    """The evaluation persists; the trajectory is readable via the public DB API.

    Uses :meth:`Database.get_trajectory_with_evaluation` (public,
    introduced in the Phase 2 pool refactor) rather than reaching into
    private internals — watch-out 7.
    """
    r = await client.post(
        "/evaluate",
        json={"trajectory": _valid_trajectory("Provenance test task")},
    )
    assert r.status_code == 200, r.text
    trajectory_id = r.json()["trajectory_id"]
    assert trajectory_id and trajectory_id != "unsaved", (
        "trajectory must be persisted in integration mode; got "
        f"trajectory_id={trajectory_id!r} — check DB connectivity"
    )

    record = db.get_trajectory_with_evaluation(trajectory_id)
    assert record is not None
    assert record.get("task") == "Provenance test task"
    assert record.get("agent_id") == "integration_test_agent"


@pytest.mark.asyncio
async def test_benchmark_run_queues_job(client):
    """POST /benchmark/run returns a non-empty job_id string.

    Uses ``max_tasks=1`` for a fast smoke run and ``agent_type='mock'``
    so this synthetic test data doesn't pollute the real leaderboard
    (which filters out ``agent_type='mock'`` by design).
    """
    r = await client.post(
        "/benchmark/run",
        json={
            "benchmark": "factual_research",
            "max_tasks": 1,
            "agent_type": "mock",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["job_id"], str) and body["job_id"]


@pytest.mark.asyncio
async def test_benchmark_status_readable(client):
    """GET /benchmark/{job_id} returns a valid in-flight or terminal status.

    Background execution is async; a short poll absorbs the race
    between POST-returning and the background task starting. Any of
    ``queued``/``running``/``complete`` is acceptable — they are all
    legitimate states the job can be observed in.
    """
    r = await client.post(
        "/benchmark/run",
        json={
            "benchmark": "factual_research",
            "max_tasks": 1,
            "agent_type": "mock",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    last_status: str | None = None
    for _ in range(40):
        rs = await client.get(f"/benchmark/{job_id}")
        if rs.status_code == 200:
            last_status = rs.json().get("status")
            if last_status in ("queued", "running", "complete"):
                return
        await asyncio.sleep(0.1)
    pytest.fail(
        f"benchmark {job_id} status never reached a valid state; "
        f"last_status={last_status!r}"
    )


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="GET /leaderboard endpoint not yet implemented; xfail flips to XPASS once it lands",
    strict=False,
)
async def test_leaderboard_excludes_mock_by_default(client):
    """GET /leaderboard returns only non-mock agents.

    The underlying ``Database.get_leaderboard`` already enforces
    ``agent_type != 'mock'``, so this test is really checking that
    the future ``/leaderboard`` endpoint surfaces that filter to
    clients (and doesn't, e.g., add an opt-out flag that bypasses it
    by default).
    """
    r = await client.get("/leaderboard")
    assert r.status_code == 200, r.text
    entries = r.json()
    assert entries == [] or all(
        e.get("agent_type") != "mock" for e in entries
    ), f"leaderboard contains mock entries: {entries}"


@pytest.mark.asyncio
async def test_trajectory_404_for_nonexistent(client):
    """GET /trajectories/{uuid} returns 404 for unknown ids.

    Passes today because FastAPI returns 404 for the (currently
    unknown) route; the assertion remains correct once a real
    ``/trajectories/{id}`` endpoint is wired up and produces 404 for
    a UUID that does not exist in ``trajectories``.
    """
    r = await client.get(
        "/trajectories/00000000-0000-0000-0000-000000000000"
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_compare_404_for_unknown_agents(client):
    """GET /compare with two unknown agent_ids returns 404."""
    r = await client.get(
        "/compare",
        params={"agent_1": "unknown_a", "agent_2": "unknown_b"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_rate_limit_evaluate(client):
    """Bursting 35 concurrent /evaluate requests must trigger ≥1 429.

  The endpoint is decorated ``@limiter.limit("30/minute")``. SlowAPI
  checks the limit in the route decorator **before** the handler body
  runs, so the payload does not need to pass trajectory validation —
  invalid trajectories still consume rate-limit quota and return 422.

  Why not burst 35 full ``_valid_trajectory()`` evaluations?

  * ``POST /evaluate`` runs :meth:`MetricEngine.run_all_with_explanations`
    synchronously inside an ``async def`` handler. That blocks the event
    loop for seconds per request (LLM-as-judge + embeddings), so the 35
    ``asyncio.gather`` calls are processed largely **one at a time**,
    not in a single wall-clock instant.
  * SlowAPI uses a **fixed 1-minute window** (in-memory storage). When
    requests are spaced by several seconds, each window sees only a
    handful of hits and the counter resets before 30 is reached — every
    request can return 200 even though the limiter is wired correctly.
  * Earlier tests in this module also call ``/evaluate``, sharing the
    process-global :data:`limiter` object; without a reset, leftover
    counts from a prior window can mask or exaggerate the burst.

  This test therefore:

  1. Calls :meth:`Limiter.reset` for an isolated quota.
  2. Bursts with a minimal invalid trajectory so handlers return 422
     immediately after the rate-limit check, allowing concurrent
     requests to hit the limiter in the same window.

  ASGITransport does **not** bypass SlowAPI — confirmed by isolated
  probes; production and in-process clients share the same middleware
  and decorator path. All requests resolve to ``127.0.0.1`` via
  :func:`slowapi.util.get_remote_address`, so they share one bucket.
    """
    limiter.reset()

    coros = [
        client.post(
            "/evaluate",
            json={"trajectory": {"missing": "required_fields"}},
        )
        for _ in range(35)
    ]
    responses = await asyncio.gather(*coros, return_exceptions=True)

    statuses: list[int] = []
    for r in responses:
        if isinstance(r, BaseException):
            continue
        statuses.append(r.status_code)

    assert any(s == 429 for s in statuses), (
        f"expected at least one 429 in 35 burst /evaluate requests, "
        f"got status distribution: "
        f"{sorted({s: statuses.count(s) for s in set(statuses)}.items())}"
    )
    # Spot-check: sub-limit requests were rejected at the handler (422),
    # not mistaken for rate-limit failures.
    assert sum(1 for s in statuses if s == 422) >= 1


# ----------------------------------------------- watch-out 6 — extra coverage


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="GET /compare endpoint not yet implemented; xfail flips to XPASS once it lands",
    strict=False,
)
async def test_compare_success_case(client, db):
    """GET /compare with two real agent_ids returns the documented shape.

    The test first seeds the DB by submitting two evaluations under
    distinct ``agent_id``s, then queries ``/compare``. The expected
    response is the :class:`~forge.server.models.CompareResponse`
    shape: ``agent_1``, ``agent_2``, ``agent_1_scores``,
    ``agent_2_scores``, plus optional ``winner`` and ``warning``.
    """
    # Seed two distinct agents so /compare has something to find.
    for agent_id in ("integration_cmp_a", "integration_cmp_b"):
        traj = _valid_trajectory(f"compare seed for {agent_id}")
        traj["agent_id"] = agent_id
        r = await client.post("/evaluate", json={"trajectory": traj})
        assert r.status_code == 200, r.text

    r = await client.get(
        "/compare",
        params={
            "agent_1": "integration_cmp_a",
            "agent_2": "integration_cmp_b",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["agent_1"] == "integration_cmp_a"
    assert body["agent_2"] == "integration_cmp_b"
    assert isinstance(body.get("agent_1_scores"), dict)
    assert isinstance(body.get("agent_2_scores"), dict)


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="GET /leaderboard endpoint not yet implemented; xfail flips to XPASS once it lands",
    strict=False,
)
async def test_empty_leaderboard_returns_list(client):
    """An empty leaderboard slice returns ``[]``, never ``null`` or 404."""
    r = await client.get(
        "/leaderboard",
        params={"benchmark_name": "no_such_benchmark_exists"},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body == []


@pytest.mark.asyncio
async def test_malformed_trajectory_id_no_500(client):
    """A malformed trajectory id must not return 500.

    Passes today because FastAPI returns 404 for the unknown route;
    once a real ``/trajectories/{trajectory_id}`` is implemented it
    must either coerce the path argument via a UUID validator (→ 422)
    or handle the ``psycopg2.DataError`` raised by the bad UUID cast
    (→ 404). Either way the user must never see a 500.
    """
    r = await client.get("/trajectories/not-a-uuid-at-all")
    assert r.status_code != 500, (
        f"malformed trajectory id should not produce 500; got "
        f"{r.status_code}: {r.text}"
    )
