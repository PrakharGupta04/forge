"""Forge FastAPI application — root, health, lifespan, rate-limit, body-size.

This module is the entrypoint for ``uvicorn forge.server.main:app``. It
sets up the long-lived resources (Postgres connection pool, Redis
client) at startup, tears them down at shutdown, and exposes two
endpoints today:

* ``GET /``       — service metadata
* ``GET /health`` — deep health check across PostgreSQL, Redis, and the
                    sentence-transformer embedding-model cache

Future endpoints (``/evaluate``, ``/benchmarks/run``,
``/benchmarks/{job_id}``, ``/compare``, ``/leaderboard``) will be added
in subsequent phases; this module is intentionally narrow so the
foundation can be verified end-to-end before stacking application
logic on top.

Startup-failure tolerance
-------------------------
``Database()`` opens ``DB_POOL_MIN`` connections eagerly via
:class:`psycopg2.pool.ThreadedConnectionPool`, so if Postgres is
unreachable, construction raises. The lifespan handler catches that
failure and stores ``None`` in ``app.state.db`` instead of crashing
the whole process — otherwise ``GET /health`` could never report a
``503 degraded`` state because the API would have already failed to
start. The same tolerance applies to Redis (although ``redis.from_url``
is lazy and only connects on the first command, so the failure path
there is via ``r.ping()`` inside ``/health``).

Rate limiting & body-size limit
-------------------------------
``slowapi`` provides a 100-requests-per-minute default per client IP.
A separate ASGI middleware enforces a 10 MB hard cap on request bodies
via the ``Content-Length`` header (chunked / unknown-length bodies are
allowed through and would be limited by uvicorn's own protocol-level
checks — this middleware is the fast pre-check for normal clients).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time  # noqa: F401  (kept available for future per-request timing)
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg2.extras import RealDictCursor

# slowapi: the rate-limit handler lives at the *top-level* ``slowapi``
# package, not in ``slowapi.errors``. The errors module only exposes the
# exception class itself (``RateLimitExceeded``). Importing
# ``_rate_limit_exceeded_handler`` from ``slowapi.errors`` would raise
# ImportError and prevent the app from starting — that is a discrepancy
# between the build spec and the installed slowapi 0.1.9, resolved here
# in favour of the path that actually exists.
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from forge.benchmark.loader import BenchmarkLoader
from forge.capture.trajectory import Trajectory
from forge.metrics.config import WeightingConfig
from forge.metrics.engine import MetricEngine
from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric
from forge.server.db import Database
from forge.server.models import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkStatusResponse,
    EvaluateRequest,
    EvaluateResponse,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("forge.api")


# ---------------------------------------------------------------- rate limiter

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


# ---------------------------------------------------------------- redis client

# Bounded socket timeouts keep ``GET /health`` responsive when ``REDIS_URL``
# points at the Docker Compose hostname (``redis://redis:6379``) but the API
# is started on the host via ``uvicorn`` — the default blocking connect can
# take ~10 s per probe and blow past httpx's verify-script budget.
_DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT = 2.0
_DEFAULT_REDIS_SOCKET_TIMEOUT = 2.0
# Wall-clock cap for the health probe (connect + PING), including thread
# scheduling overhead. Slightly above the socket timeouts so a healthy
# Docker deployment still passes when Redis is a few ms away.
_DEFAULT_REDIS_HEALTH_PROBE_TIMEOUT = 3.0


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r — using default %.1f", name, raw, default
        )
        return default


def _create_redis_client() -> redis.Redis:
    """Build a :class:`redis.Redis` client with bounded socket timeouts."""
    connect_timeout = _float_env(
        "REDIS_SOCKET_CONNECT_TIMEOUT", _DEFAULT_REDIS_SOCKET_CONNECT_TIMEOUT
    )
    socket_timeout = _float_env(
        "REDIS_SOCKET_TIMEOUT", _DEFAULT_REDIS_SOCKET_TIMEOUT
    )
    return redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
        socket_connect_timeout=connect_timeout,
        socket_timeout=socket_timeout,
    )


async def _probe_redis(client: redis.Redis) -> bool:
    """Return whether Redis answers ``PING`` within a bounded wall-clock budget.

    ``redis.ping()`` is synchronous and would block the ASGI event loop for
    the full TCP/DNS timeout; run it in a worker thread and enforce
    :envvar:`REDIS_HEALTH_PROBE_TIMEOUT` on top of the per-socket limits.
    """
    probe_timeout = _float_env(
        "REDIS_HEALTH_PROBE_TIMEOUT", _DEFAULT_REDIS_HEALTH_PROBE_TIMEOUT
    )
    try:
        await asyncio.wait_for(
            asyncio.to_thread(client.ping),
            timeout=probe_timeout,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "health: redis probe timed out after %.1fs", probe_timeout
        )
        return False
    except Exception as exc:
        logger.warning("health: redis probe raised: %s", exc)
        return False


# ----------------------------------------------------------------- body size

# 10 MB hard cap. Sized for trajectories with verbose tool outputs while
# still bounding the worst-case allocation for a single request.
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024


async def _enforce_max_body_size(request: Request, call_next):
    """ASGI middleware that rejects oversized requests via ``Content-Length``.

    Returns ``413 Request Entity Too Large`` when the declared
    ``Content-Length`` exceeds :data:`_MAX_REQUEST_BODY_BYTES`. Requests
    without a ``Content-Length`` header (chunked transfer, streaming
    upload) pass through unchecked here — uvicorn enforces its own
    protocol-level limits, and the application layer can still cap them
    per-endpoint if needed.

    The check happens before the request body is read, so an oversized
    payload is rejected without ever being buffered into memory.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError:
            length = 0
        if length > _MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body too large: {length} bytes exceeds "
                        f"the {_MAX_REQUEST_BODY_BYTES}-byte limit"
                    )
                },
            )
    return await call_next(request)


# -------------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage long-lived resources for the API process.

    Startup tries to bring up PostgreSQL (via :class:`Database`) and
    Redis (via :func:`redis.from_url`). Each component is attempted
    independently and a failure to instantiate is logged and recorded
    as ``None`` on ``app.state``; the application still starts so
    ``/health`` can report the degraded condition. The verify script
    relies on this — it accepts both ``200`` (all components up) and
    ``503`` (degraded) as valid outcomes.

    Shutdown closes whichever resources were successfully constructed.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("forge.api startup begin at %s", started_at)

    try:
        app.state.db = Database()
        logger.info("PostgreSQL connection pool initialised")
    except Exception as exc:
        app.state.db = None
        logger.warning("PostgreSQL unavailable at startup: %s", exc)

    try:
        app.state.redis = _create_redis_client()
        logger.info("Redis client created (lazy connection)")
    except Exception as exc:
        app.state.redis = None
        logger.warning("Redis client could not be created at startup: %s", exc)

    logger.info("forge.api startup complete at %s", started_at)

    try:
        yield
    finally:
        shutdown_at = datetime.now(timezone.utc).isoformat()
        logger.info("forge.api shutdown begin at %s", shutdown_at)

        if getattr(app.state, "db", None) is not None:
            try:
                app.state.db.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning("Database.close raised: %s", exc)

        if getattr(app.state, "redis", None) is not None:
            try:
                app.state.redis.close()
                logger.info("Redis client closed")
            except Exception as exc:
                logger.warning("redis.close raised: %s", exc)

        logger.info("forge.api shutdown complete at %s", shutdown_at)


# -------------------------------------------------------------------- app build

app = FastAPI(
    title="Forge Evaluation API",
    version="0.1.0",
    description="Open evaluation framework for multi-step LLM agents",
    lifespan=lifespan,
)

# Rate-limit wiring. ``limiter`` on ``app.state`` is the contract
# SlowAPIMiddleware reads at request time; the exception handler
# converts ``RateLimitExceeded`` into a 429 with a structured body.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Body-size middleware is added *before* SlowAPIMiddleware so an
# oversized request is short-circuited with 413 before being counted
# against the per-IP rate-limit budget.
app.middleware("http")(_enforce_max_body_size)
app.add_middleware(SlowAPIMiddleware)


# ----------------------------------------------------------------- dependencies


def get_db(request: Request):
    """Return the pooled :class:`Database` (or ``None`` if init failed)."""
    return request.app.state.db


def get_redis(request: Request):
    """Return the :class:`redis.Redis` client (or ``None`` if init failed)."""
    return request.app.state.redis


# --------------------------------------------------------------------- endpoints


@app.get("/")
async def root() -> dict:
    """Service metadata. Cheap, always-200; useful as a liveness probe."""
    return {"name": "Forge", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
async def health(request: Request):
    """Deep health check across PostgreSQL, Redis, and the embedding cache.

    Each subsystem is probed independently — a failure in one does not
    short-circuit the others, so the response always carries the full
    component map. Status is 200 when every probe is healthy, 503 when
    any probe reports down.

    The embedding-model probe is by design *informational*: a freshly
    started API process has not yet loaded the BGE model (it loads
    lazily on the first ``ReasoningCoherenceMetric`` instantiation),
    so the cache will report ``False`` until the first ``/evaluate``
    request that runs the coherence metric. That is not a degraded
    state — it is the expected initial state — and the spec asks the
    health endpoint to surface it explicitly.
    """
    db = request.app.state.db
    r = request.app.state.redis

    if db is not None:
        try:
            postgres_ok = bool(db.health_check())
        except Exception as exc:
            logger.warning("health: postgresql probe raised: %s", exc)
            postgres_ok = False
    else:
        postgres_ok = False

    if r is not None:
        redis_ok = await _probe_redis(r)
    else:
        redis_ok = False

    embedding_model_loaded = bool(ReasoningCoherenceMetric._model_cache)

    components = {
        "postgresql": postgres_ok,
        "redis": redis_ok,
        "embedding_model": embedding_model_loaded,
    }

    # Embedding model not-yet-loaded is the expected initial state and
    # must not flip the overall status to degraded. Only PostgreSQL and
    # Redis count toward the up/down decision.
    operational_ok = postgres_ok and redis_ok
    status = "healthy" if operational_ok else "degraded"

    payload = {
        "status": status,
        "components": components,
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not embedding_model_loaded:
        payload["notes"] = {
            "embedding_model": "not yet loaded, will load on first evaluation"
        }

    return JSONResponse(
        status_code=200 if operational_ok else 503,
        content=payload,
    )


# ---------------------------------------------------------------- /evaluate

# Stricter per-IP cap than the global default to discourage abuse of the
# evaluation path, which is the only endpoint today that can drive
# expensive LLM-as-judge calls and embedding-model warm-up.
@app.post("/evaluate", response_model=EvaluateResponse)
@limiter.limit("30/minute")
async def evaluate(
    request: Request,
    body: EvaluateRequest,
    db: Database = Depends(get_db),
) -> EvaluateResponse:
    """Score a submitted trajectory with the full Forge metric suite.

    Pipeline:
      1. Validate the trajectory payload (422 on schema / semantic issues).
      2. Select a :class:`WeightingConfig` from ``body.weighting_strategy``.
      3. Run :meth:`MetricEngine.run_all_with_explanations`.
      4. Split the rich per-metric results into ``scores`` (floats) and
         ``explanations`` (strings); synthesise ``_has_failures`` and
         ``_metric_errors`` from each metric's ``had_error`` /
         ``error_message`` channel (these keys are intentionally absent
         from ``run_all_with_explanations`` output — only ``run_all``
         emits them — so the API layer rebuilds them here).
      5. Build an immutable ``evaluation_config`` provenance dict.
      6. Best-effort persist trajectory + evaluation. Database failures
         do not fail the request: ``trajectory_id`` and/or
         ``evaluation_id`` come back as ``"unsaved"`` and the caller
         still receives the scores.

    Notes:
      * The first request after process start may be slow because
        :class:`ReasoningCoherenceMetric` lazily downloads / loads its
        embedding model on first instantiation. Subsequent requests
        reuse the class-level cache and are fast. This is expected and
        not a degraded state.
      * ``HTTPException`` raised by the validation step is propagated
        unchanged; only unexpected exceptions are converted to 500 so
        422 validation failures remain 422 to the client.
      * ``body.metrics``, when non-empty, narrows the metric set to the
        requested subset (matches :class:`MetricEngine`'s ``metric_names``
        contract). Empty list means "run every registered metric".
    """
    try:
        try:
            trajectory_model = Trajectory.from_dict(body.trajectory)
            trajectory_model.validate()
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid trajectory schema: {exc}",
            )

        if body.weighting_strategy == "research":
            weighting_config = WeightingConfig.research_weights()
        else:
            weighting_config = WeightingConfig.equal_weights()

        engine = MetricEngine(
            metric_names=body.metrics if body.metrics else None,
            weighting_config=weighting_config,
        )

        rich_results = engine.run_all_with_explanations(body.trajectory)

        composite_score = float(rich_results.get("composite_score", 0.0))
        scores: dict[str, float] = {}
        explanations: dict[str, str] = {}
        metric_errors: dict[str, str] = {}
        for name, entry in rich_results.items():
            if name == "composite_score":
                continue
            if not isinstance(entry, dict):
                continue
            scores[name] = float(entry.get("score", 0.0))
            explanations[name] = str(entry.get("explanation", ""))
            if entry.get("had_error"):
                metric_errors[name] = (
                    entry.get("error_message") or "unknown error"
                )
        has_failures = bool(metric_errors)

        provider = os.getenv("LLM_PROVIDER", "groq")
        judge_model = (
            "llama-3.3-70b-versatile" if provider == "groq" else "phi3:mini"
        )
        evaluation_config = {
            "metric_names": engine.available_metrics(),
            "weighting_strategy": body.weighting_strategy,
            "weights": weighting_config.normalized_weights(),
            "judge_provider": provider,
            "judge_model": judge_model,
            "forge_version": "0.1.0",
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        if db is not None:
            try:
                trajectory_id = db.save_trajectory(
                    body.trajectory,
                    agent_id=body.trajectory.get("agent_id", "api_submission"),
                )
            except Exception as exc:
                logger.warning("evaluate: save_trajectory failed: %s", exc)
                trajectory_id = "unsaved"
        else:
            logger.warning(
                "evaluate: db unavailable at startup; trajectory not persisted"
            )
            trajectory_id = "unsaved"

        if db is not None and trajectory_id != "unsaved":
            try:
                evaluation_id = db.save_evaluation_with_config(
                    trajectory_id, scores, evaluation_config
                )
            except Exception as exc:
                logger.warning(
                    "evaluate: save_evaluation_with_config failed: %s", exc
                )
                evaluation_id = "unsaved"
        else:
            evaluation_id = "unsaved"

        return EvaluateResponse(
            evaluation_id=evaluation_id,
            trajectory_id=trajectory_id,
            scores=scores,
            explanations=explanations if body.include_explanations else {},
            has_failures=has_failures,
            metric_errors=metric_errors,
            composite_score=composite_score,
            evaluation_config=evaluation_config,
        )

    except HTTPException:
        # Preserve client-visible HTTP errors (notably the 422 above);
        # converting them to 500 would mask validation failures behind a
        # generic server-error response.
        raise
    except Exception:
        logger.exception("evaluate: unexpected failure")
        raise HTTPException(status_code=500, detail="Evaluation failed")


# ---------------------------------------------------------- /benchmark/run

# Redis key namespace + TTLs for transient benchmark job status.
# Queued/running entries live for an hour (long enough for any sane
# benchmark slice); on completion or failure we shorten the TTL to 5 min
# so the entry expires after a brief grace window for late pollers,
# at which point GET /benchmark/{job_id} falls back to PostgreSQL — the
# permanent source of truth.
_JOB_KEY_PREFIX = "forge:job:"
_REDIS_TTL_ACTIVE_SECONDS = 60 * 60
_REDIS_TTL_TERMINAL_SECONDS = 5 * 60


def _job_key(job_id: str) -> str:
    return f"{_JOB_KEY_PREFIX}{job_id}"


def _set_job_status(r, job_id: str, payload: dict, ttl_seconds: int) -> None:
    """Best-effort write of the transient job-status entry to Redis.

    Failures are logged and swallowed so a Redis outage cannot abort an
    otherwise-healthy benchmark run; PostgreSQL remains authoritative.
    """
    try:
        r.set(_job_key(job_id), json.dumps(payload), ex=ttl_seconds)
    except Exception as exc:
        logger.warning(
            "benchmark: redis set for job %s failed (%s) — "
            "PostgreSQL is still authoritative", job_id, exc
        )


def _known_benchmark_domains() -> set[str]:
    """Set of valid domain names for ``body.benchmark`` (besides ``"all"``).

    Read directly from the benchmark data directory so adding a new
    domain doesn't require code changes here. Returns an empty set on
    any I/O error so the caller can decide whether to fail-closed.
    """
    try:
        data_dir = BenchmarkLoader().data_dir
        return {p.name for p in data_dir.iterdir() if p.is_dir()}
    except Exception as exc:
        logger.warning("benchmark: could not enumerate domains: %s", exc)
        return set()


def _fetch_benchmark_run_row(db: Database, job_id: str) -> Optional[dict]:
    """Read one ``benchmark_runs`` row by id.

    Lives in ``main.py`` because the scope of this change is "modify
    exactly one existing file"; the natural home is a
    ``Database.get_benchmark_run(job_id)`` method, which is the
    recommended follow-up. Uses the pooled connection via
    ``db._get_conn()`` so the read still rides the same pool as every
    other DB call.

    Returns ``None`` when no row matches; surfaces an empty result for
    ``GET /benchmark/{job_id}`` to convert into a 404. A malformed
    ``job_id`` (not a valid UUID) raises ``psycopg2.DataError`` which
    the endpoint converts to 404 as well — querying a non-UUID has no
    sensible "found" semantics.
    """
    conn = db._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM benchmark_runs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
        conn.rollback()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
        raise
    db._put_conn(conn)
    return dict(row) if row is not None else None


async def _run_benchmark_job(
    job_id: str,
    body: BenchmarkRunRequest,
    r,
    db: Database,
) -> None:
    """Background task: run the benchmark, update Redis + PostgreSQL.

    Runs the synchronous :meth:`BenchmarkRunner.run` inside
    ``asyncio.to_thread`` so the FastAPI event loop is not blocked by
    benchmark execution (which can issue many seconds of LLM and DB
    calls). Both Redis and PostgreSQL receive a ``running`` status
    update at the start; on success both are marked ``complete`` with
    ``avg_scores`` populated; on any exception both are marked
    ``failed`` with the exception message.

    The PostgreSQL update is the durable record; Redis is a cache for
    sub-second status polling that expires after
    :data:`_REDIS_TTL_TERMINAL_SECONDS` once the job terminates.
    """
    _set_job_status(
        r,
        job_id,
        {
            "status": "running",
            "job_id": job_id,
            "total_tasks": None,
            "completed_tasks": 0,
            "aggregate_scores": None,
            "error": None,
        },
        _REDIS_TTL_ACTIVE_SECONDS,
    )
    try:
        db.update_benchmark_run(job_id, {"job_status": "running"})
    except Exception as exc:
        logger.warning(
            "benchmark: PG running-status update for job %s failed: %s",
            job_id, exc,
        )

    def _mock_agent(task: str) -> str:
        return f"Mock answer for benchmark task: {task[:80]}"

    try:
        # Lazy import: forge.benchmark.runner pulls in langchain (~9s
        # import cost on this stack). Keeping it out of module-top
        # imports preserves fast API startup for /health, /, and
        # /evaluate, which don't need the runner. The first benchmark
        # request pays the load once per process; subsequent requests
        # reuse Python's import cache instantly.
        from forge.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(
            agent_fn=_mock_agent,
            db=db,
            agent_id=body.agent_id,
        )
        # BenchmarkRunner.run is CPU/IO blocking (loads files, calls metrics,
        # may invoke LLM judges, writes to PG). Push it off the event loop.
        domain = None if body.benchmark == "all" else body.benchmark
        results = await asyncio.to_thread(
            runner.run, domain=domain, max_tasks=body.max_tasks,
        )

        aggregate_scores = results.get("aggregate_scores") or {}
        total_tasks = int(results.get("total_tasks", 0))
        completed_tasks = int(results.get("completed_tasks", 0))

        try:
            # Pass the dict directly. Database.update_benchmark_run wraps
            # avg_scores in psycopg2.extras.Json() on its own; json.dumps()
            # here would store a stringified JSON inside a JSONB column.
            # finished_at is TIMESTAMPTZ — pass a real UTC datetime so
            # psycopg2 adapts it to a SQL timestamp. The string "NOW()"
            # would be treated as a literal value and rejected.
            db.update_benchmark_run(
                job_id,
                {
                    "job_status": "complete",
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "avg_scores": aggregate_scores,
                    "finished_at": datetime.now(timezone.utc),
                    "agent_type": body.agent_type,
                },
            )
        except Exception as exc:
            logger.error(
                "benchmark: PG complete-status update for job %s failed: %s",
                job_id, exc,
            )

        _set_job_status(
            r,
            job_id,
            {
                "status": "complete",
                "job_id": job_id,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "aggregate_scores": aggregate_scores,
                "error": None,
            },
            _REDIS_TTL_TERMINAL_SECONDS,
        )

    except Exception as exc:
        logger.exception("benchmark: job %s failed", job_id)
        try:
            db.update_benchmark_run(
                job_id,
                {
                    "job_status": "failed",
                    "finished_at": datetime.now(timezone.utc),
                    "agent_type": body.agent_type,
                },
            )
        except Exception as inner_exc:
            logger.error(
                "benchmark: PG failed-status update for job %s failed: %s",
                job_id, inner_exc,
            )
        _set_job_status(
            r,
            job_id,
            {
                "status": "failed",
                "job_id": job_id,
                "total_tasks": None,
                "completed_tasks": 0,
                "aggregate_scores": None,
                "error": str(exc),
            },
            _REDIS_TTL_TERMINAL_SECONDS,
        )


# Stricter cap than the global default — benchmarks are the most
# expensive endpoint (every accepted request schedules a multi-task
# background run). 5/min/IP still leaves plenty of headroom for an
# interactive UI driving the API.
@app.post("/benchmark/run", response_model=BenchmarkRunResponse)
@limiter.limit("5/minute")
async def run_benchmark(
    request: Request,
    body: BenchmarkRunRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
    r=Depends(get_redis),
) -> BenchmarkRunResponse:
    """Queue a benchmark run and return a job_id for status polling.

    Synchronous validation is performed before the job is queued:

    * ``body.benchmark`` must be ``"all"`` or a known domain
      subdirectory under ``data/benchmark/`` — 422 otherwise. Failing
      late (inside the background task) would make the only signal a
      ``failed`` status on the eventual GET, which is much harder for
      clients to react to.
    * ``body.max_tasks``, if set, must be a positive integer — 422
      otherwise.

    Operational dependencies:

    * PostgreSQL is the source of truth for benchmark runs. If
      ``Database`` failed to initialise at startup the endpoint
      returns 503 — the run cannot be queued because the permanent
      record cannot be created.
    * Redis holds the transient sub-second status entry. If Redis is
      unavailable the run is still queued; ``GET /benchmark/{job_id}``
      will fall back to the PostgreSQL record for status.
    """
    try:
        if body.benchmark != "all":
            known = _known_benchmark_domains()
            if known and body.benchmark not in known:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unknown benchmark {body.benchmark!r}; "
                        f"expected 'all' or one of {sorted(known)}"
                    ),
                )
        if body.max_tasks is not None and body.max_tasks <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"max_tasks must be a positive integer when set; "
                    f"got {body.max_tasks!r}"
                ),
            )

        if db is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PostgreSQL unavailable; benchmark runs require the "
                    "permanent record store and cannot be queued"
                ),
            )

        job_id = str(uuid.uuid4())

        try:
            db.save_benchmark_run(
                job_id, body.agent_id, body.benchmark, body.agent_type,
            )
        except Exception as exc:
            logger.exception("benchmark: save_benchmark_run failed")
            raise HTTPException(
                status_code=503,
                detail=f"Could not persist benchmark run: {exc}",
            )

        initial_payload = {
            "status": "queued",
            "job_id": job_id,
            "total_tasks": None,
            "completed_tasks": 0,
            "aggregate_scores": None,
            "error": None,
        }
        if r is not None:
            _set_job_status(r, job_id, initial_payload, _REDIS_TTL_ACTIVE_SECONDS)
        else:
            logger.warning(
                "benchmark: redis unavailable; job %s will only be "
                "observable via PostgreSQL", job_id
            )

        background_tasks.add_task(_run_benchmark_job, job_id, body, r, db)

        return BenchmarkRunResponse(
            job_id=job_id,
            status="queued",
            benchmark=body.benchmark,
            agent_id=body.agent_id,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("benchmark/run: unexpected failure")
        raise HTTPException(status_code=500, detail="Benchmark queue failed")


@app.get("/benchmark/{job_id}", response_model=BenchmarkStatusResponse)
async def get_benchmark_status(
    job_id: str,
    r=Depends(get_redis),
    db: Database = Depends(get_db),
) -> BenchmarkStatusResponse:
    """Return current status of a benchmark job.

    Source-of-truth split:

    * Redis is consulted first; if the ``queued`` or ``running`` entry
      is present, return it (sub-second transient state).
    * Otherwise (Redis miss, Redis unavailable, or Redis entry is a
      terminal status that may have aged out), fall back to the
      PostgreSQL ``benchmark_runs`` row — the permanent record.
    * If neither source knows the job, 404.

    The fallback ordering matters: a *completed* job's Redis entry
    expires after :data:`_REDIS_TTL_TERMINAL_SECONDS` (5 min), but the
    PostgreSQL row is durable, so a poller arriving after that window
    still gets a useful response.
    """
    if r is not None:
        try:
            raw = r.get(_job_key(job_id))
        except Exception as exc:
            logger.warning(
                "benchmark: redis get for job %s failed: %s", job_id, exc
            )
            raw = None
        if raw is not None:
            try:
                cached = json.loads(raw)
            except Exception:
                cached = None
            if cached is not None and cached.get("status") in (
                "queued", "running",
            ):
                return BenchmarkStatusResponse(
                    job_id=cached.get("job_id", job_id),
                    status=cached["status"],
                    total_tasks=cached.get("total_tasks"),
                    completed_tasks=cached.get("completed_tasks"),
                    aggregate_scores=cached.get("aggregate_scores"),
                    error=cached.get("error"),
                )

    if db is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        row = _fetch_benchmark_run_row(db, job_id)
    except Exception as exc:
        # Malformed UUID and similar DB-level rejects fall through here.
        # The job cannot be looked up either way — surface as 404 so the
        # client gets a single, consistent "doesn't exist" signal.
        logger.warning(
            "benchmark: PG lookup for job %s failed: %s", job_id, exc
        )
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return BenchmarkStatusResponse(
        job_id=str(row.get("id", job_id)),
        status=row.get("job_status") or "unknown",
        total_tasks=row.get("total_tasks"),
        completed_tasks=row.get("completed_tasks"),
        aggregate_scores=row.get("avg_scores"),
        error=None,
    )
