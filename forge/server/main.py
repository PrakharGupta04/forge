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
import json  # noqa: F401  (kept available for future JSON-bodied endpoints)
import logging
import os
import time  # noqa: F401  (kept available for future per-request timing)
import uuid  # noqa: F401  (kept available for future evaluation_id generation)
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: F401
from fastapi.responses import JSONResponse

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

from forge.metrics.reasoning_coherence import ReasoningCoherenceMetric
from forge.server.db import Database


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
