"""Storage layer for Forge — pooled, thread-safe, environment-configurable.

Uses :class:`psycopg2.pool.ThreadedConnectionPool` so the upcoming
FastAPI workers (each on its own thread) can share a bounded set of
PostgreSQL connections instead of serialising through a single global
connection. Pool sizing is read from the environment::

    DB_POOL_MIN  (default 2)   — connections opened eagerly at startup
    DB_POOL_MAX  (default 10)  — hard cap; ``getconn()`` blocks above this

Per-method lifecycle:

* ``conn = self._get_conn()`` borrows a connection from the pool.
* Every write commits on success / rolls back on exception inside the
  ``try`` block.
* Every read rolls back before returning the connection (autocommit is
  off, so even ``SELECT`` opens an implicit transaction that must be
  closed before ``putconn`` — otherwise the next borrower inherits the
  half-open transaction).
* ``finally: self._put_conn(conn)`` always returns the connection to
  the pool, including on the success path.

Every cursor is opened with ``cursor_factory=RealDictCursor`` so rows
come back as dict-like ``RealDictRow`` objects. The legacy public API
(``save_trajectory``, ``load_trajectory``, ``save_evaluation``,
``save_evaluation_with_config``) returns exactly the same values it
returned before this refactor — only the internal cursor format
changed.

The implementation deliberately mirrors the current ``schema.sql`` plus
``migrations/001_add_evaluation_provenance.sql``:

* ``save_trajectory`` writes to the columns the schema actually has
  (``agent_id``, ``task``, ``final_answer``, ``ground_truth``,
  ``raw_trajectory``, ``total_steps``, ``total_tokens``, ``duration_ms``).
* ``save_evaluation`` writes the 7 per-metric columns plus
  ``composite_score``. Metric scores absent from the input dict are
  persisted as ``NULL``.
* ``save_evaluation_with_config`` additionally writes
  ``evaluation_config`` (JSONB, post-migration-001).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


_EVALUATION_METRIC_COLUMNS: tuple[str, ...] = (
    "task_completion",
    "tool_call_fidelity",
    "reasoning_coherence",
    "hallucination_score",
    "step_efficiency",
    "recovery_rate",
    "consistency",
)


# Whitelist of benchmark_runs columns that ``update_benchmark_run`` is
# allowed to mutate. SQL column names are formatted directly into the
# UPDATE statement, so the allowlist is the security boundary that
# prevents callers from injecting arbitrary SQL via dict keys.
_BENCHMARK_RUN_UPDATABLE: frozenset[str] = frozenset({
    "job_status",
    "completed_tasks",
    "total_tasks",
    "avg_scores",
    "finished_at",
    "agent_type",
})


class Database:
    """Pooled psycopg2 wrapper over the Forge tables."""

    def __init__(self) -> None:
        load_dotenv()
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to .env (e.g. "
                "postgresql://forge:forge_dev@localhost:5432/forge) before "
                "constructing Database()."
            )

        min_conn = int(os.getenv("DB_POOL_MIN", "2"))
        max_conn = int(os.getenv("DB_POOL_MAX", "10"))
        if min_conn < 1:
            raise ValueError(
                f"DB_POOL_MIN must be >= 1, got {min_conn}"
            )
        if max_conn < min_conn:
            raise ValueError(
                f"DB_POOL_MAX ({max_conn}) must be >= DB_POOL_MIN ({min_conn})"
            )

        self._pool = ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            dsn=database_url,
        )

    # ------------------------------------------------------------ pool plumbing

    def _get_conn(self):
        """Borrow a connection from the pool.

        Blocks if the pool has reached ``DB_POOL_MAX`` and every
        connection is in use; this is the standard ThreadedConnectionPool
        behaviour and bounds how aggressively callers can saturate the
        database.
        """
        return self._pool.getconn()

    def _put_conn(self, conn, close: bool = False) -> None:
        """Return ``conn`` to the pool (or close it if ``close`` is True).

        Callers MUST commit or rollback first; the pool does not perform
        an automatic rollback, and an uncommitted transaction would leak
        to the next borrower of this connection.
        """
        try:
            self._pool.putconn(conn, close=close)
        except Exception as exc:
            logger.warning("Database._put_conn failed: %s", exc)

    # ------------------------------------------------------------------- reads

    def health_check(self) -> bool:
        """Run ``SELECT 1`` and report success/failure as a bool."""
        try:
            conn = self._get_conn()
        except Exception as exc:
            logger.error("Database.health_check could not borrow conn: %s", exc)
            return False
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.rollback()
            return True
        except Exception as exc:
            logger.error("Database.health_check failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            self._put_conn(conn)

    def load_trajectory(self, trajectory_id: str) -> dict:
        """Return the ``raw_trajectory`` JSONB for ``trajectory_id`` as a dict.

        Return shape is unchanged: a plain dict that is the deserialised
        contents of the ``raw_trajectory`` JSONB column (the same dict
        that ``save_trajectory`` originally wrote). Callers can index
        ``loaded["trajectory_id"]``, ``loaded["task"]``,
        ``loaded["steps"]``, etc.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT raw_trajectory FROM trajectories WHERE id = %s",
                    (trajectory_id,),
                )
                row = cur.fetchone()
            conn.rollback()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            self._put_conn(conn)
            raise

        self._put_conn(conn)

        if row is None:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        return row["raw_trajectory"]

    def get_trajectory_with_evaluation(self, trajectory_id: str) -> dict:
        """Return ``raw_trajectory`` merged with that trajectory's evaluation.

        Performs a ``LEFT JOIN`` so trajectories without any evaluation
        row still come back as the bare trajectory dict (no metric
        scores attached, no evaluation provenance). When an evaluation
        does exist, its non-NULL metric scores and ``composite_score``
        are layered on top of the trajectory dict as flat top-level
        keys.

        Field-name collisions: the trajectory dict's top-level fields
        (``task``, ``trajectory_id``, ``agent_id``, ``final_answer``,
        ``steps``, ``metadata``, ``conversation_history``,
        ``retrieved_context``, ``total_duration_ms``, ``total_tokens``,
        ``error_summary``, ``timestamp``, ``ground_truth``) do not
        collide with any metric column name, so the merge is
        non-destructive in practice. The merge is right-biased on
        purpose: if a future schema change introduced a collision, the
        authoritative scored value (from ``evaluations``) wins over
        whatever happens to live in the JSONB blob.

        Raises:
            ValueError: if no trajectory exists with ``id = trajectory_id``.
        """
        eval_cols = list(_EVALUATION_METRIC_COLUMNS) + ["composite_score"]
        # Prefix the eval columns in the SELECT list so they cannot shadow
        # any future ``trajectories`` column with the same name (e.g.
        # ``id``). We alias each prefixed column back to the bare metric
        # name when merging, below.
        eval_select = ", ".join(f"e.{c} AS eval_{c}" for c in eval_cols)

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT t.raw_trajectory AS raw_trajectory,
                           {eval_select},
                           e.evaluated_at AS eval_evaluated_at
                    FROM trajectories t
                    LEFT JOIN evaluations e ON e.trajectory_id = t.id
                    WHERE t.id = %s
                    """,
                    (trajectory_id,),
                )
                row = cur.fetchone()
            conn.rollback()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            self._put_conn(conn)
            raise

        self._put_conn(conn)

        if row is None:
            raise ValueError(f"Trajectory {trajectory_id} not found")

        raw = row.get("raw_trajectory")
        merged: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

        for col in eval_cols:
            value = row.get(f"eval_{col}")
            if value is not None:
                merged[col] = value

        evaluated_at = row.get("eval_evaluated_at")
        if evaluated_at is not None:
            merged["evaluated_at"] = (
                evaluated_at.isoformat()
                if hasattr(evaluated_at, "isoformat")
                else evaluated_at
            )

        return merged

    def get_leaderboard(self, benchmark_name: Optional[str] = None) -> list:
        """Return completed real-agent benchmark runs ordered by composite.

        Filters out ``agent_type = 'mock'`` so leaderboards aren't
        polluted by sanity-check / fixture runs. ``agent_type IS NULL``
        is also excluded by ordinary SQL NULL semantics — after
        migration 001 every row has a non-NULL default of ``'unknown'``,
        but if a pre-migration row somehow survives without backfill it
        will not appear on the leaderboard.

        Composite-score sort: ``(avg_scores->>'composite_score')::float
        DESC NULLS LAST`` — rows whose ``avg_scores`` is NULL, or whose
        ``avg_scores`` JSON happens to lack a ``composite_score`` key,
        sort to the bottom rather than failing the query. Rows whose
        ``avg_scores->>'composite_score'`` is present but un-castable
        to float will raise; that is a data integrity problem the
        caller should learn about, not silently swallow.

        Args:
            benchmark_name: Optional benchmark filter. If omitted,
                leaderboard spans every benchmark.

        Returns:
            A list of plain dicts (one per matching ``benchmark_runs``
            row), highest composite first. Empty list when no rows
            match.
        """
        params: tuple = ()
        sql = """
            SELECT *
            FROM benchmark_runs
            WHERE job_status = 'complete' AND agent_type != 'mock'
        """
        if benchmark_name is not None:
            sql += " AND benchmark_name = %s"
            params = (benchmark_name,)
        sql += (
            " ORDER BY (avg_scores->>'composite_score')::float DESC NULLS LAST"
        )

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.rollback()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            self._put_conn(conn)
            raise

        self._put_conn(conn)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ writes

    def save_trajectory(self, trajectory_dict: dict, agent_id: str) -> str:
        """Insert one row into ``trajectories``; return the generated UUID."""
        total_steps = trajectory_dict.get("total_steps")
        if total_steps is None:
            total_steps = len(trajectory_dict.get("steps", []))

        duration_ms = trajectory_dict.get("duration_ms")
        if duration_ms is None:
            duration_ms = trajectory_dict.get("total_duration_ms")

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO trajectories (
                        agent_id, task, final_answer, ground_truth,
                        raw_trajectory, total_steps, total_tokens, duration_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        agent_id,
                        trajectory_dict.get("task"),
                        trajectory_dict.get("final_answer"),
                        trajectory_dict.get("ground_truth"),
                        Json(trajectory_dict),
                        total_steps,
                        trajectory_dict.get("total_tokens"),
                        duration_ms,
                    ),
                )
                new_id = cur.fetchone()["id"]
            conn.commit()
            return str(new_id)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._put_conn(conn)

    def save_evaluation(self, trajectory_id: str, scores: dict) -> str:
        """Insert one row into ``evaluations`` and return the generated UUID.

        ``scores`` is the dict returned by ``MetricEngine.run_all``. Only the
        7 per-metric columns and ``composite_score`` are persisted; meta keys
        such as ``_metric_errors`` and ``_has_failures`` are ignored at the
        storage layer (they are run-time diagnostics, not durable state).
        Missing metric scores are persisted as ``NULL``.
        """
        columns = list(_EVALUATION_METRIC_COLUMNS) + ["composite_score"]
        values: list[Optional[float]] = [scores.get(col) for col in columns]

        col_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    INSERT INTO evaluations (trajectory_id, {col_list})
                    VALUES (%s, {placeholders})
                    RETURNING id
                    """,
                    (trajectory_id, *values),
                )
                new_id = cur.fetchone()["id"]
            conn.commit()
            return str(new_id)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._put_conn(conn)

    def save_evaluation_with_config(
        self,
        trajectory_id: str,
        scores: dict,
        evaluation_config: dict,
    ) -> str:
        """Insert one row into ``evaluations`` including provenance config.

        Identical to :meth:`save_evaluation` except it also writes the
        ``evaluation_config`` JSONB column added by migration 001
        (``migrations/001_add_evaluation_provenance.sql``). The
        ``evaluation_config`` dict is expected to carry per-evaluation
        provenance — ``metric_names`` (list), ``weighting_strategy``
        ("equal" or "custom"), ``weights`` (dict),
        ``judge_model`` (str), ``judge_provider`` ("groq" or "ollama"),
        ``forge_version`` (str), ``evaluated_at`` (ISO timestamp str) —
        but the storage layer does not validate the shape; it is stored
        verbatim as JSONB so it can evolve without further migrations.

        Requires that migration 001 has been applied. On a fresh schema
        (pre-migration) this method will fail with a ``psycopg2`` error
        about the missing ``evaluation_config`` column — use
        :meth:`save_evaluation` against an un-migrated database.
        """
        columns = list(_EVALUATION_METRIC_COLUMNS) + ["composite_score"]
        values: list[Optional[float]] = [scores.get(col) for col in columns]

        col_list = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    INSERT INTO evaluations (
                        trajectory_id, {col_list}, evaluation_config
                    )
                    VALUES (%s, {placeholders}, %s)
                    RETURNING id
                    """,
                    (trajectory_id, *values, Json(evaluation_config)),
                )
                new_id = cur.fetchone()["id"]
            conn.commit()
            return str(new_id)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._put_conn(conn)

    def save_benchmark_run(
        self,
        job_id: str,
        agent_id: str,
        benchmark_name: str,
        agent_type: str = "unknown",
    ) -> None:
        """Insert a new ``benchmark_runs`` row in ``queued`` state.

        ``job_id`` is supplied by the caller (rather than letting
        Postgres generate it via ``gen_random_uuid()``) so the API can
        echo it back to the client synchronously before the actual run
        starts. Must be a valid UUID string; Postgres will reject any
        other format.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO benchmark_runs (
                        id, agent_id, benchmark_name, agent_type, job_status
                    )
                    VALUES (%s, %s, %s, %s, 'queued')
                    """,
                    (job_id, agent_id, benchmark_name, agent_type),
                )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._put_conn(conn)

    def update_benchmark_run(self, job_id: str, updates: dict) -> None:
        """Patch a ``benchmark_runs`` row by id.

        Only columns listed in :data:`_BENCHMARK_RUN_UPDATABLE` are
        allowed; unknown keys in ``updates`` are silently dropped. This
        is the security boundary that lets us format the column name
        directly into the SQL — values still flow through
        parameterised placeholders. Dict values bound for ``avg_scores``
        are wrapped in :class:`psycopg2.extras.Json` so they round-trip
        as JSONB; other columns pass through as-is.

        A no-op (empty ``updates`` or only unknown keys) returns silently
        without issuing SQL, including without consuming a connection
        from the pool for an INSERT.
        """
        filtered = {
            col: val for col, val in updates.items()
            if col in _BENCHMARK_RUN_UPDATABLE
        }
        if not filtered:
            return

        set_parts: list[str] = []
        values: list[Any] = []
        for col, val in filtered.items():
            set_parts.append(f"{col} = %s")
            if col == "avg_scores" and isinstance(val, dict):
                values.append(Json(val))
            else:
                values.append(val)
        values.append(job_id)

        sql = (
            f"UPDATE benchmark_runs SET {', '.join(set_parts)} WHERE id = %s"
        )

        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, tuple(values))
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._put_conn(conn)

    # ----------------------------------------------------------------- cleanup

    def close(self) -> None:
        """Close every pooled connection. Safe to call more than once."""
        try:
            self._pool.closeall()
        except Exception as exc:
            logger.warning("Database.close encountered: %s", exc)
