"""Minimal storage layer for Forge.

Intentionally narrow scope: just enough save/load surface to support the
storage-roundtrip verification script. A fuller repository abstraction
(filtering, pagination, batch reads, retry policy, connection pooling)
will replace this once the API server layer lands.

The implementation deliberately mirrors the current ``schema.sql``:

* ``save_trajectory`` writes to the columns the schema actually has
  (``agent_id``, ``task``, ``final_answer``, ``ground_truth``,
  ``raw_trajectory``, ``total_steps``, ``total_tokens``, ``duration_ms``).
* ``save_evaluation`` writes the 7 per-metric columns plus
  ``composite_score``. Metric scores absent from the input dict are
  persisted as ``NULL``, so the table accommodates partial metric
  coverage during the build-out of the seven-metric suite without
  schema changes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import Json
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


class Database:
    """Thin psycopg2 wrapper over the trajectories + evaluations tables."""

    def __init__(self) -> None:
        load_dotenv()
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to .env (e.g. "
                "postgresql://forge:forge_dev@localhost:5432/forge) before "
                "constructing Database()."
            )
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = False

    def health_check(self) -> bool:
        """Run ``SELECT 1`` and report success/failure as a bool."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception as exc:
            logger.error("Database.health_check failed: %s", exc)
            try:
                self.conn.rollback()
            except Exception:
                pass
            return False

    def save_trajectory(self, trajectory_dict: dict, agent_id: str) -> str:
        """Insert one row into ``trajectories``; return the generated UUID."""
        total_steps = trajectory_dict.get("total_steps")
        if total_steps is None:
            total_steps = len(trajectory_dict.get("steps", []))

        duration_ms = trajectory_dict.get("duration_ms")
        if duration_ms is None:
            duration_ms = trajectory_dict.get("total_duration_ms")

        try:
            with self.conn.cursor() as cur:
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
                new_id = cur.fetchone()[0]
            self.conn.commit()
            return str(new_id)
        except Exception:
            self.conn.rollback()
            raise

    def load_trajectory(self, trajectory_id: str) -> dict:
        """Return the ``raw_trajectory`` JSONB for ``trajectory_id`` as a dict."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT raw_trajectory FROM trajectories WHERE id = %s",
                    (trajectory_id,),
                )
                row = cur.fetchone()
        except Exception:
            self.conn.rollback()
            raise

        if row is None:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        return row[0]

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

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO evaluations (trajectory_id, {col_list})
                    VALUES (%s, {placeholders})
                    RETURNING id
                    """,
                    (trajectory_id, *values),
                )
                new_id = cur.fetchone()[0]
            self.conn.commit()
            return str(new_id)
        except Exception:
            self.conn.rollback()
            raise

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

        try:
            with self.conn.cursor() as cur:
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
                new_id = cur.fetchone()[0]
            self.conn.commit()
            return str(new_id)
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Close the underlying connection. Safe to call more than once."""
        try:
            self.conn.close()
        except Exception as exc:
            logger.warning("Database.close encountered: %s", exc)
