-- Migration 001 — add evaluation provenance to the Forge schema.
--
-- Targets PostgreSQL 15 (matches schema.sql). Apply manually with::
--
--     psql "$DATABASE_URL" -f migrations/001_add_evaluation_provenance.sql
--
-- This migration is idempotent: every statement uses ``IF NOT EXISTS`` so
-- re-running it on an already-migrated database is a no-op. Apply BEFORE
-- any evaluations are written through the API layer — adding provenance
-- columns later requires backfilling NULLs for every existing row and is
-- a meaningfully harder operation than starting clean.
--
-- What this migration does
-- ------------------------
--   1. evaluations.evaluation_config  (JSONB, nullable)  — per-run
--      provenance bag: metric_names, weighting_strategy, weights,
--      judge_model, judge_provider, forge_version, evaluated_at.
--      Existing rows get NULL, which is acceptable and queryable.
--
--   2. benchmark_runs.agent_type  (TEXT, default 'unknown')  —
--      distinguishes real agent runs from mock runs so leaderboards
--      can filter appropriately. Existing rows fall through to the
--      default at SELECT time.
--
--   3. human_evaluations  (new table)  — anonymous human ratings of
--      individual trajectories, used as a ground-truth signal for
--      calibrating the LLM-as-judge metrics over time.

BEGIN;

-- 1. evaluation_config on evaluations -----------------------------------------
ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS evaluation_config JSONB;

COMMENT ON COLUMN evaluations.evaluation_config IS
    'Per-evaluation provenance: metric_names, weighting_strategy, weights, '
    'judge_model, judge_provider, forge_version, evaluated_at. NULL for '
    'rows written before migration 001.';

-- 2. agent_type on benchmark_runs ---------------------------------------------
ALTER TABLE benchmark_runs
    ADD COLUMN IF NOT EXISTS agent_type TEXT DEFAULT 'unknown';

COMMENT ON COLUMN benchmark_runs.agent_type IS
    'Agent implementation flavour, e.g. "langchain_react", "openai_functions", '
    '"mock". Defaults to "unknown" so leaderboard filters degrade safely.';

-- 3. human_evaluations table --------------------------------------------------
CREATE TABLE IF NOT EXISTS human_evaluations (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id          UUID REFERENCES trajectories(id) ON DELETE CASCADE,
    evaluator_id           TEXT NOT NULL,
    task_completed         BOOLEAN,
    answer_quality         INTEGER CHECK (answer_quality BETWEEN 1 AND 5),
    hallucination_observed BOOLEAN,
    notes                  TEXT,
    evaluated_at           TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE human_evaluations IS
    'Anonymous per-trajectory human ratings. ON DELETE CASCADE so removing '
    'a trajectory also drops its human evaluations (they are meaningless '
    'in isolation).';

CREATE INDEX IF NOT EXISTS idx_human_evaluations_trajectory_id
    ON human_evaluations (trajectory_id);

CREATE INDEX IF NOT EXISTS idx_human_evaluations_evaluator_id
    ON human_evaluations (evaluator_id);

COMMIT;
