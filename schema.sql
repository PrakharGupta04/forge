-- Forge schema. Targets PostgreSQL 15.
-- gen_random_uuid() is provided by core since PostgreSQL 13, no extension needed.

CREATE TABLE IF NOT EXISTS trajectories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        TEXT NOT NULL,
    task            TEXT NOT NULL,
    final_answer    TEXT,
    ground_truth    TEXT,
    raw_trajectory  JSONB NOT NULL,
    total_steps     INTEGER,
    total_tokens    INTEGER,
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluations (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id        UUID REFERENCES trajectories(id) ON DELETE CASCADE,
    task_completion      FLOAT,
    tool_call_fidelity   FLOAT,
    reasoning_coherence  FLOAT,
    hallucination_score  FLOAT,
    step_efficiency      FLOAT,
    recovery_rate        FLOAT,
    consistency          FLOAT,
    composite_score      FLOAT,
    evaluated_at         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id         TEXT NOT NULL,
    benchmark_name   TEXT NOT NULL,
    job_status       TEXT NOT NULL DEFAULT 'queued'
                     CHECK (job_status IN ('queued', 'running', 'complete', 'failed')),
    total_tasks      INTEGER,
    completed_tasks  INTEGER DEFAULT 0,
    avg_scores       JSONB,
    started_at       TIMESTAMPTZ DEFAULT now(),
    finished_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trajectories_agent_id     ON trajectories (agent_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_trajectory_id ON evaluations (trajectory_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_agent_id   ON benchmark_runs (agent_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_name       ON benchmark_runs (benchmark_name);
