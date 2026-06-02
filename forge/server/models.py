"""Pydantic v2 request and response models for the Forge API.

These models are the public contract of the HTTP layer. They are
deliberately permissive on the request side (e.g. ``trajectory: dict``
rather than a fully-typed Trajectory schema) so the API can accept
trajectories produced by external agents that may not perfectly mirror
:class:`forge.capture.trajectory.Trajectory`, and strict on the
response side so clients get a stable, documented shape regardless of
internal refactors.

Mutable defaults use ``Field(default_factory=...)`` to avoid the
shared-mutable-state foot-gun (Pydantic v2 actually deep-copies plain
mutable defaults internally, but ``default_factory`` is the
unambiguous, IDE-friendly style).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------- /evaluate


class EvaluateRequest(BaseModel):
    """Request body for ``POST /evaluate``.

    The caller submits a captured trajectory (in the dict shape produced
    by :meth:`forge.capture.trajectory.Trajectory.to_dict`) plus optional
    knobs for which metrics to run, whether to return per-metric
    explanations, and which weighting strategy the composite score
    should use.
    """

    trajectory: dict
    metrics: list[str] = Field(default_factory=list)
    include_explanations: bool = True
    weighting_strategy: str = "equal"


class EvaluateResponse(BaseModel):
    """Response body for ``POST /evaluate``.

    ``has_failures`` and ``metric_errors`` surface the structured
    failure information from :class:`forge.metrics.engine.MetricEngine`
    so a partial result (one metric failed, others succeeded) is
    visible to the caller rather than being silently zeroed out.
    """

    evaluation_id: str
    trajectory_id: str
    scores: dict
    explanations: dict = Field(default_factory=dict)
    has_failures: bool = False
    metric_errors: dict = Field(default_factory=dict)
    composite_score: float
    evaluation_config: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------- /benchmarks


class BenchmarkRunRequest(BaseModel):
    """Request body for ``POST /benchmarks/run``.

    ``agent_type`` is forwarded to ``benchmark_runs.agent_type`` so the
    leaderboard query in :meth:`forge.server.db.Database.get_leaderboard`
    can filter out ``"mock"`` runs without losing the provenance.
    """

    benchmark: str = "all"
    agent_id: str = "api_agent"
    max_tasks: Optional[int] = None
    agent_type: str = "mock"


class BenchmarkRunResponse(BaseModel):
    """Synchronous handshake for ``POST /benchmarks/run``.

    The actual run is asynchronous; the API returns ``job_id`` so the
    client can poll :class:`BenchmarkStatusResponse`. ``status`` is
    typically ``"queued"`` at this point.
    """

    job_id: str
    status: str
    benchmark: str
    agent_id: str


class BenchmarkStatusResponse(BaseModel):
    """Polling response for ``GET /benchmarks/{job_id}``.

    All progress fields are optional so the same model can represent a
    job in any state: just-queued (no progress yet), running (counts
    populated), complete (scores populated), or failed (``error`` set).
    """

    job_id: str
    status: str
    total_tasks: Optional[int] = None
    completed_tasks: Optional[int] = None
    aggregate_scores: Optional[dict] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------- /compare


class CompareResponse(BaseModel):
    """Pairwise agent comparison.

    ``winner`` is the agent_id of the higher composite scorer, or
    ``None`` when the comparison is inconclusive (e.g. identical
    composite, missing data). ``warning`` carries any caveat the
    comparator wants surfaced (e.g. "agents evaluated on different
    benchmark subsets").
    """

    agent_1: str
    agent_2: str
    agent_1_scores: dict
    agent_2_scores: dict
    winner: Optional[str] = None
    warning: Optional[str] = None


# ---------------------------------------------------------------------- /leaderboard


class LeaderboardEntry(BaseModel):
    """One row of ``GET /leaderboard``.

    ``composite_score`` and ``avg_scores`` are optional because a
    completed run may still legitimately have null scores (every metric
    failed, or scores were not persisted). ``completed_at`` is ISO
    string so the response is JSON-trivially-serialisable across
    clients that may not parse Python datetimes.
    """

    agent_id: str
    benchmark_name: str
    composite_score: Optional[float]
    avg_scores: Optional[dict]
    agent_type: str
    completed_at: Optional[str]


# ---------------------------------------------------------------------- /health


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    ``components`` is a flat ``{name: bool}`` map of subsystem
    reachability, so the same payload shape works whether the API
    returns 200 (all components healthy) or 503 (at least one
    component is down).
    """

    status: str
    components: dict
    version: str
    timestamp: str
