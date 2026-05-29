"""Core trajectory data model.

Defines the canonical record produced by capture, persisted by storage,
and consumed by metrics, server, and benchmark components.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
from typing import Any, List, Optional


_REQUIRED_STEP_KEYS = ("step_index", "type", "input", "output")


@dataclass
class Trajectory:
    task: str
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "default_agent"
    ground_truth: Optional[str] = None
    final_answer: Optional[str] = None
    steps: List[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    total_duration_ms: int = 0
    total_tokens: int = 0
    metadata: Optional[dict] = None

    # --- Optional fields for downstream metrics (added in Phase 4 schema bump).
    # All default to None so old code, old JSON, and the tracer (which never
    # populates them) remain fully compatible.
    conversation_history: Optional[List[dict]] = None
    """Multi-turn exchanges, each ``{"role": "user"|"assistant", "content": str}``.
    Consumed by MultiTurnConsistencyMetric."""

    retrieved_context: Optional[List[str]] = None
    """Context chunks from a RAG / knowledge-base lookup.
    Consumed by HallucinationMetric for grounding checks."""

    error_summary: Optional[str] = None
    """Human-readable description of what went wrong on a failed/incomplete run."""

    def to_dict(self) -> dict:
        """Return a fully JSON-serializable dict representation.

        Includes the optional ``conversation_history``, ``retrieved_context``,
        and ``error_summary`` fields; they appear with ``None`` values when
        unset, which is harmless and forward-compatible.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """Reconstruct a Trajectory from a dict.

        Unknown keys are ignored. Missing optional fields fall back to
        their declared defaults; missing required fields (``task``) will
        raise ``TypeError`` from the dataclass constructor.

        Old-style trajectory JSON written before the schema added
        ``conversation_history``, ``retrieved_context``, or ``error_summary``
        loads unchanged — those fields simply default to ``None``.
        """
        if not isinstance(data, dict):
            raise TypeError(f"from_dict expects a dict, got {type(data).__name__}")

        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    def validate(self) -> None:
        """Validate invariants of the trajectory.

        Raises:
            ValueError: if ``task`` is empty/whitespace, ``trajectory_id``
                is not a valid UUID, any step is missing required keys, or
                any of the optional schema-extension fields
                (``conversation_history``, ``retrieved_context``,
                ``error_summary``) is set to a value of the wrong type.
        """
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("Trajectory.task must be a non-empty, non-whitespace string")

        try:
            uuid.UUID(str(self.trajectory_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(
                f"Trajectory.trajectory_id is not a valid UUID string: {self.trajectory_id!r}"
            ) from exc

        for i, step in enumerate(self.steps):
            if not isinstance(step, dict):
                raise ValueError(
                    f"steps[{i}] must be a dict, got {type(step).__name__}"
                )
            missing = [k for k in _REQUIRED_STEP_KEYS if k not in step]
            if missing:
                raise ValueError(
                    f"steps[{i}] is missing required keys: {missing} "
                    f"(required: {list(_REQUIRED_STEP_KEYS)})"
                )

        if self.conversation_history is not None and not isinstance(
            self.conversation_history, list
        ):
            raise ValueError(
                "Trajectory.conversation_history must be a list when set, "
                f"got {type(self.conversation_history).__name__}"
            )

        if self.retrieved_context is not None and not isinstance(
            self.retrieved_context, list
        ):
            raise ValueError(
                "Trajectory.retrieved_context must be a list when set, "
                f"got {type(self.retrieved_context).__name__}"
            )

        if self.error_summary is not None and not isinstance(self.error_summary, str):
            raise ValueError(
                "Trajectory.error_summary must be a string when set, "
                f"got {type(self.error_summary).__name__}"
            )
