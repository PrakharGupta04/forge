"""Configuration dataclasses for the Forge metric engine and metrics.

Two configs live here:

* :class:`WeightingConfig` — per-metric weights used by
  :class:`forge.metrics.engine.MetricEngine` to compute the weighted
  composite score. ``equal_weights()`` reproduces the original
  arithmetic-mean behaviour; ``research_weights()`` is a presetcalibrated
  for research-style benchmarking where correctness (task completion +
  hallucination) matters more than efficiency.
* :class:`FidelityConfig` — knobs for
  :class:`forge.metrics.tool_call_fidelity.ToolCallFidelityMetric`,
  including the name/input weight split, an order-sensitivity toggle,
  and forward-looking fields (semantic similarity, extra-call penalties)
  not yet wired into the scoring logic.

All weights and toggles default to the same values that were hard-coded
prior to this change, so existing callers and tests see no behavioural
change unless they explicitly pass a different config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_METRIC_NAMES: tuple[str, ...] = (
    "task_completion",
    "tool_call_fidelity",
    "step_efficiency",
    "reasoning_coherence",
    "hallucination_score",
    "recovery_rate",
    "multi_turn_consistency",
)


@dataclass
class WeightingConfig:
    """Per-metric weights used by ``MetricEngine`` for composite scoring.

    Weights need not sum to 1.0 — :meth:`normalized_weights` produces a
    re-normalized copy with values that sum to 1.0. With the default
    equal-weight config, the weighted composite equals the arithmetic
    mean, preserving every existing composite-score test.
    """

    weights: dict = field(
        default_factory=lambda: {
            "task_completion": 1.0,
            "tool_call_fidelity": 1.0,
            "step_efficiency": 1.0,
            "reasoning_coherence": 1.0,
            "hallucination_score": 1.0,
            "recovery_rate": 1.0,
            "multi_turn_consistency": 1.0,
        }
    )

    def normalized_weights(self) -> dict:
        """Return a copy of :attr:`weights` rescaled to sum to 1.0.

        Empty or all-zero weights yield an empty dict (the engine treats
        an empty contribution set as composite 0.0).
        """
        total = sum(self.weights.values())
        if total <= 0:
            return {}
        return {name: float(w) / total for name, w in self.weights.items()}

    @classmethod
    def equal_weights(cls) -> "WeightingConfig":
        """Equal weights for all seven canonical metrics (default)."""
        return cls()

    @classmethod
    def research_weights(cls) -> "WeightingConfig":
        """Preset: emphasises task completion + hallucination correctness.

        Useful for research-style benchmark runs where being right matters
        far more than being fast or efficient.
        """
        return cls(
            weights={
                "task_completion": 2.0,
                "tool_call_fidelity": 1.0,
                "step_efficiency": 0.5,
                "reasoning_coherence": 0.5,
                "hallucination_score": 2.0,
                "recovery_rate": 1.0,
                "multi_turn_consistency": 1.0,
            }
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if any weight is negative or any of the
        seven canonical metric names is missing from :attr:`weights`."""
        for name, w in self.weights.items():
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(
                    f"WeightingConfig.weights[{name!r}]={w!r} is invalid; "
                    f"weights must be non-negative numbers"
                )
        missing = [n for n in _METRIC_NAMES if n not in self.weights]
        if missing:
            raise ValueError(
                f"WeightingConfig.weights is missing required metric names: "
                f"{missing}"
            )


@dataclass
class FidelityConfig:
    """Knobs for :class:`ToolCallFidelityMetric`.

    ``name_weight`` and ``input_weight`` must sum to 1.0 within 0.001.
    ``use_semantic_similarity`` and ``penalize_extra_calls`` are reserved
    forward-looking fields that the current scorer does not yet act on;
    they are accepted today so callers can begin opting in without a
    follow-up API break.
    """

    name_weight: float = 0.7
    input_weight: float = 0.3
    use_semantic_similarity: bool = False
    order_sensitive: bool = True
    penalize_extra_calls: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` if ``name_weight + input_weight`` is not
        within 0.001 of 1.0."""
        total = float(self.name_weight) + float(self.input_weight)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"FidelityConfig.name_weight + input_weight must equal 1.0 "
                f"(±0.001); got {self.name_weight} + {self.input_weight} = {total}"
            )
