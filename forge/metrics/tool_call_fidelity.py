"""Tool-call fidelity metric.

What it measures
----------------
Compares the agent's actual tool-call sequence to a "golden" reference
sequence supplied via ``trajectory["metadata"]["golden_trajectory"]``.
The golden trajectory is a dict shaped like::

    {"steps": [{"tool_name": "...", "tool_input": "..."}, ...]}

and is typically loaded from a benchmark task JSON (see
:mod:`forge.benchmark.loader`).

Scoring algorithm
-----------------
Two sub-scores are blended into the final value:

* **name_score** — Longest Common Subsequence (LCS) length over the
  *tool names* the agent invoked, divided by the golden length and
  clamped to ``[0, 1]``. LCS rewards calling the right tools in the
  right order; deletions, insertions, and out-of-order calls cost
  alignment.
* **input_score** — average Jaccard similarity (whitespace-tokenised
  set overlap) of the *tool inputs* across the matched pairs from the
  reconstructed LCS path. Critically, pairs come from the LCS backtrace
  rather than naïve ``zip(actual, golden)``: when
  ``actual = [A, B, C]`` and ``golden = [A, C]``, ``B`` is correctly
  skipped and ``C`` is compared against ``actual[2]``.

Final score (clamped to ``[0, 1]``):

    name_weight · name_score + input_weight · input_score

Weights default to ``0.7 / 0.3`` and are configurable per metric
instance via :class:`~forge.metrics.config.FidelityConfig`.

Known limitation
----------------
``input_score`` is a Jaccard set-overlap measure, which is brittle for
semantically equivalent but lexically different inputs
(``"capital of France"`` vs ``"France capital city"``). The
``use_semantic_similarity`` flag on :class:`FidelityConfig` is reserved
for a future embedding-based input comparison; the current scorer does
not yet act on it. Likewise ``penalize_extra_calls`` is accepted today
but not yet wired into the LCS-normalised name score.
"""

from __future__ import annotations

from typing import Optional

from forge.metrics.base import BaseMetric
from forge.metrics.config import FidelityConfig
from forge.metrics.result import MetricResult


class ToolCallFidelityMetric(BaseMetric):
    """Score the agent's tool-call sequence against a golden reference."""

    METRIC_NAME = "tool_call_fidelity"

    def __init__(self, config: Optional[FidelityConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else FidelityConfig()

    @property
    def name(self) -> str:
        return self.METRIC_NAME

    def score(self, trajectory: dict) -> float:
        return self._compute(trajectory)["score"]

    def score_with_explanation(self, trajectory: dict) -> MetricResult:
        """Return a :class:`MetricResult` with intermediate alignment values.

        Calls ``self.score(trajectory)`` first so that any test that
        patches ``score`` (see ``tests/unit/test_engine_complete.py``)
        observes the patched value here too. Intermediate values for the
        explanation are captured via a best-effort second call to
        :meth:`_compute`; if that re-computation raises (e.g. patched
        scores on a synthetic trajectory with no golden), the
        explanation degrades to a generic ``"Score: 0.xxx"`` sentence
        but the (patched) score is still surfaced.
        """
        final_score = self.score(trajectory)

        try:
            info = self._compute(trajectory)
        except Exception:
            return MetricResult(
                score=final_score,
                explanation=f"Score: {final_score:.3f}",
                metadata={},
                metric_name=self.name,
            )

        lcs_length = info["lcs_length"]
        golden_length = info["golden_length"]
        name_score = info["name_score"]
        input_score = info["input_score"]
        actual_tool_count = info["actual_tool_count"]
        golden_tool_count = info["golden_tool_count"]

        explanation = (
            f"Tool sequence matched {lcs_length}/{golden_length} golden "
            f"steps (name score: {name_score:.2f}, input overlap: "
            f"{input_score:.2f})"
        )
        metadata = {
            "lcs_length": lcs_length,
            "golden_length": golden_length,
            "name_score": name_score,
            "input_score": input_score,
            "actual_tool_count": actual_tool_count,
            "golden_tool_count": golden_tool_count,
        }
        return MetricResult(
            score=final_score,
            explanation=explanation,
            metadata=metadata,
            metric_name=self.name,
        )

    # ----------------------------------------------------------------- internal

    def _compute(self, trajectory: dict) -> dict:
        """Run the full LCS + Jaccard pipeline and return intermediates.

        Used by both :meth:`score` (for the float) and
        :meth:`score_with_explanation` (for the metadata bag). Raises
        ``ValueError`` on missing/empty ``golden_trajectory``, matching
        the prior behaviour of :meth:`score`.
        """
        actual_calls = [
            s for s in trajectory.get("steps", []) if s.get("type") == "tool_call"
        ]

        metadata = trajectory.get("metadata") or {}
        golden = metadata.get("golden_trajectory")
        if not golden:
            raise ValueError(
                "No golden_trajectory in metadata — cannot compute tool call fidelity"
            )

        golden_steps = [s for s in golden.get("steps", []) if "tool_name" in s]
        if not golden_steps:
            raise ValueError("golden_trajectory has no tool_call steps")

        pairs = self._lcs_alignment(actual_calls, golden_steps)

        name_score = min(1.0, len(pairs) / len(golden_steps))

        if not pairs:
            input_score = 0.0
        else:
            similarities = [
                self._input_overlap_score(
                    self._extract_input(actual_calls[ai]),
                    self._extract_input(golden_steps[gi]),
                )
                for ai, gi in pairs
            ]
            input_score = sum(similarities) / len(similarities)

        nw = float(self.config.name_weight)
        iw = float(self.config.input_weight)
        final = nw * name_score + iw * input_score
        final = max(0.0, min(1.0, final))

        return {
            "score": final,
            "name_score": name_score,
            "input_score": input_score,
            "lcs_length": len(pairs),
            "golden_length": len(golden_steps),
            "actual_tool_count": len(actual_calls),
            "golden_tool_count": len(golden_steps),
        }

    def _lcs_tool_names(self, actual: list, golden: list) -> int:
        """Return the LCS length when matching only on ``tool_name`` equality."""
        n, m = len(actual), len(golden)
        if n == 0 or m == 0:
            return 0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if actual[i].get("tool_name") == golden[j].get("tool_name"):
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        return dp[n][m]

    def _lcs_alignment(self, actual: list, golden: list) -> list[tuple[int, int]]:
        """Reconstruct the LCS path and return matched ``(actual_idx, golden_idx)`` pairs.

        Pairs are returned in ascending order of both indices. This is the
        canonical LCS backtrace and avoids the naive ``zip(actual, golden)``
        pairing bug where ``actual = [A, B, C]`` / ``golden = [A, C]`` would
        incorrectly pair ``B`` with ``C``.
        """
        n, m = len(actual), len(golden)
        if n == 0 or m == 0:
            return []

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                if actual[i].get("tool_name") == golden[j].get("tool_name"):
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        pairs: list[tuple[int, int]] = []
        i, j = n, m
        while i > 0 and j > 0:
            if actual[i - 1].get("tool_name") == golden[j - 1].get("tool_name"):
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        pairs.reverse()
        return pairs

    def _input_overlap_score(self, actual_input, golden_input) -> float:
        """Jaccard similarity of whitespace-tokenised inputs."""
        if not isinstance(actual_input, str):
            actual_input = str(actual_input)
        if not isinstance(golden_input, str):
            golden_input = str(golden_input)

        if actual_input == "" and golden_input == "":
            return 1.0

        a = set(actual_input.split())
        g = set(golden_input.split())
        union = a | g
        if not union:
            return 1.0
        return len(a & g) / len(union)

    @staticmethod
    def _extract_input(step: dict):
        """Pick the most natural input field on a tool-call step."""
        if "tool_input" in step:
            return step["tool_input"]
        return step.get("input", "")
