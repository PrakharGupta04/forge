"""Tool-call fidelity metric.

Compares the agent's actual tool-call sequence against a "golden" reference
trajectory (supplied via ``trajectory["metadata"]["golden_trajectory"]``).

The score blends two signals:

* **name_score** — Longest Common Subsequence (LCS) over the *tool names*
  the agent invoked, normalised by the golden length. This rewards calling
  the right tools in the right order, with deletions/insertions tolerated.
* **input_score** — average Jaccard similarity over the *tool inputs* of
  the matched pairs. Crucially, the matched pairs come from the
  reconstructed LCS alignment, not naive sequential pairing — so when
  ``actual = [A, B, C]`` and ``golden = [A, C]``, ``B`` is correctly skipped
  and the input similarity for ``C`` is compared against ``actual[2]``.

Final = ``0.7 * name_score + 0.3 * input_score`` (clamped).
"""

from __future__ import annotations

from forge.metrics.base import BaseMetric


class ToolCallFidelityMetric(BaseMetric):
    """Score the agent's tool-call sequence against a golden reference."""

    @property
    def name(self) -> str:
        return "tool_call_fidelity"

    def score(self, trajectory: dict) -> float:
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

        final = 0.7 * name_score + 0.3 * input_score
        return max(0.0, min(1.0, final))

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
