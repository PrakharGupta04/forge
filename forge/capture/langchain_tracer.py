"""LangChain callback handler that records a Forge ``Trajectory``.

``ForgeTracer`` plugs into any LangChain Runnable / AgentExecutor as a
callback handler. It observes LLM and tool events and accumulates them
into a single ``Trajectory`` instance that can later be retrieved with
``get_trajectory()`` / ``get_trajectory_dict()``.

The tracer is *passive*: every callback body is wrapped in a try/except
that logs at WARNING and swallows the error, so a tracing bug can never
crash the agent it is observing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler

from forge.capture.trajectory import Trajectory


logger = logging.getLogger(__name__)


class ForgeTracer(BaseCallbackHandler):
    """Capture an agent's run as a Forge ``Trajectory``."""

    def __init__(
        self,
        task: str,
        ground_truth: Optional[str] = None,
        agent_id: str = "default_agent",
    ) -> None:
        super().__init__()
        self.trajectory = Trajectory(
            task=task,
            ground_truth=ground_truth,
            agent_id=agent_id,
        )
        self._start_time: float = time.time()
        self._step_start: Optional[float] = None
        self._current_step: dict = {}
        self._current_input: str = ""
        self._tool_name: str = "unknown"
        self._tool_input: str = ""

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:  # noqa: D401
        try:
            self._step_start = time.time()
            self._current_input = prompts[0] if prompts else ""
        except Exception as exc:
            logger.warning("ForgeTracer.on_llm_start failed: %s", exc)

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            duration_ms = int((time.time() - (self._step_start or time.time())) * 1000)
            output_text = response.generations[0][0].text
            tokens = self._extract_tokens(response)

            self.trajectory.steps.append(
                {
                    "step_index": len(self.trajectory.steps),
                    "type": "llm_call",
                    "input": self._current_input,
                    "output": output_text,
                    "duration_ms": duration_ms,
                    "tokens": tokens,
                }
            )
        except Exception as exc:
            logger.warning("ForgeTracer.on_llm_end failed: %s", exc)

    def _extract_tokens(self, response) -> int:
        """Best-effort total-token extraction across LangChain providers/versions.

        LangChain's ``LLMResult.llm_output`` is populated inconsistently across
        provider integrations (OpenAI vs Groq vs Anthropic) and across major
        LangChain versions. This helper tries the known shapes in order and
        returns the first non-zero integer found. If every path yields zero
        (or is absent), returns ``0`` and emits a DEBUG log line — the missing
        token count is visible in logs without polluting the trajectory with
        fabricated estimates. Any unexpected exception swallowed -> ``0``.

        Paths attempted (in order):

        1. ``response.llm_output["token_usage"]["total_tokens"]``
           — canonical LangChain / OpenAI shape.
        2. ``response.llm_output["usage"]["total_tokens"]``
           — observed in some ``langchain-groq`` releases.
        3. ``response.llm_output["usage"]["input_tokens"] +
           response.llm_output["usage"]["output_tokens"]``
           — Anthropic-style split-field naming.
        4. Iterate ``response.generations[0]`` and sum numeric values in each
           generation's ``generation_info["finish_reason_details"]``. Silently
           skips generations whose ``generation_info`` is absent.
        """
        try:
            llm_output = getattr(response, "llm_output", None) or {}

            token_usage = llm_output.get("token_usage") or {}
            t = token_usage.get("total_tokens", 0) or 0
            if t:
                return int(t)

            usage = llm_output.get("usage") or {}
            t = usage.get("total_tokens", 0) or 0
            if t:
                return int(t)

            t = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            if t:
                return int(t)

            generations = getattr(response, "generations", None) or []
            if generations:
                total = 0
                for gen in generations[0]:
                    info = getattr(gen, "generation_info", None)
                    if info is None:
                        continue
                    details = info.get("finish_reason_details", {})
                    if isinstance(details, dict):
                        for v in details.values():
                            if isinstance(v, (int, float)):
                                total += int(v)
                    elif isinstance(details, (int, float)):
                        total += int(details)
                if total:
                    return total

            logger.debug("token count unavailable for this provider/response format")
            return 0
        except Exception:
            return 0

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        try:
            self._step_start = time.time()
            self._tool_name = (serialized or {}).get("name", "unknown")
            self._tool_input = input_str
        except Exception as exc:
            logger.warning("ForgeTracer.on_tool_start failed: %s", exc)

    def on_tool_end(self, output, **kwargs) -> None:
        try:
            duration_ms = int((time.time() - (self._step_start or time.time())) * 1000)
            self.trajectory.steps.append(
                {
                    "step_index": len(self.trajectory.steps),
                    "type": "tool_call",
                    "tool_name": self._tool_name,
                    "tool_input": self._tool_input,
                    "input": self._tool_input,
                    "output": str(output),
                    "tool_output": str(output),
                    "duration_ms": duration_ms,
                    "tokens": 0,
                    "error": None,
                }
            )
        except Exception as exc:
            logger.warning("ForgeTracer.on_tool_end failed: %s", exc)

    def on_tool_error(self, error, **kwargs) -> None:
        try:
            if self._step_start is not None:
                duration_ms = int((time.time() - self._step_start) * 1000)
            else:
                duration_ms = 0
            self.trajectory.steps.append(
                {
                    "step_index": len(self.trajectory.steps),
                    "type": "tool_call",
                    "tool_name": getattr(self, "_tool_name", "unknown"),
                    "input": getattr(self, "_tool_input", ""),
                    "output": "",
                    "tool_output": "",
                    "duration_ms": duration_ms,
                    "tokens": 0,
                    "error": str(error),
                }
            )
        except Exception as exc:
            logger.warning("ForgeTracer.on_tool_error failed: %s", exc)

    def get_trajectory(self) -> Trajectory:
        """Finalize totals and return the underlying Trajectory."""
        self.trajectory.total_duration_ms = int((time.time() - self._start_time) * 1000)
        self.trajectory.total_tokens = sum(
            s.get("tokens", 0) for s in self.trajectory.steps
        )
        return self.trajectory

    def get_trajectory_dict(self) -> dict[str, Any]:
        """Convenience: return the finalized trajectory as a JSON-serializable dict."""
        return self.get_trajectory().to_dict()
