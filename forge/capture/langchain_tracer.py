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

            llm_output = getattr(response, "llm_output", None)
            if llm_output is None:
                tokens = 0
            else:
                tokens = llm_output.get("token_usage", {}).get("total_tokens", 0) or 0

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
