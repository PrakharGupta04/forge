"""Benchmark task loader and validator.

Scans the benchmark data directory tree (one subdirectory per domain),
loads every ``*.json`` task file under it, and validates each task
against the Forge benchmark schema.

Strengthened schema rules enforced here:

* The full set of required top-level keys: ``task_id``, ``domain``,
  ``task``, ``ground_truth``, ``minimum_steps``, ``required_tools``,
  ``golden_trajectory``, ``difficulty``, ``notes``.
* ``golden_trajectory`` must be a dict containing a non-empty ``steps``
  list, and every step must include at minimum ``tool_name`` and
  ``tool_input``.
* Multi-turn tasks (``domain == "multi_turn"``) must additionally include
  a ``conversation_history`` field.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


_REQUIRED_TASK_KEYS: tuple[str, ...] = (
    "task_id",
    "domain",
    "task",
    "ground_truth",
    "minimum_steps",
    "required_tools",
    "golden_trajectory",
    "difficulty",
    "notes",
)

_REQUIRED_STEP_KEYS: tuple[str, ...] = ("tool_name", "tool_input")


class BenchmarkLoader:
    """Load and validate Forge benchmark task JSON files from a directory tree."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir is None:
            # Resolve to the project root's data/benchmark/ regardless of cwd:
            # forge/benchmark/loader.py -> forge/benchmark -> forge -> <root>
            self.data_dir = (
                Path(__file__).parent.parent.parent / "data" / "benchmark"
            ).resolve()
        else:
            self.data_dir = Path(data_dir).resolve()

    def load(self, domain: Optional[str] = None) -> list[dict]:
        """Load every task JSON under the data directory (optionally filtered by domain)."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Benchmark data directory does not exist: {self.data_dir}"
            )

        if domain is not None:
            target = self.data_dir / domain
            if not target.is_dir():
                raise FileNotFoundError(
                    f"Domain subdirectory not found: {target}"
                )
            domain_dirs = [target]
        else:
            domain_dirs = sorted(
                p for p in self.data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )

        tasks: list[dict] = []
        for domain_dir in domain_dirs:
            for json_file in sorted(domain_dir.glob("*.json")):
                with json_file.open("r", encoding="utf-8") as f:
                    task = json.load(f)
                self._validate_task(task, str(json_file))
                tasks.append(task)
        return tasks

    def _validate_task(self, task: dict, filepath: str) -> None:
        """Raise ``ValueError`` (with ``filepath`` in the message) on any schema violation."""
        if not isinstance(task, dict):
            raise ValueError(
                f"Benchmark task at {filepath} is not a JSON object "
                f"(got {type(task).__name__})"
            )

        missing = [k for k in _REQUIRED_TASK_KEYS if k not in task]
        if missing:
            raise ValueError(
                f"Benchmark task at {filepath} is missing required keys: {missing}"
            )

        golden = task["golden_trajectory"]
        if not isinstance(golden, dict):
            raise ValueError(
                f"Benchmark task at {filepath}: golden_trajectory must be a dict, "
                f"got {type(golden).__name__}"
            )
        steps = golden.get("steps")
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError(
                f"Benchmark task at {filepath}: golden_trajectory.steps must be a "
                f"non-empty list"
            )
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(
                    f"Benchmark task at {filepath}: golden_trajectory.steps[{i}] "
                    f"must be a dict, got {type(step).__name__}"
                )
            missing_step = [k for k in _REQUIRED_STEP_KEYS if k not in step]
            if missing_step:
                raise ValueError(
                    f"Benchmark task at {filepath}: golden_trajectory.steps[{i}] "
                    f"is missing required keys: {missing_step}"
                )

        if task.get("domain") == "multi_turn" and "conversation_history" not in task:
            raise ValueError(
                f"Benchmark task at {filepath}: multi_turn tasks must include "
                f"a 'conversation_history' field"
            )

    def domains(self) -> list[str]:
        """Return the list of available domain names (subdirectories under data_dir)."""
        if not self.data_dir.exists():
            return []
        return sorted(
            p.name for p in self.data_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def count(self, domain: Optional[str] = None) -> int:
        """Return the number of valid tasks discovered (overall or in one domain)."""
        return len(self.load(domain=domain))
