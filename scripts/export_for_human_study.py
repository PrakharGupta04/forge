"""Export stored trajectories for a human annotation study.

Pulls the most recent N trajectories (with their latest evaluation, if any)
from PostgreSQL and writes three artifacts into the output directory:

* ``trajectories_for_rating.json`` — what raters see. Contains ONLY the
  task, step summaries, final answer, and ground truth. Deliberately
  excludes Forge's automated scores so raters are not anchored by them.
* ``forge_scores_internal.json`` — the automated metric scores, keyed by
  trajectory_id, kept separate for later correlation analysis. NOT to be
  shown to raters.
* ``rating_form_instructions.md`` — the rater instructions with explicit
  anchors for all five rating dimensions.

Schema note: the canonical trajectory key is ``trajectories.id``; the
evaluations table references it via ``evaluations.trajectory_id``. The
DB column ``consistency`` is surfaced here under its metric name
``multi_turn_consistency``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from forge.server.db import Database

load_dotenv()

FORGE_VERSION = "0.1.0"
DEFAULT_COUNT = 30
MIN_COUNT = 5
MAX_COUNT = 200
OUTPUT_PREVIEW_CHARS = 100

# DB metric columns (left) mapped to the metric names used in the export
# (right). Everything matches except `consistency` -> `multi_turn_consistency`.
METRIC_COLUMN_MAP = {
    "task_completion": "task_completion",
    "tool_call_fidelity": "tool_call_fidelity",
    "step_efficiency": "step_efficiency",
    "reasoning_coherence": "reasoning_coherence",
    "hallucination_score": "hallucination_score",
    "recovery_rate": "recovery_rate",
    "consistency": "multi_turn_consistency",
}


def _count_type(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--count must be an integer, got {raw!r}")
    if value < MIN_COUNT or value > MAX_COUNT:
        raise argparse.ArgumentTypeError(
            f"--count must be between {MIN_COUNT} and {MAX_COUNT}, got {value}"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export stored trajectories for a human annotation study."
    )
    parser.add_argument(
        "--count",
        type=_count_type,
        default=DEFAULT_COUNT,
        help=f"Number of trajectories to export ({MIN_COUNT}-{MAX_COUNT}, default {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="human_study",
        help="Directory to write export artifacts into (default: human_study).",
    )
    return parser.parse_args()


def _total_trajectories(db: Database) -> int:
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trajectories")
            (count,) = cur.fetchone()
        conn.rollback()
        return int(count)
    finally:
        db._put_conn(conn)


def _fetch_recent(db: Database, limit: int) -> list[dict]:
    from psycopg2.extras import RealDictCursor

    metric_cols = ", ".join(f"e.{col}" for col in METRIC_COLUMN_MAP)
    sql = f"""
        SELECT
            t.id AS trajectory_id,
            t.task AS task,
            t.ground_truth AS ground_truth,
            t.final_answer AS final_answer,
            t.raw_trajectory AS raw_trajectory,
            t.created_at AS created_at,
            (e.id IS NOT NULL) AS has_evaluation,
            {metric_cols},
            e.composite_score AS composite_score
        FROM trajectories t
        LEFT JOIN LATERAL (
            SELECT * FROM evaluations ev
            WHERE ev.trajectory_id = t.id
            ORDER BY ev.evaluated_at DESC
            LIMIT 1
        ) e ON true
        ORDER BY t.created_at DESC
        LIMIT %s
    """
    conn = db._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        conn.rollback()
        return [dict(r) for r in rows]
    finally:
        db._put_conn(conn)


def _steps_of(raw_trajectory) -> list:
    if isinstance(raw_trajectory, dict):
        steps = raw_trajectory.get("steps")
        return steps if isinstance(steps, list) else []
    return []


def _step_summary(steps: list) -> list[str]:
    summary = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_type = step.get("type", "unknown")
        raw_output = step.get("output", "")
        text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
        preview = text[:OUTPUT_PREVIEW_CHARS]
        summary.append(f"Step {i}: {step_type} — {preview}")
    return summary


def _forge_scores(row: dict) -> dict | None:
    if not row.get("has_evaluation"):
        return None
    scores: dict = {}
    for db_col, metric_name in METRIC_COLUMN_MAP.items():
        value = row.get(db_col)
        scores[metric_name] = float(value) if isinstance(value, (int, float)) else None
    composite = row.get("composite_score")
    scores["composite_score"] = (
        float(composite) if isinstance(composite, (int, float)) else None
    )
    return scores


RATING_INSTRUCTIONS_TEMPLATE = """# Forge Human Rating Study — Instructions

Thank you for participating in the Forge human evaluation study. The purpose of this study is to collect human judgments of AI agent execution traces so we can measure how well Forge's automated metrics agree with human raters.

You will evaluate **{count} AI agent execution traces** and rate each one on **5 dimensions**. Each trace shows the task, the agent's reasoning and tool-use steps, and its final answer. Rate the agent's *process and output* — not how you would have solved the task yourself.

> You will **not** see Forge's automated scores while rating. This is intentional: it keeps your judgment independent and avoids anchoring bias.

## Rating dimensions

### Question 1 — Task Completion
*Did the agent accomplish the task?*

- **0** — Agent completely failed or produced a wrong answer. *Example: asked for the capital of Japan, the agent answers "Beijing".*
- **1** — Agent partially completed the task. *Example: asked for a country's capital and its population, the agent gives the correct capital but no population.*
- **2** — Agent fully and correctly completed the task. *Example: asked for the capital of Japan, the agent answers "Tokyo" with correct supporting detail.*

### Question 2 — Reasoning Quality
*Are the reasoning steps coherent and on-topic?* **Evaluate the reasoning steps shown, not the final answer.**

- **1** — Reasoning steps are completely off-topic or contradictory.
- **2** — Reasoning is mostly off-topic with some relevant steps.
- **3** — Reasoning is somewhat relevant but with gaps.
- **4** — Reasoning is mostly coherent and relevant.
- **5** — Reasoning steps follow logically and stay on topic throughout.

### Question 3 — Tool Usage
*Did the agent use the right tools with the right inputs?*

- **1** — Agent used wrong tools or called tools with completely incorrect inputs.
- **2** — Agent used some correct tools but with suboptimal inputs or unnecessary calls.
- **3** — Agent used appropriate tools with correct inputs.

> Note: if the task required no tools and the agent used none, rate as **3**.

### Question 4 — Hallucination
*Does the final answer contain unsupported factual claims?*

- **0** — The final answer contains at least one factual claim that is incorrect or unsupported by the information retrieved.
- **1** — The final answer contains no identifiable incorrect factual claims.

> Note: if the agent said "I don't know" or produced no answer, rate as **1** (no hallucination detected).

### Question 5 — Efficiency
*Did the agent reach the answer directly?*

- **1** — Agent took many unnecessary steps to reach the answer.
- **2** — Agent took somewhat more steps than needed but reached the answer.
- **3** — Agent reached the answer in a minimal and direct way.

## How to submit

Record your ratings in the Google Form / spreadsheet provided by the study coordinator. Rate every trajectory before submitting. If a trajectory is unclear, provide your best rating and add a note explaining the uncertainty.
"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 56)
    print("Forge human-study export")
    print("=" * 56)
    print(f"Requested count : {args.count}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Forge version   : {FORGE_VERSION}")
    print("-" * 56)

    try:
        db = Database()
    except Exception as exc:
        print(f"ERROR: could not connect to the database: {exc}", file=sys.stderr)
        return 1

    try:
        total = _total_trajectories(db)
        if total == 0:
            print("ERROR: No trajectories found in the database. Nothing to export.", file=sys.stderr)
            return 1

        if total < args.count:
            print(
                f"WARNING: Only {total} trajectories found in database, fewer "
                f"than requested {args.count}. Proceeding with {total}."
            )
        effective_count = min(args.count, total)

        rows = _fetch_recent(db, effective_count)
    except Exception as exc:
        print(f"ERROR: export query failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            db.close()
        except Exception:
            pass

    rater_trajectories: list[dict] = []
    internal_scores: list[dict] = []
    with_eval = 0

    for seq, row in enumerate(rows, start=1):
        steps = _steps_of(row.get("raw_trajectory"))
        trajectory_id = str(row.get("trajectory_id"))

        rater_trajectories.append(
            {
                "id": seq,
                "trajectory_id": trajectory_id,
                "task": row.get("task"),
                "ground_truth": row.get("ground_truth"),
                "final_answer": row.get("final_answer"),
                "num_steps": len(steps),
                "step_summary": _step_summary(steps),
            }
        )

        scores = _forge_scores(row)
        if scores is None:
            internal_scores.append({"trajectory_id": trajectory_id, "forge_scores": None})
        else:
            with_eval += 1
            internal_scores.append({"trajectory_id": trajectory_id, **scores})

    without_eval = len(rows) - with_eval
    output_dir.mkdir(parents=True, exist_ok=True)

    rating_file = output_dir / "trajectories_for_rating.json"
    internal_file = output_dir / "forge_scores_internal.json"
    instructions_file = output_dir / "rating_form_instructions.md"

    rating_payload = {
        "metadata": {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "trajectory_count": len(rater_trajectories),
            "forge_version": FORGE_VERSION,
            "note": (
                "forge_scores are not shown to raters during rating — they are "
                "stored separately in forge_scores_internal.json and used only "
                "for correlation analysis."
            ),
        },
        "trajectories": rater_trajectories,
    }

    with rating_file.open("w", encoding="utf-8") as f:
        json.dump(rating_payload, f, indent=2)

    with internal_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "export_timestamp": rating_payload["metadata"]["export_timestamp"],
                    "trajectory_count": len(internal_scores),
                    "forge_version": FORGE_VERSION,
                    "note": "Internal automated scores. Do NOT show to human raters.",
                },
                "scores": internal_scores,
            },
            f,
            indent=2,
        )

    with instructions_file.open("w", encoding="utf-8") as f:
        f.write(RATING_INSTRUCTIONS_TEMPLATE.format(count=len(rater_trajectories)))

    print(f"Exported trajectories : {len(rater_trajectories)}")
    print(f"  with evaluations    : {with_eval}")
    print(f"  without evaluations : {without_eval}")
    print("-" * 56)
    print(f"Rater file     : {rating_file.resolve()}")
    print(f"Internal scores: {internal_file.resolve()}")
    print(f"Instructions   : {instructions_file.resolve()}")
    print("=" * 56)
    print(
        f"Human study export complete — {len(rater_trajectories)} trajectories ready for rating"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
