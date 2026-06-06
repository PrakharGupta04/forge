"""Generate a clean research corpus for the Forge human-evaluation study.

This script runs a *real* LangChain ReAct agent (Groq-backed) against a
deterministically-selected slice of the Forge benchmark, captures each
run as a Forge ``Trajectory`` via :class:`ForgeTracer`, scores it with
:class:`MetricEngine`, and persists both the trajectory and its
evaluation (with provenance) under ``agent_id = "human_study_agent"``.

After generation it applies a transparent, Python-side eligibility
filter over every ``human_study_agent`` trajectory and reports how many
pass — so the operator knows whether enough clean traces exist before
running the export step (``scripts/export_for_human_study.py``).

Why a dedicated agent_id and post-hoc filter?
---------------------------------------------
The shared ``trajectories`` table is polluted with mock/CI/test traces.
Tagging every research run with a single, distinctive ``agent_id`` and
then filtering on real multi-step execution gives the human study a
clean, reproducible corpus without touching any other component.

Import note
-----------
The spec asks for ``langchain.agents`` / ``langchain.prompts`` imports,
but on this project's installed stack those symbols live under
``langchain_classic.agents`` and ``langchain_core.prompts`` (the
``langchain.*`` shims do not re-export them). We mirror the exact
try/except fallback that ``forge/benchmark/runner.py`` already uses so
the script actually runs here while still preferring the canonical path.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from forge.benchmark.loader import BenchmarkLoader
from forge.capture.langchain_tracer import ForgeTracer
from forge.metrics.config import WeightingConfig
from forge.metrics.engine import MetricEngine
from forge.server.db import Database

# AgentExecutor / create_react_agent: prefer the canonical langchain path,
# fall back to langchain_classic (the path that actually works on this
# stack — see forge/benchmark/runner.py for the same pattern).
try:  # pragma: no cover - import resolution is environment-specific
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:  # pragma: no cover
    from langchain_classic.agents import AgentExecutor, create_react_agent

try:  # pragma: no cover
    from langchain.prompts import PromptTemplate
except ImportError:  # pragma: no cover
    from langchain_core.prompts import PromptTemplate

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq


load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("generate_research_corpus")


AGENT_ID = "human_study_agent"
GROQ_MODEL = "llama-3.3-70b-versatile"
FORGE_VERSION = "0.1.0"
TARGET_ELIGIBLE = 30
MIN_ELIGIBLE = 25
RANDOM_SEED = 42

DEFAULT_DOMAIN_SPLIT = "factual_research:10,data_analysis:10,code_tasks:10"

# Eligibility-filter constants.
MIN_STEPS = 2
MIN_ANSWER_LEN = 5
MOCK_ANSWER_PREFIXES = ("mock", "ci mock", "api benchmark")
TEST_ARTIFACT_SUBSTRINGS = (
    "test",
    "roundtrip",
    "fixture",
    "provenance",
    "seed",
    "integration",
)

# DB metric column names that save_evaluation_with_config expects, mapped
# from the metric names returned by MetricEngine. Everything is identity
# except multi_turn_consistency -> consistency (the evaluations table
# stores the metric under the column name "consistency").
_METRIC_NAME_TO_COLUMN = {
    "task_completion": "task_completion",
    "tool_call_fidelity": "tool_call_fidelity",
    "step_efficiency": "step_efficiency",
    "reasoning_coherence": "reasoning_coherence",
    "hallucination_score": "hallucination_score",
    "recovery_rate": "recovery_rate",
    "multi_turn_consistency": "consistency",
}


# Inline ReAct prompt (hub.pull has deprecation issues). Must declare the
# four input variables LangChain's create_react_agent requires.
REACT_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using available tools.
Tools available:
{tools}

Tool names: {tool_names}

Use this format:
Thought: think about what to do
Action: tool_name
Action Input: the input to the tool
Observation: the result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer

Question: {input}
{agent_scratchpad}"""


# --------------------------------------------------------------------------- args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a clean research corpus for the Forge human-evaluation "
            "study (agent_id=human_study_agent)."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=40,
        help="Maximum number of tasks to attempt (default: 40).",
    )
    parser.add_argument(
        "--domain-split",
        type=str,
        default=DEFAULT_DOMAIN_SPLIT,
        help=(
            "Comma-separated domain:count pairs "
            f"(default: {DEFAULT_DOMAIN_SPLIT})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected tasks and exit without running the agent.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip tasks whose text already exists in the DB under "
            "agent_id=human_study_agent (avoids duplicate generation)."
        ),
    )
    return parser.parse_args()


def parse_domain_split(raw: str) -> dict[str, int]:
    """Parse ``domain:count,domain:count`` into an ordered ``{domain: count}``."""
    split: dict[str, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise SystemExit(
                f"Invalid --domain-split entry {pair!r}; expected 'domain:count'."
            )
        domain, _, count_str = pair.partition(":")
        domain = domain.strip()
        try:
            count = int(count_str.strip())
        except ValueError:
            raise SystemExit(
                f"Invalid count in --domain-split entry {pair!r}; "
                f"{count_str!r} is not an integer."
            )
        if count < 0:
            raise SystemExit(f"Negative count in --domain-split entry {pair!r}.")
        split[domain] = count
    if not split:
        raise SystemExit("--domain-split produced no domain:count pairs.")
    return split


# ----------------------------------------------------------------- task selection


def select_tasks(domain_split: dict[str, int], max_count: int) -> list[dict]:
    """Deterministically select tasks per domain, then shuffle (seed=42)."""
    loader = BenchmarkLoader()
    available_domains = set(loader.domains())

    selected: list[dict] = []
    for domain, want in domain_split.items():
        if domain not in available_domains:
            logger.warning(
                "Domain %r not found (available: %s); skipping.",
                domain,
                sorted(available_domains),
            )
            continue
        if want <= 0:
            continue
        tasks = loader.load(domain=domain)
        tasks.sort(key=lambda t: t.get("task_id", ""))
        if len(tasks) < want:
            logger.warning(
                "Domain %r has only %d tasks but %d requested; taking all %d.",
                domain,
                len(tasks),
                want,
                len(tasks),
            )
            chosen = tasks
        else:
            chosen = tasks[:want]
        selected.extend(chosen)

    # Shuffle with a fixed seed so the cross-domain order is reproducible
    # but interleaved (not grouped by domain).
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(selected)

    if max_count is not None and len(selected) > max_count:
        logger.info(
            "Selected %d tasks exceeds --count %d; truncating to %d.",
            len(selected),
            max_count,
            max_count,
        )
        selected = selected[:max_count]

    return selected


def print_selection_table(tasks: list[dict]) -> None:
    print("=" * 100)
    print(f"Selected {len(tasks)} task(s)")
    print("=" * 100)
    print(f"{'task_id':<10} {'domain':<18} {'difficulty':<10} task")
    print("-" * 100)
    for t in tasks:
        task_text = (t.get("task") or "").replace("\n", " ")[:80]
        print(
            f"{t.get('task_id', ''):<10} "
            f"{t.get('domain', ''):<18} "
            f"{str(t.get('difficulty', '')):<10} "
            f"{task_text}"
        )
    print("=" * 100)


# ----------------------------------------------------------------- DB helpers


def _existing_task_texts(db: Database) -> set[str]:
    """Return the stripped/lowercased task texts already stored for AGENT_ID."""
    from psycopg2.extras import RealDictCursor

    conn = db._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT task FROM trajectories WHERE agent_id = %s",
                (AGENT_ID,),
            )
            rows = cur.fetchall()
        conn.rollback()
    finally:
        db._put_conn(conn)
    return {
        (r.get("task") or "").strip().lower()
        for r in rows
        if r.get("task")
    }


def _fetch_all_human_study(db: Database) -> list[dict]:
    """Fetch every AGENT_ID trajectory joined with its latest evaluation."""
    from psycopg2.extras import RealDictCursor

    sql = """
        SELECT
            t.id AS trajectory_id,
            t.task AS task,
            t.final_answer AS final_answer,
            t.total_steps AS total_steps,
            t.created_at AS created_at,
            e.composite_score AS composite_score,
            (e.id IS NOT NULL) AS has_evaluation
        FROM trajectories t
        LEFT JOIN LATERAL (
            SELECT * FROM evaluations ev
            WHERE ev.trajectory_id = t.id
            ORDER BY ev.evaluated_at DESC
            LIMIT 1
        ) e ON true
        WHERE t.agent_id = %s
        ORDER BY t.created_at ASC
    """
    conn = db._get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (AGENT_ID,))
            rows = cur.fetchall()
        conn.rollback()
        return [dict(r) for r in rows]
    finally:
        db._put_conn(conn)


# ----------------------------------------------------------------- agent setup


def build_executor() -> AgentExecutor:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    tools = [DuckDuckGoSearchRun()]
    prompt = PromptTemplate(
        template=REACT_PROMPT_TEMPLATE,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
    )


# ----------------------------------------------------------------- generation


def build_scores_and_explanations(rich: dict) -> tuple[dict, dict]:
    """Split MetricEngine.run_all_with_explanations() output.

    Returns ``(scores, explanations)`` where ``scores`` is keyed by the
    DB column names expected by ``save_evaluation_with_config`` (so
    ``multi_turn_consistency`` is stored as ``consistency``) plus
    ``composite_score``, and ``explanations`` is keyed by metric name.
    """
    scores: dict = {}
    explanations: dict = {}
    for name, entry in rich.items():
        if name == "composite_score":
            scores["composite_score"] = entry
            continue
        column = _METRIC_NAME_TO_COLUMN.get(name, name)
        scores[column] = entry.get("score")
        explanations[name] = entry.get("explanation")
    return scores, explanations


def generate(tasks: list[dict], executor: AgentExecutor) -> None:
    db = Database()
    engine = MetricEngine(weighting_config=WeightingConfig.equal_weights())

    attempted = 0
    succeeded = 0
    failed = 0
    evaluation_errors = 0

    judge_provider = os.environ.get("LLM_PROVIDER", "groq")

    try:
        for task in tasks:
            attempted += 1
            logger.info(
                "[%d/%d] Running task %s: %s...",
                attempted,
                len(tasks),
                task["task_id"],
                task["task"][:60],
            )

            tracer = ForgeTracer(
                task=task["task"],
                ground_truth=task["ground_truth"],
                agent_id=AGENT_ID,
            )

            try:
                result = executor.invoke(
                    {"input": task["task"]},
                    config={"callbacks": [tracer]},
                )
                answer = result.get("output", "") if isinstance(result, dict) else ""
                tracer.trajectory.final_answer = answer

                trajectory_dict = tracer.get_trajectory_dict()

                # Inject task metadata required by ToolCallFidelityMetric
                # (golden_trajectory) and StepEfficiencyMetric (minimum_steps).
                trajectory_dict["metadata"] = trajectory_dict.get("metadata") or {}
                trajectory_dict["metadata"].update(
                    {
                        "minimum_steps": task.get("minimum_steps"),
                        "golden_trajectory": task.get("golden_trajectory"),
                        "task_id": task.get("task_id"),
                        "domain": task.get("domain"),
                        "difficulty": task.get("difficulty"),
                    }
                )

                # conversation_history is NOT auto-propagated by the tracer;
                # copy it through so MultiTurnConsistencyMetric can score
                # multi_turn tasks correctly.
                if task.get("conversation_history") is not None:
                    trajectory_dict["conversation_history"] = task[
                        "conversation_history"
                    ]
            except Exception as exc:
                logger.warning(
                    "Agent failed on task %s: %s", task["task_id"], exc
                )
                failed += 1
                continue

            try:
                trajectory_id = db.save_trajectory(trajectory_dict, AGENT_ID)
            except Exception as exc:
                logger.warning(
                    "DB save_trajectory failed for task %s: %s",
                    task["task_id"],
                    exc,
                )
                failed += 1
                continue

            try:
                rich = engine.run_all_with_explanations(trajectory_dict)
                scores, _explanations = build_scores_and_explanations(rich)
                evaluation_config = {
                    "judge_provider": judge_provider,
                    "judge_model": GROQ_MODEL,
                    "forge_version": FORGE_VERSION,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
                db.save_evaluation_with_config(
                    trajectory_id, scores, evaluation_config
                )
            except Exception as exc:
                logger.error(
                    "Evaluation/persistence failed for task %s "
                    "(trajectory %s saved but not evaluated): %s",
                    task["task_id"],
                    trajectory_id,
                    exc,
                )
                evaluation_errors += 1
                # NOT marked failed — the trajectory was saved.
            else:
                succeeded += 1
                logger.info(
                    "  -> saved trajectory_id=%s, composite=%.3f, steps=%d",
                    trajectory_id,
                    scores.get("composite_score", 0) or 0.0,
                    len(trajectory_dict.get("steps", [])),
                )

            # Be gentle with DuckDuckGo to avoid rate limiting.
            time.sleep(2)
    finally:
        try:
            db.close()
        except Exception:
            pass

    logger.info(
        "Generation summary: attempted=%d succeeded=%d failed=%d "
        "evaluation_errors=%d",
        attempted,
        succeeded,
        failed,
        evaluation_errors,
    )


# ----------------------------------------------------------------- eligibility


def _classify_failure(row: dict) -> str | None:
    """Return the first failing eligibility criterion, or None if it passes
    all per-row checks (duplicate detection happens separately)."""
    total_steps = row.get("total_steps") or 0
    if total_steps < MIN_STEPS:
        return "insufficient_steps"

    final_answer = row.get("final_answer")
    if not final_answer or len(final_answer) <= MIN_ANSWER_LEN:
        return "empty_answer"

    lowered_answer = final_answer.strip().lower()
    if lowered_answer.startswith(MOCK_ANSWER_PREFIXES):
        return "mock_answer"

    task_lower = (row.get("task") or "").lower()
    if any(sub in task_lower for sub in TEST_ARTIFACT_SUBSTRINGS):
        return "test_artifact"

    if not row.get("has_evaluation") or row.get("composite_score") is None:
        return "no_evaluation"

    return None


def run_eligibility_check(db: Database) -> int:
    rows = _fetch_all_human_study(db)
    total = len(rows)

    counts = {
        "insufficient_steps": 0,
        "empty_answer": 0,
        "mock_answer": 0,
        "test_artifact": 0,
        "no_evaluation": 0,
        "duplicate_task": 0,
    }

    eligible = 0
    seen_tasks: set[str] = set()
    for row in rows:
        reason = _classify_failure(row)
        if reason is not None:
            counts[reason] += 1
            continue
        # Passed all per-row filters; now deduplicate by task text.
        key = (row.get("task") or "").strip().lower()
        if key in seen_tasks:
            counts["duplicate_task"] += 1
            continue
        seen_tasks.add(key)
        eligible += 1

    print()
    print("Eligibility Filter Results")
    print("==========================")
    print(f"Total with agent_id=human_study_agent : {total}")
    print(f"  passed all filters                  : {eligible}")
    print(f"  failed insufficient_steps           : {counts['insufficient_steps']}")
    print(f"  failed empty_answer                 : {counts['empty_answer']}")
    print(f"  failed mock_answer                  : {counts['mock_answer']}")
    print(f"  failed test_artifact                : {counts['test_artifact']}")
    print(f"  failed no_evaluation                : {counts['no_evaluation']}")
    print(f"  failed duplicate_task               : {counts['duplicate_task']}")
    print()
    print(f"Eligible for human study: {eligible} trajectories")
    print(f"Target was: {TARGET_ELIGIBLE}")
    status = "SUFFICIENT" if eligible >= MIN_ELIGIBLE else "INSUFFICIENT"
    print(f"Status: {status} (need at least {MIN_ELIGIBLE} eligible)")

    return eligible


# ----------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    domain_split = parse_domain_split(args.domain_split)

    tasks = select_tasks(domain_split, args.count)
    print_selection_table(tasks)

    if args.dry_run:
        logger.info("--dry-run set; exiting after task selection.")
        return 0

    if not tasks:
        logger.error("No tasks selected; nothing to generate.")
        return 1

    if args.skip_existing:
        db = Database()
        try:
            existing = _existing_task_texts(db)
        finally:
            try:
                db.close()
            except Exception:
                pass
        before = len(tasks)
        tasks = [
            t
            for t in tasks
            if (t.get("task") or "").strip().lower() not in existing
        ]
        skipped = before - len(tasks)
        logger.info(
            "--skip-existing: removed %d task(s) already stored for %s; "
            "%d remaining.",
            skipped,
            AGENT_ID,
            len(tasks),
        )
        if not tasks:
            logger.info("All selected tasks already exist; nothing to generate.")
            # Still run the eligibility check so the operator sees corpus state.
            db = Database()
            try:
                eligible = run_eligibility_check(db)
            finally:
                try:
                    db.close()
                except Exception:
                    pass
            return _final_report(eligible)

    executor = build_executor()
    generate(tasks, executor)

    db = Database()
    try:
        eligible = run_eligibility_check(db)
    finally:
        try:
            db.close()
        except Exception:
            pass

    return _final_report(eligible)


def _final_report(eligible: int) -> int:
    if eligible < MIN_ELIGIBLE:
        print(
            "WARNING: Insufficient eligible trajectories. Re-run with "
            "--count 50 or check agent failure logs above."
        )
        return 1
    print("Research corpus generation complete. Run export script next:")
    print("python scripts/export_for_human_study.py --count 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
