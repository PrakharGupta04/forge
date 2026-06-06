# FORGE

**An open evaluation framework for multi-step LLM agents — capture trajectories, score them across seven metrics, benchmark agents, and gate regressions in CI.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Forge CI Evaluation](https://github.com/PrakharGupta04/forge/actions/workflows/forge-eval.yml/badge.svg)](https://github.com/PrakharGupta04/forge/actions/workflows/forge-eval.yml)

---

## The Problem

Large language model agents no longer answer a single prompt and stop. They plan, call tools, read results, recover from errors, and carry context across many turns. Evaluating them with a single accuracy number — "did the final answer match?" — throws away almost everything that determines whether an agent is actually good. Two agents can return the same answer while one took two clean steps and the other flailed through nine, hallucinated a citation, and contradicted itself halfway through. A scalar score cannot tell them apart, and unit tests cannot capture behavior that is probabilistic and multi-step by nature.

The tooling gap is structural. Teams building agents tend to re-invent the same scaffolding every time: a way to record what the agent actually did step-by-step, a set of metrics that look at the *process* and not just the output, a benchmark corpus with known-good trajectories to compare against, somewhere to persist results so they can be compared over time, and a way to catch a regression *before* it merges. Each of these is fiddly on its own, and stitching them together so the numbers are reproducible and provenance-tracked is harder still. Without that infrastructure, "the new agent feels better" is the best most teams can do.

## What Forge Does

Forge provides the missing infrastructure as a set of composable pieces: a tracer that captures a LangChain agent run as a structured **trajectory**, a **metric engine** that scores that trajectory across seven process-aware metrics, a **benchmark** corpus of 50 tasks across 5 domains with golden trajectories, a **FastAPI** service that persists evaluations to PostgreSQL with full provenance, a **React dashboard** for inspecting results, and a **GitHub Actions** workflow that runs a deterministic, LLM-free subset of metrics on every pull request to block score regressions.

| Feature | Status |
| --- | --- |
| Trajectory capture from LangChain agents (`ForgeTracer`) | ✅ Implemented |
| Seven-metric evaluation engine with weighted composite | ✅ Implemented |
| Benchmark corpus (5 domains, 50 tasks, golden trajectories) | ✅ Implemented |
| Benchmark runner with native `AgentExecutor` dispatch | ✅ Implemented |
| FastAPI service (`/evaluate`, benchmarks, leaderboard, compare) | ✅ Implemented |
| PostgreSQL persistence with evaluation provenance | ✅ Implemented |
| React dashboard (Evaluate, Leaderboard, Explorer, Regression) | ✅ Implemented |
| Deterministic CI regression gate (structural metrics only) | ✅ Implemented |
| Semantic similarity for tool-input comparison | 🚧 In progress (reserved flag in `FidelityConfig`) |
| Constraint-based evaluation (required/forbidden tools) | 🚧 In progress (fields loaded, not yet enforced) |
| Human correlation study for metric validation | 🚧 Not yet conducted |

## Architecture

```mermaid
graph LR
    A[LangChain Agent] --> T[ForgeTracer]
    T --> TR[Trajectory]
    TR --> ME[MetricEngine]
    ME --> PG[(PostgreSQL)]

    AC[API Client] --> API[FastAPI]
    API --> ME

    PR[GitHub PR] --> CI[CI Action]
    CI --> BR[BenchmarkRunner]
    BR --> ME
```

## Quick Start

**Prerequisites:** Python 3.11+, Docker (for PostgreSQL and Redis), Node 18+ (for the dashboard), and a Groq API key if you want LLM-based metrics.

```bash
# 1. Clone and enter the project
git clone https://github.com/PrakharGupta04/forge.git
cd forge

# 2. Create a virtualenv and install Forge (editable)
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Start PostgreSQL and Redis
docker compose up -d postgres redis

# 4. Apply the database schema
psql "postgresql://forge:forge_dev@localhost:5432/forge" -f schema.sql

# 5. Configure environment (note: localhost, since the API runs on your host)
cat > .env <<'EOF'
DATABASE_URL=postgresql://forge:forge_dev@localhost:5432/forge
REDIS_URL=redis://localhost:6379
GROQ_API_KEY=your_groq_key_here
LLM_PROVIDER=groq
EOF

# 6. Run the API
uvicorn forge.server.main:app --reload
# API docs at http://localhost:8000/docs

# 7. (Optional) Run the dashboard in a second terminal
cd dashboard
npm install
npm run dev
# Dashboard at http://localhost:5173 (proxies /api to localhost:8000)
```

Verify the service is up:

```bash
curl http://localhost:8000/health
```

## Metrics

Forge scores each trajectory across seven metrics. Three are **structural** (deterministic, no model calls — these run in CI); one is **embedding-based**; three are **LLM-as-judge**. The composite score is a weighted mean (equal weights by default; a `research` profile is also available).

1. **Task Completion** *(LLM-as-judge)* — Measures whether the agent's final answer actually satisfies the task given the ground truth. An LLM judge compares the final answer against the expected answer and returns a graded score. *Limitation: the judge is itself an LLM and can be wrong or inconsistent, and it adds API cost and latency.*

2. **Tool Call Fidelity** *(structural)* — Measures how closely the agent's sequence of tool calls matches the task's golden trajectory, using a longest-common-subsequence match over tool names plus Jaccard overlap of tool inputs. *Limitation: it requires a `golden_trajectory` in the task metadata and cannot be computed without one; input matching is lexical, not yet semantic.*

3. **Step Efficiency** *(structural)* — Measures whether the agent reached its answer in a reasonable number of steps relative to the task's estimated `minimum_steps`. *Limitation: it is a heuristic based on a hand-estimated minimum and penalizes verbosity even when the extra steps were harmless or correct.*

4. **Reasoning Coherence** *(embedding-based)* — Measures topical consistency between consecutive LLM reasoning steps using cosine similarity of sentence embeddings (`BAAI/bge-small-en-v1.5`). *Limitation: it measures whether steps stay on-topic, not whether the reasoning is logically correct — two wrong-but-related statements score highly.*

5. **Hallucination Score** *(LLM-as-judge)* — Extracts factual claims from the final answer and checks each against the agent's tool outputs and retrieved context for grounding. *Limitation: the judge can itself hallucinate, it only checks against provided context rather than world knowledge, and it returns a neutral score when no grounding context is available.*

6. **Recovery Rate** *(structural)* — Measures whether the agent recovered and continued productively after encountering error steps in its trajectory. *Limitation: it is purely structural and confirms that the agent proceeded after an error, not that the recovery was actually correct.*

7. **Multi-Turn Consistency** *(LLM-as-judge)* — Measures whether the agent contradicts its own earlier statements across a multi-turn conversation; an LLM compares later turns against the initial statements and the first contradiction drops the score to zero. *Limitation: single-turn runs trivially score 1.0, and when every contradiction check fails it returns a neutral 0.5 rather than claiming a consistency it never measured.*

## Benchmark

The benchmark corpus lives in `data/benchmark/` and contains **50 tasks across 5 domains** (10 tasks each):

| Domain | Tasks | Focus |
| --- | --- | --- |
| `factual_research` | 10 | Single/multi-hop factual lookup |
| `code_tasks` | 10 | Code-oriented reasoning |
| `data_analysis` | 10 | Data inspection and analysis |
| `multi_turn` | 10 | Multi-turn conversational consistency |
| `tool_recovery` | 10 | Recovery after tool/error failures |

Each task includes a `golden_trajectory`, an estimated `minimum_steps`, and `ground_truth`, which the metrics use as references.

Run a benchmark programmatically:

```python
from forge.benchmark.runner import BenchmarkRunner

def my_agent(task: str) -> str:
    return "..."  # or pass a LangChain AgentExecutor directly

runner = BenchmarkRunner(agent_fn=my_agent, agent_id="my_agent")
result = runner.run(domain="factual_research", max_tasks=3)
print(result["aggregate_scores"])
```

Or asynchronously via the API:

```bash
curl -X POST http://localhost:8000/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"benchmark": "factual_research", "agent_id": "my_agent", "max_tasks": 5}'
# returns a job_id; poll GET /benchmark/{job_id}
```

## Evaluation Provenance

Every evaluation persisted through `POST /evaluate` stores an `evaluation_config` JSON object alongside the scores. It records exactly how the scores were produced:

```json
{
  "metric_names": ["task_completion", "tool_call_fidelity", "..."],
  "weighting_strategy": "equal",
  "weights": { "task_completion": 0.1428, "...": 0.1428 },
  "judge_provider": "groq",
  "judge_model": "llama-3.3-70b-versatile",
  "forge_version": "0.1.0",
  "evaluated_at": "2026-06-05T07:56:11.200615+00:00"
}
```

This matters because scores are only comparable when the configuration behind them matches. A composite produced with a different judge model, a different weighting strategy, or a different metric set is **not** directly comparable to another. Persisting provenance lets the leaderboard and the regression monitor flag incompatible comparisons instead of silently ranking apples against oranges.

## API Reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata (name, version, docs link) |
| `GET` | `/health` | Deep health check: PostgreSQL, Redis, embedding-model cache |
| `POST` | `/evaluate` | Evaluate a trajectory; returns scores, explanations, and provenance |
| `POST` | `/benchmark/run` | Queue an asynchronous benchmark run; returns a `job_id` |
| `GET` | `/benchmark/{job_id}` | Poll benchmark job status (Redis while running, PostgreSQL once complete) |
| `GET` | `/compare` | Compare two agents by their averaged metric scores |
| `GET` | `/leaderboard` | Ranked completed benchmark runs (mock agents excluded) |
| `GET` | `/trajectories/{trajectory_id}` | Fetch a stored trajectory merged with its evaluation |

Interactive OpenAPI docs are available at `/docs` when the server is running.

## Contributing

### Adding a metric

1. Create a class in `forge/metrics/` that subclasses `BaseMetric`.
2. Set a unique `METRIC_NAME` class attribute (this is the key used in result dicts).
3. Implement `score_with_explanation(self, trajectory: dict) -> MetricResult`, returning a score in `[0.0, 1.0]`, a one-sentence explanation, and any supporting metadata. (The base class derives `safe_score_with_explanation` and the legacy `safe_score` for you, with exception handling.)
4. Register it — either append the class to `MetricEngine.ALL_METRICS` in `forge/metrics/engine.py`, or call `MetricEngine.register(YourMetric)`.
5. Add a corresponding weight entry in `forge/metrics/config.py` so it participates in the composite.
6. Add unit tests under `tests/unit/`.

Keep new metrics honest: document what the metric does **not** measure, and prefer structural/deterministic logic where possible so the metric can run in CI without external services.

## Validation

Forge's metrics are designed to be process-aware, but their agreement with human judgment has **not yet been empirically measured**. The planned validation study will report inter-rater agreement between each metric and human raters using **Cohen's κ (kappa)**. The table below is a placeholder for that study.

| Metric | Agreement with human judgment (Cohen's κ) | Status |
| --- | --- | --- |
| Task Completion | TBD | Not yet validated |
| Tool Call Fidelity | TBD | Not yet validated |
| Step Efficiency | TBD | Not yet validated |
| Reasoning Coherence | TBD | Not yet validated |
| Hallucination Score | TBD | Not yet validated |
| Recovery Rate | TBD | Not yet validated |
| Multi-Turn Consistency | TBD | Not yet validated |

> **Note:** Human correlation study not yet conducted. The Cohen's κ values above are placeholders (TBD) and should not be cited as evidence of metric validity.

## License

Forge is released under the **MIT License**.
