import { useState } from 'react'
import { evaluateTrajectory } from '../api'

// Canonical metric order, matching forge.metrics.engine.MetricEngine.ALL_METRICS.
const METRIC_ORDER = [
  'task_completion',
  'tool_call_fidelity',
  'step_efficiency',
  'reasoning_coherence',
  'hallucination_score',
  'recovery_rate',
  'multi_turn_consistency',
]

// ---------------------------------------------------------------- helpers

function formatMetricName(key) {
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

function tierOf(score) {
  const pct = score * 100
  if (pct >= 70) return 'green'
  if (pct >= 40) return 'amber'
  return 'red'
}

const FILL_CLASS = {
  green: 'bg-[#10B981]',
  amber: 'bg-[#F59E0B]',
  red: 'bg-[#EF4444]',
}

const VAL_TEXT_CLASS = {
  green: 'text-[#059669]',
  amber: 'text-[#D97706]',
  red: 'text-[#DC2626]',
}

const PILL_CLASS = {
  green: 'bg-[#F0FDF4] text-[#059669]',
  amber: 'bg-[#FFFBEB] text-[#D97706]',
  red: 'bg-[#FEF2F2] text-[#DC2626]',
}

const BADGE_CLASS = {
  green: 'bg-[#F0FDF4] border-[#BBF7D0] text-[#059669]',
  amber: 'bg-[#FFFBEB] border-[#FDE68A] text-[#D97706]',
  red: 'bg-[#FEF2F2] border-[#FECACA] text-[#DC2626]',
}

function formatPct(score) {
  const pct = score * 100
  const rounded = Math.round(pct * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}%` : `${rounded.toFixed(1)}%`
}

// Built once at component mount so the trajectory_id is a fresh valid UUID
// without regenerating on every render.
function buildSampleTrajectory() {
  return {
    trajectory_id: crypto.randomUUID(),
    task: 'What is the capital of Japan?',
    agent_id: 'sample_agent',
    ground_truth: 'Tokyo',
    final_answer: 'The capital of Japan is Tokyo.',
    steps: [
      {
        step_index: 0,
        type: 'llm_call',
        input: 'What is the capital of Japan?',
        output: 'I should search for the capital of Japan.',
        duration_ms: 100,
        tokens: 15,
      },
      {
        step_index: 1,
        type: 'tool_call',
        tool_name: 'web_search',
        input: 'capital of Japan',
        output: 'Tokyo is the capital of Japan.',
        duration_ms: 400,
        tokens: 0,
      },
      {
        step_index: 2,
        type: 'llm_call',
        input: 'Search returned: Tokyo is the capital of Japan.',
        output: 'The capital of Japan is Tokyo.',
        duration_ms: 200,
        tokens: 20,
      },
    ],
    timestamp: new Date().toISOString(),
    total_duration_ms: 700,
    total_tokens: 35,
    metadata: {},
  }
}

// ---------------------------------------------------------------- subcomponents

function MetricBar({ name, score }) {
  const tier = tierOf(score)
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-[140px] flex-shrink-0 text-xs text-[#6B7280]">
        {formatMetricName(name)}
      </div>
      <div className="h-2 flex-1 overflow-hidden rounded bg-[#F3F4F6]">
        <div
          className={`h-full rounded ${FILL_CLASS[tier]}`}
          style={{ width: `${Math.max(0, Math.min(100, score * 100))}%` }}
        />
      </div>
      <div
        className={`w-12 flex-shrink-0 text-right text-xs font-medium ${VAL_TEXT_CLASS[tier]}`}
      >
        {formatPct(score)}
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#E5E7EB] border-t-[#2563EB]" />
      <div className="text-sm font-medium text-[#374151]">
        Running evaluation — this may take 30–60 seconds
      </div>
      <div className="max-w-xs text-xs text-[#9CA3AF]">
        Task completion, hallucination, and multi-turn consistency use
        LLM-as-judge calls, and reasoning coherence runs a local embedding
        model — these are the slow steps.
      </div>
    </div>
  )
}

function ErrorBox({ message, onRetry }) {
  return (
    <div className="rounded-lg border border-[#FECACA] bg-[#FEF2F2] p-4">
      <div className="mb-2 text-sm font-medium text-[#B91C1C]">
        Evaluation failed
      </div>
      <div className="mb-3 break-words text-xs text-[#DC2626]">{message}</div>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-md border border-[#E5E7EB] bg-white px-3 py-1.5 text-xs font-medium text-[#374151]"
      >
        Try again
      </button>
    </div>
  )
}

function ExplanationTable({ scores, explanations, metricErrors }) {
  const rows = METRIC_ORDER.filter((key) => key in scores)
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <th className="w-40 border-b border-[#F3F4F6] px-2.5 py-2 text-left text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Metric
            </th>
            <th className="w-16 border-b border-[#F3F4F6] px-2.5 py-2 text-left text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Score
            </th>
            <th className="border-b border-[#F3F4F6] px-2.5 py-2 text-left text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
              Explanation
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((key) => {
            const score = scores[key]
            const tier = tierOf(score)
            const errorMsg = metricErrors?.[key]
            const explanation = explanations?.[key]
            return (
              <tr key={key}>
                <td className="border-b border-[#F9FAFB] px-2.5 py-2.5 font-medium text-[#374151]">
                  {formatMetricName(key)}
                </td>
                <td className="border-b border-[#F9FAFB] px-2.5 py-2.5">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-semibold ${PILL_CLASS[tier]}`}
                  >
                    {formatPct(score)}
                  </span>
                </td>
                <td className="border-b border-[#F9FAFB] px-2.5 py-2.5 align-top">
                  {errorMsg ? (
                    <span className="break-words text-[#DC2626]">
                      {errorMsg}
                    </span>
                  ) : explanation ? (
                    <span className="break-words text-[#6B7280]">
                      {explanation}
                    </span>
                  ) : (
                    <span className="text-[#9CA3AF]">—</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------- page

export default function EvaluatePage() {
  const [sample] = useState(() => buildSampleTrajectory())
  const [trajectoryInput, setTrajectoryInput] = useState('')
  const [weightingStrategy, setWeightingStrategy] = useState('equal')
  const [includeExplanations, setIncludeExplanations] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)

  const loadSample = () => {
    setTrajectoryInput(JSON.stringify(sample, null, 2))
  }

  const runEvaluation = async () => {
    let parsed
    try {
      parsed = JSON.parse(trajectoryInput)
    } catch (e) {
      setError(`Invalid JSON: ${e.message}`)
      setResults(null)
      return
    }

    setIsLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await evaluateTrajectory(
        parsed,
        includeExplanations,
        weightingStrategy,
      )
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setIsLoading(false)
    }
  }

  const runDisabled = isLoading || trajectoryInput.trim() === ''

  const hasResults = results && !isLoading && !error
  const compositeScore = hasResults ? results.composite_score ?? 0 : 0
  const compositeTier = tierOf(compositeScore)

  const explanationsObj = results?.explanations || {}
  const showExplanations =
    hasResults && Object.keys(explanationsObj).length > 0

  const isSaved =
    hasResults &&
    results.trajectory_id &&
    results.evaluation_id &&
    results.trajectory_id !== 'unsaved' &&
    results.evaluation_id !== 'unsaved'

  return (
    <div>
      <div className="border-b border-[#E5E7EB] bg-white px-6 pt-4">
        <div className="text-lg font-semibold text-[#111827]">
          Evaluate trajectory
        </div>
        <div className="mt-0.5 text-[13px] text-[#6B7280]">
          Submit an agent execution trace and receive metric scores with
          explanations
        </div>
        <div className="mt-3.5 flex gap-0">
          <div className="-mb-px border-b-2 border-[#2563EB] px-4 py-2 text-[13px] font-medium text-[#2563EB]">
            Single evaluation
          </div>
        </div>
      </div>

      <div className="px-6 py-5">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Left: input panel */}
          <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
            <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
              <span className="text-[13px] font-semibold text-[#374151]">
                Trajectory JSON
              </span>
              <button
                type="button"
                onClick={loadSample}
                className="rounded-md border border-[#E5E7EB] bg-white px-2.5 py-1 text-[11px] font-medium text-[#374151]"
              >
                Load sample
              </button>
            </div>
            <div className="p-4">
              <textarea
                value={trajectoryInput}
                onChange={(e) => setTrajectoryInput(e.target.value)}
                rows={10}
                spellCheck={false}
                placeholder='{"trajectory_id": "...", "task": "...", "steps": [...]}'
                className="w-full resize-none rounded-md border border-[#E5E7EB] p-2.5 font-mono text-[11px] text-[#374151] outline-none focus:border-[#2563EB]"
              />
              <div className="mt-2.5 flex flex-wrap items-center gap-2">
                <select
                  value={weightingStrategy}
                  onChange={(e) => setWeightingStrategy(e.target.value)}
                  className="rounded-md border border-[#E5E7EB] bg-white px-2.5 py-1.5 text-xs text-[#374151] outline-none"
                >
                  <option value="equal">Equal weights</option>
                  <option value="research">Research weights</option>
                </select>
                <label className="flex items-center gap-1.5 whitespace-nowrap text-xs text-[#6B7280]">
                  <input
                    type="checkbox"
                    checked={includeExplanations}
                    onChange={(e) => setIncludeExplanations(e.target.checked)}
                  />
                  Include explanations
                </label>
                <button
                  type="button"
                  onClick={runEvaluation}
                  disabled={runDisabled}
                  className="ml-auto whitespace-nowrap rounded-md border border-[#2563EB] bg-[#2563EB] px-3 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Run evaluation
                </button>
              </div>
              <div className="mt-2 text-[11px] text-[#9CA3AF]">
                trajectory_id must be a valid UUID. Use crypto.randomUUID() to
                generate one.
              </div>
            </div>
          </div>

          {/* Right: results panel */}
          <div>
            {isLoading && (
              <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
                <Spinner />
              </div>
            )}

            {error && !isLoading && (
              <ErrorBox message={error} onRetry={() => setError(null)} />
            )}

            {hasResults && (
              <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
                <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                  <span className="text-[13px] font-semibold text-[#374151]">
                    Evaluation results
                  </span>
                  <div
                    className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 ${BADGE_CLASS[compositeTier]}`}
                  >
                    <span className="text-xl font-bold leading-none">
                      {formatPct(compositeScore)}
                    </span>
                    <span className="text-xs text-[#6B7280]">composite</span>
                  </div>
                </div>
                <div className="p-4">
                  <div className="flex flex-col gap-2.5">
                    {METRIC_ORDER.filter((key) => key in results.scores).map(
                      (key) => (
                        <MetricBar
                          key={key}
                          name={key}
                          score={results.scores[key]}
                        />
                      ),
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Metric explanations (full width) */}
        {showExplanations && (
          <div className="mt-4 rounded-[10px] border border-[#E5E7EB] bg-white">
            <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
              <span className="text-[13px] font-semibold text-[#374151]">
                Metric explanations
              </span>
            </div>
            <div>
              <ExplanationTable
                scores={results.scores}
                explanations={explanationsObj}
                metricErrors={results.metric_errors}
              />
            </div>
          </div>
        )}

        {/* Persistence status */}
        {hasResults && (
          <div className="mt-3 text-[11px] text-[#9CA3AF]">
            {isSaved ? (
              <>
                trajectory_id: {results.trajectory_id} · evaluation_id:{' '}
                {results.evaluation_id} · Saved to database
              </>
            ) : (
              <>Not persisted — database unavailable; results not saved.</>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
