import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getTrajectory } from '../api'

// Evaluation metric keys as they appear (flat) in the merged
// GET /trajectories/{id} response. The DB persists the column `consistency`
// (not `multi_turn_consistency`); both are mapped so the page is robust to
// either shape. Composite is included for completeness but is frequently
// absent (not persisted by the current Week 4 contract) — handled gracefully.
const METRIC_LABELS = {
  task_completion: 'Task completion',
  tool_call_fidelity: 'Tool call fidelity',
  step_efficiency: 'Step efficiency',
  reasoning_coherence: 'Reasoning coherence',
  hallucination_score: 'Hallucination score',
  recovery_rate: 'Recovery rate',
  consistency: 'Multi-turn consistency',
  multi_turn_consistency: 'Multi-turn consistency',
}

const METRIC_ORDER = [
  'task_completion',
  'tool_call_fidelity',
  'step_efficiency',
  'reasoning_coherence',
  'hallucination_score',
  'recovery_rate',
  'consistency',
  'multi_turn_consistency',
]

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

// ---------------------------------------------------------------- helpers

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

function formatPct(score) {
  const pct = score * 100
  const rounded = Math.round(pct * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}%` : `${rounded.toFixed(1)}%`
}

function presentMetricKeys(data) {
  const seen = new Set()
  const keys = []
  for (const key of METRIC_ORDER) {
    if (typeof data[key] === 'number') {
      const label = METRIC_LABELS[key]
      if (seen.has(label)) continue
      seen.add(label)
      keys.push(key)
    }
  }
  return keys
}

// ---------------------------------------------------------------- subcomponents

function Spinner({ label }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#E5E7EB] border-t-[#2563EB]" />
      <div className="text-sm font-medium text-[#374151]">{label}</div>
    </div>
  )
}

function MetricBar({ name, score }) {
  const tier = tierOf(score)
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-[140px] flex-shrink-0 text-xs text-[#6B7280]">
        {METRIC_LABELS[name] || name}
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

function MetaField({ label, children }) {
  return (
    <div className="text-xs">
      <span className="text-[#9CA3AF]">{label}: </span>
      {children}
    </div>
  )
}

const STEP_BADGE = {
  llm_call: { label: 'LLM CALL', cls: 'bg-[#EFF6FF] text-[#1D4ED8]' },
  tool_call: { label: 'TOOL CALL', cls: 'bg-[#F5F3FF] text-[#5B21B6]' },
  error: { label: 'ERROR', cls: 'bg-[#FEF2F2] text-[#B91C1C]' },
}

function stringify(value) {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function MonoBox({ value }) {
  return (
    <div className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded bg-[#F9FAFB] p-2 font-mono text-[11px] text-[#374151]">
      {value}
    </div>
  )
}

function StepCard({ step, index, expanded, onToggle }) {
  const type = step.type || 'llm_call'
  const badge = STEP_BADGE[type] || {
    label: String(type).toUpperCase(),
    cls: 'bg-[#F3F4F6] text-[#374151]',
  }
  const isTool = type === 'tool_call'
  const isError = type === 'error'
  const idx = step.step_index ?? index
  const duration = step.duration_ms
  const tokens = step.tokens

  const meta = []
  if (typeof duration === 'number') meta.push(`${duration}ms`)
  if (type === 'llm_call' && typeof tokens === 'number')
    meta.push(`${tokens} tok`)

  const inputText = stringify(step.input)
  const outputText = stringify(step.output)
  const errorText = stringify(step.error ?? step.output ?? step.error_summary)

  return (
    <div className="overflow-hidden rounded-lg border border-[#E5E7EB]">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2.5 bg-[#FAFAFA] px-3 py-2.5 text-left"
        aria-expanded={expanded}
      >
        <span className="w-5 flex-shrink-0 text-[11px] text-[#9CA3AF]">
          {idx}
        </span>
        <span
          className={`flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${badge.cls}`}
        >
          {badge.label}
        </span>
        {isTool && step.tool_name && (
          <span className="truncate text-xs text-[#6B7280]">
            {step.tool_name}
          </span>
        )}
        {meta.length > 0 && (
          <span className="ml-auto flex-shrink-0 text-[11px] text-[#9CA3AF]">
            {meta.join(' · ')}
          </span>
        )}
        <span
          className={`flex-shrink-0 text-[#9CA3AF] ${meta.length > 0 ? 'ml-2.5' : 'ml-auto'}`}
        >
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-[#F3F4F6] px-3 py-2.5">
          {isError ? (
            <div className="mb-1.5">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
                Error
              </div>
              <div className="whitespace-pre-wrap break-words rounded border border-[#FECACA] bg-[#FEF2F2] p-2 text-xs text-[#DC2626]">
                {errorText || '—'}
              </div>
            </div>
          ) : (
            <>
              <div className="mb-1.5">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
                  Input
                </div>
                {isTool ? (
                  <MonoBox value={inputText || '—'} />
                ) : (
                  <div className="whitespace-pre-wrap break-words text-xs leading-relaxed text-[#374151]">
                    {inputText || '—'}
                  </div>
                )}
              </div>
              <div>
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF]">
                  Output
                </div>
                {isTool ? (
                  <MonoBox value={outputText || '—'} />
                ) : (
                  <div className="whitespace-pre-wrap break-words text-xs leading-relaxed text-[#374151]">
                    {outputText || '—'}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- search mode

function SearchMode() {
  const navigate = useNavigate()
  const [value, setValue] = useState('')
  const [validationError, setValidationError] = useState(null)

  const submit = (e) => {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) {
      setValidationError('Please enter a trajectory UUID.')
      return
    }
    if (!UUID_RE.test(trimmed)) {
      setValidationError(
        'That does not look like a valid UUID. Expected format: 8-4-4-4-12 hex characters.',
      )
      return
    }
    setValidationError(null)
    navigate(`/explorer/${trimmed}`)
  }

  return (
    <div>
      <div className="border-b border-[#E5E7EB] bg-white px-6 pt-4">
        <div className="text-lg font-semibold text-[#111827]">
          Trajectory explorer
        </div>
        <div className="mt-0.5 pb-4 text-[13px] text-[#6B7280]">
          Inspect agent execution steps, tool calls, and per-step debugging
          information
        </div>
      </div>

      <div className="px-6 py-5">
        <div className="mx-auto max-w-xl rounded-[10px] border border-[#E5E7EB] bg-white p-6">
          <form onSubmit={submit} className="flex flex-col gap-3">
            <label className="text-[13px] font-semibold text-[#374151]">
              Load a trajectory
            </label>
            <div className="flex gap-2">
              <input
                value={value}
                onChange={(e) => {
                  setValue(e.target.value)
                  if (validationError) setValidationError(null)
                }}
                placeholder="Enter trajectory UUID"
                className="w-full rounded-md border border-[#E5E7EB] px-2.5 py-2 font-mono text-[13px] text-[#374151] outline-none focus:border-[#2563EB]"
              />
              <button
                type="submit"
                className="whitespace-nowrap rounded-md border border-[#2563EB] bg-[#2563EB] px-4 py-2 text-xs font-medium text-white"
              >
                Load
              </button>
            </div>
            {validationError && (
              <div className="text-xs text-[#DC2626]">{validationError}</div>
            )}
            <div className="text-[11px] text-[#9CA3AF]">
              Trajectory UUIDs are returned by POST /evaluate and stored in
              PostgreSQL.
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- view mode

function ViewMode({ id }) {
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(() => new Set())

  useEffect(() => {
    let active = true
    setIsLoading(true)
    setError(null)
    setData(null)
    setExpanded(new Set())
    getTrajectory(id)
      .then((res) => {
        if (active) setData(res)
      })
      .catch((e) => {
        if (active) setError(e.message)
      })
      .finally(() => {
        if (active) setIsLoading(false)
      })
    return () => {
      active = false
    }
  }, [id])

  const steps = useMemo(
    () => (Array.isArray(data?.steps) ? data.steps : []),
    [data],
  )
  const metricKeys = useMemo(
    () => (data ? presentMetricKeys(data) : []),
    [data],
  )
  const hasEvaluation =
    !!data && (metricKeys.length > 0 || data.evaluated_at != null)
  const hasComposite = !!data && typeof data.composite_score === 'number'

  const toggleStep = (key) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }
  const expandAll = () => setExpanded(new Set(steps.map((s, i) => s.step_index ?? i)))
  const collapseAll = () => setExpanded(new Set())

  const groundTruth = data?.ground_truth
  const finalAnswer = data?.final_answer
  const answerMatches =
    typeof groundTruth === 'string' &&
    typeof finalAnswer === 'string' &&
    groundTruth.trim() !== '' &&
    finalAnswer.toLowerCase().includes(groundTruth.toLowerCase())

  return (
    <div>
      <div className="border-b border-[#E5E7EB] bg-white px-6 pt-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-[#111827]">
              Trajectory explorer
            </div>
            <div className="mt-0.5 text-[13px] text-[#6B7280]">
              Inspect agent execution steps, tool calls, and per-step
              debugging information
            </div>
          </div>
          <button
            type="button"
            onClick={() => navigate('/explorer')}
            className="rounded-md border border-[#E5E7EB] bg-white px-3 py-1.5 text-xs font-medium text-[#374151]"
          >
            New search
          </button>
        </div>
        <div className="mt-3 pb-2 font-mono text-[11px] text-[#9CA3AF]">
          {id}
        </div>
      </div>

      <div className="px-6 py-5">
        {isLoading && (
          <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
            <Spinner label="Loading trajectory..." />
          </div>
        )}

        {error && !isLoading && (
          <div className="mx-auto max-w-xl rounded-lg border border-[#FECACA] bg-[#FEF2F2] p-5">
            <div className="mb-1 text-sm font-medium text-[#B91C1C]">
              Trajectory not found
            </div>
            <div className="mb-3 break-words text-xs text-[#DC2626]">
              The UUID may be incorrect or the trajectory may not be stored in
              the database.
            </div>
            <button
              type="button"
              onClick={() => navigate('/explorer')}
              className="rounded-md border border-[#E5E7EB] bg-white px-3 py-1.5 text-xs font-medium text-[#374151]"
            >
              Back
            </button>
          </div>
        )}

        {data && !isLoading && !error && (
          <div className="flex flex-col gap-4">
            {/* Metadata card */}
            <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
              <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                <span className="text-[13px] font-semibold text-[#374151]">
                  Trajectory
                </span>
                {hasComposite && (
                  <span
                    className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${PILL_CLASS[tierOf(data.composite_score)]}`}
                  >
                    {formatPct(data.composite_score)} composite
                  </span>
                )}
              </div>
              <div className="grid grid-cols-1 gap-x-6 gap-y-2.5 p-4 sm:grid-cols-2 lg:grid-cols-3">
                <MetaField label="Task">
                  <span className="font-medium text-[#374151]">
                    {data.task ?? '—'}
                  </span>
                </MetaField>
                <MetaField label="Agent">
                  <span className="text-[#374151]">{data.agent_id ?? '—'}</span>
                </MetaField>
                <MetaField label="Ground truth">
                  {groundTruth ? (
                    <span
                      className={
                        answerMatches
                          ? 'font-medium text-[#059669]'
                          : 'text-[#374151]'
                      }
                    >
                      {groundTruth}
                    </span>
                  ) : (
                    <span className="text-[#9CA3AF]">—</span>
                  )}
                </MetaField>
                <MetaField label="Final answer">
                  <span className="break-words text-[#374151]">
                    {finalAnswer ?? '—'}
                  </span>
                </MetaField>
                <MetaField label="Total tokens">
                  <span className="text-[#374151]">
                    {typeof data.total_tokens === 'number'
                      ? data.total_tokens
                      : '—'}
                  </span>
                </MetaField>
                <MetaField label="Duration">
                  <span className="text-[#374151]">
                    {typeof data.total_duration_ms === 'number'
                      ? `${data.total_duration_ms}ms`
                      : '—'}
                  </span>
                </MetaField>
                <MetaField label="Timestamp">
                  <span className="text-[#374151]">
                    {data.timestamp ?? '—'}
                  </span>
                </MetaField>
                {data.evaluated_at && (
                  <MetaField label="Evaluated">
                    <span className="text-[#374151]">{data.evaluated_at}</span>
                  </MetaField>
                )}
              </div>
            </div>

            {/* Score bars */}
            {hasEvaluation && metricKeys.length > 0 && (
              <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
                <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                  <span className="text-[13px] font-semibold text-[#374151]">
                    Metric scores
                  </span>
                </div>
                <div className="flex flex-col gap-2 p-4">
                  {metricKeys.map((key) => (
                    <MetricBar key={key} name={key} score={data[key]} />
                  ))}
                </div>
              </div>
            )}

            {/* No-evaluation informational note (not an error) */}
            {!hasEvaluation && (
              <div className="rounded-lg border border-[#FED7AA] bg-[#FFF7ED] p-3.5">
                <div className="text-xs leading-relaxed text-[#92400E]">
                  No evaluation data found for this trajectory. Submit it via
                  POST /evaluate to generate scores.
                </div>
              </div>
            )}

            {/* Steps */}
            <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
              <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                <span className="text-[13px] font-semibold text-[#374151]">
                  Steps{' '}
                  <span className="font-normal text-[#9CA3AF]">
                    ({steps.length})
                  </span>
                </span>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={expandAll}
                    disabled={steps.length === 0}
                    className="rounded-md border border-[#E5E7EB] bg-white px-2.5 py-1 text-[11px] font-medium text-[#374151] disabled:opacity-50"
                  >
                    Expand all
                  </button>
                  <button
                    type="button"
                    onClick={collapseAll}
                    disabled={steps.length === 0}
                    className="rounded-md border border-[#E5E7EB] bg-white px-2.5 py-1 text-[11px] font-medium text-[#374151] disabled:opacity-50"
                  >
                    Collapse all
                  </button>
                </div>
              </div>
              <div className="p-4">
                {steps.length === 0 ? (
                  <div className="py-6 text-center text-xs text-[#9CA3AF]">
                    This trajectory has no recorded steps.
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {steps.map((step, i) => {
                      const key = step.step_index ?? i
                      return (
                        <StepCard
                          key={key}
                          step={step}
                          index={i}
                          expanded={expanded.has(key)}
                          onToggle={() => toggleStep(key)}
                        />
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- page

export default function TrajectoryPage() {
  const { id } = useParams()
  return id ? <ViewMode id={id} /> : <SearchMode />
}
