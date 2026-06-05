import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { getLeaderboard } from '../api'

// Known metric keys (as stored in avg_scores) with a fixed, deterministic
// color palette. Composite is handled separately (thicker, dark navy line).
const METRICS = [
  { key: 'task_completion', label: 'Task completion', color: '#2563EB' },
  { key: 'tool_call_fidelity', label: 'Tool fidelity', color: '#7C3AED' },
  { key: 'step_efficiency', label: 'Step efficiency', color: '#059669' },
  { key: 'reasoning_coherence', label: 'Reasoning coherence', color: '#D97706' },
  { key: 'hallucination_score', label: 'Hallucination', color: '#DC2626' },
  { key: 'recovery_rate', label: 'Recovery', color: '#0891B2' },
  { key: 'multi_turn_consistency', label: 'Consistency', color: '#DB2777' },
]
const METRIC_LABELS = Object.fromEntries(
  METRICS.map((m) => [m.key, m.label]),
)
const COMPOSITE_COLOR = '#1A1D23'
const REGRESSION_DELTA = 0.05

// ---------------------------------------------------------------- helpers

function formatPct(score) {
  const pct = score * 100
  const rounded = Math.round(pct * 10) / 10
  return Number.isInteger(rounded) ? `${rounded}%` : `${rounded.toFixed(1)}%`
}

// Defensive timestamp parsing — invalid dates never break rendering.
function parseTime(iso) {
  if (!iso) return NaN
  const t = Date.parse(iso)
  return Number.isNaN(t) ? NaN : t
}

function formatRunLabel(iso, index) {
  const t = parseTime(iso)
  if (Number.isNaN(t)) return `Run ${index + 1}`
  const d = new Date(t)
  const date = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  const time = d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
  return `${date} ${time}`
}

function getComposite(run) {
  if (typeof run.composite_score === 'number') return run.composite_score
  const nested = run.avg_scores?.composite_score
  return typeof nested === 'number' ? nested : null
}

// Numeric metric keys actually present in a run's avg_scores (excludes composite).
function metricKeySet(run) {
  const scores = run.avg_scores || {}
  return new Set(
    METRICS.map((m) => m.key).filter((k) => typeof scores[k] === 'number'),
  )
}

// ---------------------------------------------------------------- subcomponents

function Spinner() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#E5E7EB] border-t-[#2563EB]" />
      <div className="text-sm font-medium text-[#374151]">
        Loading score history...
      </div>
    </div>
  )
}

function NotEnoughData({ agentId }) {
  return (
    <div className="rounded-[10px] border border-[#E5E7EB] bg-white px-6 py-10 text-center">
      <div className="mb-2 text-3xl">📈</div>
      <div className="mb-1 text-sm font-medium text-[#374151]">
        Not enough data for regression analysis
      </div>
      <div className="mx-auto max-w-md text-[13px] text-[#6B7280]">
        Run at least 2 benchmark evaluations for agent{' '}
        <span className="font-medium text-[#374151]">{agentId}</span>.
      </div>
      <div className="mx-auto mt-2 max-w-md text-[11px] text-[#9CA3AF]">
        Tip: avoid using agent_type=&apos;mock&apos; as these are excluded from
        the leaderboard.
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- page

export default function RegressionPage() {
  const [agentInput, setAgentInput] = useState('')
  const [loadedAgent, setLoadedAgent] = useState(null)
  const [runs, setRuns] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [validationError, setValidationError] = useState(null)

  const load = async (e) => {
    e.preventDefault()
    const id = agentInput.trim()
    if (!id) {
      setValidationError('Please enter an agent ID.')
      return
    }
    setValidationError(null)
    setIsLoading(true)
    setError(null)
    setRuns([])
    setLoadedAgent(id)
    try {
      const data = await getLeaderboard(null, false)
      const list = Array.isArray(data) ? data : []
      const matching = list
        .filter((r) => String(r.agent_id ?? '') === id)
        .slice()
        .sort((a, b) => {
          const ta = parseTime(a.completed_at)
          const tb = parseTime(b.completed_at)
          const va = Number.isNaN(ta) ? 0 : ta
          const vb = Number.isNaN(tb) ? 0 : tb
          return va - vb
        })
      setRuns(matching)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  // Compatibility + drawable-metric analysis (only meaningful with >= 2 runs).
  const keyStructures = new Set(
    runs.map((r) => JSON.stringify([...metricKeySet(r)].sort())),
  )
  const agentTypes = new Set(runs.map((r) => r.agent_type || 'unknown'))
  const incompatible = keyStructures.size > 1 || agentTypes.size > 1

  // Metrics that exist consistently across every run (intersection).
  const consistentMetrics = METRICS.filter((m) =>
    runs.every((r) => typeof (r.avg_scores || {})[m.key] === 'number'),
  )
  const drawComposite =
    runs.length > 0 && runs.every((r) => typeof getComposite(r) === 'number')

  const chartData = runs.map((run, i) => {
    const point = { name: formatRunLabel(run.completed_at, i) }
    for (const m of METRICS) {
      const v = run.avg_scores?.[m.key]
      point[m.key] = typeof v === 'number' ? v : null
    }
    point.composite_score = getComposite(run)
    return point
  })

  // Regression alerts: compare ONLY the latest run and the run immediately
  // before it. Never computed with fewer than 2 runs.
  let alerts = []
  if (runs.length >= 2) {
    const prev = runs[runs.length - 2]
    const latest = runs[runs.length - 1]
    const candidates = [
      ...consistentMetrics.map((m) => ({ key: m.key, label: m.label })),
      ...(drawComposite
        ? [{ key: 'composite_score', label: 'Composite' }]
        : []),
    ]
    for (const c of candidates) {
      const prevVal =
        c.key === 'composite_score'
          ? getComposite(prev)
          : prev.avg_scores?.[c.key]
      const curVal =
        c.key === 'composite_score'
          ? getComposite(latest)
          : latest.avg_scores?.[c.key]
      if (typeof prevVal === 'number' && typeof curVal === 'number') {
        const delta = curVal - prevVal
        if (delta < -REGRESSION_DELTA) {
          alerts.push({
            label: c.label,
            prev: prevVal,
            cur: curVal,
            delta,
          })
        }
      }
    }
  }

  const hasChart = runs.length >= 2

  return (
    <div>
      <div className="border-b border-[#E5E7EB] bg-white px-6 pt-4">
        <div className="text-lg font-semibold text-[#111827]">
          Regression monitor
        </div>
        <div className="mt-0.5 pb-4 text-[13px] text-[#6B7280]">
          Track evaluation score changes across benchmark runs for a given
          agent
        </div>
      </div>

      <div className="px-6 py-5">
        {/* Search */}
        <form onSubmit={load} className="mb-4 flex flex-wrap items-center gap-2">
          <input
            value={agentInput}
            onChange={(e) => {
              setAgentInput(e.target.value)
              if (validationError) setValidationError(null)
            }}
            placeholder="Enter agent ID..."
            className="w-full max-w-[260px] rounded-md border border-[#E5E7EB] px-2.5 py-1.5 text-[13px] text-[#374151] outline-none focus:border-[#2563EB]"
          />
          <button
            type="submit"
            className="whitespace-nowrap rounded-md border border-[#2563EB] bg-[#2563EB] px-4 py-1.5 text-xs font-medium text-white"
          >
            Load history
          </button>
          {validationError && (
            <span className="text-xs text-[#DC2626]">{validationError}</span>
          )}
        </form>

        {/* Standing comparability note (mirrors approved design).
            Only meaningful once there are >= 2 runs to compare, so it is
            gated behind hasChart and never shown alongside the
            "Not enough data" state. */}
        {!isLoading && !error && hasChart && (
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-[#FED7AA] bg-[#FFF7ED] px-3.5 py-2.5">
            <span className="mt-0.5 flex-shrink-0 text-[#F59E0B]">⚠</span>
            <div className="text-xs leading-relaxed text-[#92400E]">
              <strong>Comparability:</strong> regression analysis is only
              meaningful across runs with the same evaluation configuration
              (same metric set and agent type). Runs with different
              configurations are charted but excluded from regression
              calculations.
            </div>
          </div>
        )}

        {isLoading && (
          <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
            <Spinner />
          </div>
        )}

        {error && !isLoading && (
          <div className="rounded-lg border border-[#FECACA] bg-[#FEF2F2] p-4">
            <div className="mb-2 text-sm font-medium text-[#B91C1C]">
              Failed to load score history
            </div>
            <div className="break-words text-xs text-[#DC2626]">{error}</div>
          </div>
        )}

        {/* No agent loaded yet → search UI only (nothing more below) */}

        {!isLoading && !error && loadedAgent && !hasChart && (
          <NotEnoughData agentId={loadedAgent} />
        )}

        {!isLoading && !error && hasChart && (
          <div className="flex flex-col gap-4">
            {incompatible && (
              <div className="flex items-start gap-2 rounded-lg border border-[#FDE68A] bg-[#FFFBEB] px-3.5 py-2.5">
                <span className="mt-0.5 flex-shrink-0 text-[#D97706]">⚠</span>
                <div className="text-xs leading-relaxed text-[#92400E]">
                  Some benchmark runs may have used different evaluation
                  configurations. Regression lines are only drawn between
                  compatible runs.
                </div>
              </div>
            )}

            {/* Chart */}
            <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
              <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                <span className="text-[13px] font-semibold text-[#374151]">
                  Score history{' '}
                  <span className="font-normal text-[#9CA3AF]">
                    ({runs.length} runs)
                  </span>
                </span>
              </div>
              <div className="p-4">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart
                    data={chartData}
                    margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                  >
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 11, fill: '#6B7280' }}
                      stroke="#E5E7EB"
                    />
                    <YAxis
                      domain={[0, 1]}
                      ticks={[0, 0.25, 0.5, 0.75, 1]}
                      tickFormatter={(v) => `${Math.round(v * 100)}%`}
                      tick={{ fontSize: 11, fill: '#6B7280' }}
                      stroke="#E5E7EB"
                    />
                    <Tooltip
                      formatter={(value) =>
                        typeof value === 'number' ? formatPct(value) : '—'
                      }
                    />
                    <ReferenceLine
                      y={0.7}
                      stroke="#9CA3AF"
                      strokeDasharray="4 4"
                      label={{
                        value: 'Quality threshold',
                        position: 'insideTopRight',
                        fontSize: 10,
                        fill: '#9CA3AF',
                      }}
                    />
                    {consistentMetrics.map((m) => (
                      <Line
                        key={m.key}
                        type="monotone"
                        dataKey={m.key}
                        name={m.label}
                        stroke={m.color}
                        strokeWidth={1.5}
                        dot={{ r: 2 }}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                    ))}
                    {drawComposite && (
                      <Line
                        type="monotone"
                        dataKey="composite_score"
                        name="Composite"
                        stroke={COMPOSITE_COLOR}
                        strokeWidth={2.5}
                        dot={{ r: 3 }}
                        connectNulls={false}
                        isAnimationActive={false}
                      />
                    )}
                    <Legend
                      wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Regression alerts */}
            <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
              <div className="flex items-center justify-between border-b border-[#F3F4F6] px-4 py-3.5">
                <span className="text-[13px] font-semibold text-[#374151]">
                  Regression alerts
                </span>
                <span className="text-[11px] text-[#9CA3AF]">
                  latest vs previous run
                </span>
              </div>
              <div className="p-4">
                {alerts.length === 0 ? (
                  <div className="py-2 text-[13px] text-[#6B7280]">
                    No significant regressions detected across the last two
                    runs.
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {alerts.map((a) => (
                      <div
                        key={a.label}
                        className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-[#FECACA] bg-[#FEF2F2] px-3 py-2 text-xs"
                      >
                        <span className="font-medium text-[#374151]">
                          {a.label}
                        </span>
                        <span className="text-[#6B7280]">
                          {formatPct(a.prev)} → {formatPct(a.cur)}
                        </span>
                        <span className="ml-auto font-semibold text-[#DC2626]">
                          ▼ {formatPct(Math.abs(a.delta))}
                        </span>
                      </div>
                    ))}
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
