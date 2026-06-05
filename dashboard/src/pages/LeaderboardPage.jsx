import { useEffect, useMemo, useState } from 'react'
import { getLeaderboard } from '../api'

// ---------------------------------------------------------------- threshold coloring
// Identical thresholds/colors to the Evaluate and Explorer pages.

function tierOf(score) {
  const pct = score * 100
  if (pct >= 70) return 'green'
  if (pct >= 40) return 'amber'
  return 'red'
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

// ---------------------------------------------------------------- column model
//
// Score columns read from the nested `avg_scores` dict produced by the
// benchmark runner aggregate. `composite_score` is also surfaced at the top
// level of each LeaderboardEntry, so we prefer the top-level value and fall
// back to avg_scores. Consistency is stored under `multi_turn_consistency`
// (the metric name); `consistency` is accepted as a defensive fallback.

const COLUMNS = [
  { key: 'rank', label: 'Rank', type: 'rank', sortable: false },
  { key: 'agent_id', label: 'Agent ID', type: 'text', sortable: true },
  { key: 'benchmark_name', label: 'Benchmark', type: 'text', sortable: true },
  { key: 'composite_score', label: 'Composite', type: 'score', sortable: true },
  { key: 'task_completion', label: 'Task Completion', type: 'score', sortable: true },
  { key: 'tool_call_fidelity', label: 'Tool Fidelity', type: 'score', sortable: true },
  { key: 'step_efficiency', label: 'Step Efficiency', type: 'score', sortable: true },
  { key: 'reasoning_coherence', label: 'Reasoning Coherence', type: 'score', sortable: true },
  { key: 'hallucination_score', label: 'Hallucination', type: 'score', sortable: true },
  { key: 'recovery_rate', label: 'Recovery', type: 'score', sortable: true },
  { key: 'multi_turn_consistency', label: 'Consistency', type: 'score', sortable: true },
  { key: 'completed_at', label: 'Completed', type: 'date', sortable: true },
]

// Per-column value extraction from a raw LeaderboardEntry.
function getValue(entry, col) {
  switch (col.key) {
    case 'agent_id':
      return entry.agent_id ?? null
    case 'benchmark_name':
      return entry.benchmark_name ?? null
    case 'composite_score': {
      const top = entry.composite_score
      if (typeof top === 'number') return top
      const nested = entry.avg_scores?.composite_score
      return typeof nested === 'number' ? nested : null
    }
    case 'multi_turn_consistency': {
      const s = entry.avg_scores || {}
      const v = s.multi_turn_consistency ?? s.consistency
      return typeof v === 'number' ? v : null
    }
    case 'completed_at':
      return entry.completed_at ?? null
    default: {
      // remaining score columns live in avg_scores keyed by metric name
      const v = entry.avg_scores?.[col.key]
      return typeof v === 'number' ? v : null
    }
  }
}

// Sticky-left frozen block: Rank + Agent ID + Composite. Offsets must be
// contiguous (left = sum of prior sticky widths) so the three pin into one
// solid block when the metric columns scroll horizontally beneath them.
const STICKY = {
  rank: { left: 0, width: 56 },
  agent_id: { left: 56, width: 176 },
  composite_score: { left: 232, width: 104 },
}

function stickyStyle(key) {
  const s = STICKY[key]
  if (!s) return undefined
  return { left: s.left, width: s.width, minWidth: s.width }
}

// ---------------------------------------------------------------- sorting

function compareValues(a, b, type, dir) {
  const aMissing = a === null || a === undefined || a === ''
  const bMissing = b === null || b === undefined || b === ''
  // Missing values always sink to the bottom regardless of direction.
  if (aMissing && bMissing) return 0
  if (aMissing) return 1
  if (bMissing) return -1

  let cmp
  if (type === 'text') {
    cmp = String(a).localeCompare(String(b))
  } else if (type === 'date') {
    const ta = Date.parse(a)
    const tb = Date.parse(b)
    const va = Number.isNaN(ta) ? 0 : ta
    const vb = Number.isNaN(tb) ? 0 : tb
    cmp = va - vb
  } else {
    // numeric (score) — coerce to Number so string-typed numbers still sort numerically
    cmp = Number(a) - Number(b)
  }
  return dir === 'asc' ? cmp : -cmp
}

// ---------------------------------------------------------------- subcomponents

function Spinner() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#E5E7EB] border-t-[#2563EB]" />
      <div className="text-sm font-medium text-[#374151]">
        Loading leaderboard...
      </div>
    </div>
  )
}

function ScoreCell({ value }) {
  if (typeof value !== 'number') {
    return <span className="text-[#9CA3AF]">—</span>
  }
  const tier = tierOf(value)
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-semibold ${PILL_CLASS[tier]}`}
    >
      {formatPct(value)}
    </span>
  )
}

function EmptyLeaderboard({ showMock }) {
  const curl = `curl -X POST http://localhost:8000/benchmark/run -H "Content-Type: application/json" -d '{"benchmark": "factual_research", "agent_id": "my_agent", "max_tasks": 5}'`
  return (
    <div className="rounded-[10px] border border-[#E5E7EB] bg-white px-6 py-10 text-center">
      <div className="mb-2 text-3xl">📊</div>
      <div className="mb-1 text-sm font-medium text-[#374151]">
        No benchmark results yet
      </div>
      <div className="mx-auto mb-1 max-w-md text-[13px] text-[#6B7280]">
        Run a real-agent benchmark via the API to populate rankings here.
      </div>
      <div className="mx-auto mb-4 max-w-md text-[11px] text-[#9CA3AF]">
        {showMock
          ? 'Mock agent runs do not appear on the leaderboard (the backend filters them at the database layer).'
          : 'Mock agent runs are excluded by default.'}
      </div>
      <pre className="mx-auto max-w-2xl overflow-x-auto whitespace-pre rounded-md bg-[#F4F3EC] p-3 text-left font-mono text-[11px] text-[#374151]">
        {curl}
      </pre>
    </div>
  )
}

// ---------------------------------------------------------------- page

export default function LeaderboardPage() {
  const [showMock, setShowMock] = useState(false)
  const [agentFilter, setAgentFilter] = useState('')
  const [entries, setEntries] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  // Default sort: composite descending.
  const [sortKey, setSortKey] = useState('composite_score')
  const [sortDir, setSortDir] = useState('desc')

  // Re-fetch whenever the mock toggle changes. excludeMock = !showMock:
  // "Show mock" unchecked => hide mock => exclude_mock=true (backend contract).
  useEffect(() => {
    let active = true
    setIsLoading(true)
    setError(null)
    getLeaderboard(null, !showMock)
      .then((data) => {
        if (active) setEntries(Array.isArray(data) ? data : [])
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
  }, [showMock])

  const filtered = useMemo(() => {
    const q = agentFilter.trim().toLowerCase()
    if (!q) return entries
    return entries.filter((e) =>
      String(e.agent_id ?? '').toLowerCase().includes(q),
    )
  }, [entries, agentFilter])

  const sorted = useMemo(() => {
    const col = COLUMNS.find((c) => c.key === sortKey)
    if (!col) return filtered
    const copy = [...filtered]
    copy.sort((a, b) =>
      compareValues(getValue(a, col), getValue(b, col), col.type, sortDir),
    )
    return copy
  }, [filtered, sortKey, sortDir])

  const handleSort = (col) => {
    if (!col.sortable) return
    if (sortKey === col.key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(col.key)
      setSortDir('asc')
    }
  }

  const hasEntries = entries.length > 0
  const filterActive = agentFilter.trim() !== ''

  return (
    <div>
      <div className="border-b border-[#E5E7EB] bg-white px-6 pt-4">
        <div className="text-lg font-semibold text-[#111827]">
          Benchmark leaderboard
        </div>
        <div className="mt-0.5 pb-4 text-[13px] text-[#6B7280]">
          Ranked agent performance across benchmark runs
        </div>
      </div>

      <div className="px-6 py-5">
        {/* Filter bar */}
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <input
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value)}
            placeholder="Filter by agent ID"
            className="w-full max-w-[220px] rounded-md border border-[#E5E7EB] px-2.5 py-1.5 text-[13px] text-[#374151] outline-none focus:border-[#2563EB]"
          />
          <label className="ml-auto flex items-center gap-1.5 text-xs text-[#6B7280]">
            <input
              type="checkbox"
              checked={showMock}
              onChange={(e) => setShowMock(e.target.checked)}
            />
            Show mock agents
          </label>
        </div>

        {isLoading && (
          <div className="rounded-[10px] border border-[#E5E7EB] bg-white">
            <Spinner />
          </div>
        )}

        {error && !isLoading && (
          <div className="rounded-lg border border-[#FECACA] bg-[#FEF2F2] p-4">
            <div className="mb-2 text-sm font-medium text-[#B91C1C]">
              Failed to load leaderboard
            </div>
            <div className="mb-3 break-words text-xs text-[#DC2626]">
              {error}
            </div>
            <button
              type="button"
              onClick={() => setShowMock((v) => v)}
              className="rounded-md border border-[#E5E7EB] bg-white px-3 py-1.5 text-xs font-medium text-[#374151]"
            >
              Retry
            </button>
          </div>
        )}

        {!isLoading && !error && !hasEntries && (
          <EmptyLeaderboard showMock={showMock} />
        )}

        {!isLoading && !error && hasEntries && sorted.length === 0 && (
          <div className="rounded-[10px] border border-[#E5E7EB] bg-white px-6 py-10 text-center">
            <div className="mb-1 text-sm font-medium text-[#374151]">
              No agents matching this filter
            </div>
            <div className="text-[13px] text-[#6B7280]">
              No agent ID contains “{agentFilter.trim()}”. Clear the filter to
              see all {entries.length} result{entries.length === 1 ? '' : 's'}.
            </div>
          </div>
        )}

        {!isLoading && !error && hasEntries && sorted.length > 0 && (
          <div className="overflow-x-auto rounded-[10px] border border-[#E5E7EB] bg-white">
            <table className="w-full min-w-[1000px] border-collapse text-xs">
              <thead>
                <tr>
                  {COLUMNS.map((col) => {
                    const isSticky = col.key in STICKY
                    const isActive = sortKey === col.key
                    return (
                      <th
                        key={col.key}
                        onClick={() => handleSort(col)}
                        style={stickyStyle(col.key)}
                        className={[
                          'border-b border-[#F3F4F6] bg-[#FFFFFF] px-2.5 py-2 text-left text-[10px] font-semibold uppercase tracking-wide text-[#9CA3AF] whitespace-nowrap',
                          col.sortable ? 'cursor-pointer select-none' : '',
                          isSticky ? 'sticky z-20' : '',
                          col.key === 'composite_score'
                            ? 'border-l border-[#E5E7EB]'
                            : '',
                        ].join(' ')}
                      >
                        <span className="inline-flex items-center gap-1">
                          {col.label}
                          {isActive && col.sortable && (
                            <span className="text-[#2563EB]">
                              {sortDir === 'asc' ? '▲' : '▼'}
                            </span>
                          )}
                        </span>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {sorted.map((entry, i) => (
                  <tr
                    key={`${entry.agent_id}-${entry.benchmark_name}-${i}`}
                    className="group"
                  >
                    {COLUMNS.map((col) => {
                      const isSticky = col.key in STICKY
                      const stickyCls = isSticky
                        ? 'sticky z-10 bg-white group-hover:bg-[#FAFAFA]'
                        : 'group-hover:bg-[#FAFAFA]'
                      const baseCls =
                        'border-b border-[#F9FAFB] px-2.5 py-2.5 whitespace-nowrap'
                      const borderCls =
                        col.key === 'composite_score'
                          ? 'border-l border-[#E5E7EB]'
                          : ''

                      let content
                      if (col.type === 'rank') {
                        content = (
                          <span className="text-[#9CA3AF]">{i + 1}</span>
                        )
                      } else if (col.type === 'score') {
                        content = <ScoreCell value={getValue(entry, col)} />
                      } else if (col.type === 'date') {
                        const v = getValue(entry, col)
                        content = v ? (
                          <span className="text-[#6B7280]">{v}</span>
                        ) : (
                          <span className="text-[#9CA3AF]">—</span>
                        )
                      } else {
                        const v = getValue(entry, col)
                        content =
                          v !== null && v !== undefined && v !== '' ? (
                            <span
                              className={
                                col.key === 'agent_id'
                                  ? 'font-medium text-[#374151]'
                                  : 'text-[#374151]'
                              }
                            >
                              {v}
                            </span>
                          ) : (
                            <span className="text-[#9CA3AF]">—</span>
                          )
                      }

                      return (
                        <td
                          key={col.key}
                          style={stickyStyle(col.key)}
                          className={`${baseCls} ${stickyCls} ${borderCls}`}
                        >
                          {content}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
