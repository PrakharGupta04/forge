import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  timeout: 120000,
})

function toError(error) {
  const message = error.response?.data?.detail || error.message
  return new Error(message)
}

export async function evaluateTrajectory(
  trajectory,
  includeExplanations = true,
  weightingStrategy = 'equal',
) {
  try {
    const res = await api.post('/evaluate', {
      trajectory,
      include_explanations: includeExplanations,
      weighting_strategy: weightingStrategy,
    })
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function getHealth() {
  try {
    const res = await api.get('/health')
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function runBenchmark(
  benchmark = 'all',
  agentId = 'api_agent',
  maxTasks = null,
  agentType = 'mock',
) {
  try {
    const res = await api.post('/benchmark/run', {
      benchmark,
      agent_id: agentId,
      max_tasks: maxTasks,
      agent_type: agentType,
    })
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function getBenchmarkStatus(jobId) {
  try {
    const res = await api.get(`/benchmark/${jobId}`)
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function getLeaderboard(benchmark = null, excludeMock = false) {
  try {
    const params = { exclude_mock: excludeMock }
    if (benchmark !== null) {
      params.benchmark = benchmark
    }
    const res = await api.get('/leaderboard', { params })
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function getTrajectory(trajectoryId) {
  try {
    const res = await api.get(`/trajectories/${trajectoryId}`)
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export async function compareAgents(agent1, agent2) {
  try {
    const res = await api.get(
      `/compare?agent_1=${agent1}&agent_2=${agent2}`,
    )
    return res.data
  } catch (error) {
    throw toError(error)
  }
}

export default api
