import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getHealth } from '../api'

const navLinks = [
  { to: '/', label: 'Evaluate', end: true },
  { to: '/leaderboard', label: 'Leaderboard', end: false },
  { to: '/explorer', label: 'Explorer', end: false },
  { to: '/regression', label: 'Regression', end: false },
]

export default function NavBar() {
  const [healthy, setHealthy] = useState(null)

  useEffect(() => {
    let active = true
    getHealth()
      .then((data) => {
        if (active) setHealthy(data?.status === 'healthy')
      })
      .catch(() => {
        if (active) setHealthy(false)
      })
    return () => {
      active = false
    }
  }, [])

  const dotColor =
    healthy === null
      ? 'bg-[#9CA3AF]'
      : healthy
        ? 'bg-green-500'
        : 'bg-red-500'
  const healthText =
    healthy === null ? 'Checking…' : healthy ? 'Healthy' : 'Unavailable'

  return (
    <nav className="flex items-center justify-between bg-[#1A1D23] px-6 py-3">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded bg-[#aa3bff] font-semibold text-white">
          F
        </div>
        <span className="font-semibold text-white">Forge</span>
        <span className="text-sm text-[#9CA3AF]">v0.1.0</span>
      </div>

      <div className="flex items-center gap-6">
        {navLinks.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) =>
              isActive ? 'text-white' : 'text-[#9CA3AF]'
            }
          >
            {link.label}
          </NavLink>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${dotColor}`} />
        <span className="text-sm text-[#9CA3AF]">{healthText}</span>
      </div>
    </nav>
  )
}
