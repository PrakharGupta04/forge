import { BrowserRouter, Routes, Route } from 'react-router-dom'
import NavBar from './components/NavBar'
import EvaluatePage from './pages/EvaluatePage'
import LeaderboardPage from './pages/LeaderboardPage'
import TrajectoryPage from './pages/TrajectoryPage'
import RegressionPage from './pages/RegressionPage'

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<EvaluatePage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/explorer" element={<TrajectoryPage />} />
        <Route path="/explorer/:id" element={<TrajectoryPage />} />
        <Route path="/regression" element={<RegressionPage />} />
      </Routes>
    </BrowserRouter>
  )
}
