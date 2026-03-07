import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './context/ThemeContext'
import { UserProvider } from './context/UserContext'
import { CaptionProvider } from './context/CaptionContext'
import { Navbar } from './components/ui/Navbar'

// Pages
import { Landing } from './pages/Landing'
import { Login } from './pages/Login'
import { Register } from './pages/Register'
import { Dashboard } from './pages/Dashboard'
import { CreateRoom } from './pages/CreateRoom'
import { JoinRoom } from './pages/JoinRoom'
import { TeacherRoom } from './pages/TeacherRoom'
import { StudentRoom } from './pages/StudentRoom'
import { Upload } from './pages/Upload'
import { Player } from './pages/Player'
import { History } from './pages/History'
import { Settings } from './pages/Settings'

function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <CaptionProvider>
          <Router>
            <Navbar />
            <main className="min-h-screen bg-background text-foreground">
              <Routes>
                {/* Public pages */}
                <Route path="/" element={<Landing />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />

                {/* Dashboard & room management */}
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/room/create" element={<CreateRoom />} />
                <Route path="/room/join" element={<JoinRoom />} />

                {/* Live classroom */}
                <Route path="/room/:code/teacher" element={<TeacherRoom />} />
                <Route path="/room/:code/student/:participantId" element={<StudentRoom />} />

                {/* Video & captions */}
                <Route path="/upload" element={<Upload />} />
                <Route path="/player" element={<Player />} />
                <Route path="/history" element={<History />} />
                <Route path="/settings" element={<Settings />} />

                {/* Catch all */}
                <Route path="*" element={<Navigate to="/" />} />
              </Routes>
            </main>
          </Router>
        </CaptionProvider>
      </UserProvider>
    </ThemeProvider>
  )
}

export default App
