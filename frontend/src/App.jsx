import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { UserProvider } from './context/UserContext';
import { CaptionProvider } from './context/CaptionContext';
import { Navbar } from './components/ui/Navbar';
import { Footer } from './components/ui/Footer';

// Pages
import { Landing } from './pages/Landing';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { Dashboard } from './pages/Dashboard';
import { CreateRoom } from './pages/CreateRoom';
import { JoinRoom } from './pages/JoinRoom';
import { TeacherRoom } from './pages/TeacherRoom';
import { StudentRoom } from './pages/StudentRoom';
import { Upload } from './pages/Upload';
import { Player } from './pages/Player';
import { History } from './pages/History';
import { MyVideos } from './pages/MyVideos';
import { Settings } from './pages/Settings';

function App() {
  return (
    <ThemeProvider>
      <UserProvider>
        <CaptionProvider>
          <Router>
            <div className="min-h-screen bg-bg-dark text-white flex flex-col">
              <Navbar />

              <main className="flex-grow">
                <Routes>
                  {/* Public routes */}
                  <Route path="/" element={<Landing />} />
                  <Route path="/login" element={<Login />} />
                  <Route path="/register" element={<Register />} />

                  {/* Protected/Main routes */}
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/room/create" element={<CreateRoom />} />
                  <Route path="/room/join" element={<JoinRoom />} />
                  <Route path="/room/:code/teacher" element={<TeacherRoom />} />
                  <Route path="/room/:code/student/:participantId" element={<StudentRoom />} />

                  {/* Media routes */}
                  <Route path="/upload" element={<Upload />} />
                  <Route path="/player" element={<Player />} />

                  {/* Info routes */}
                  <Route path="/history" element={<History />} />
                  <Route path="/my-videos" element={<MyVideos />} />
                  <Route path="/settings" element={<Settings />} />

                  {/* Fallback */}
                  <Route path="*" element={<Landing />} />
                </Routes>
              </main>

              <Footer />
            </div>
          </Router>
        </CaptionProvider>
      </UserProvider>
    </ThemeProvider>
  );
}

export default App;
