import { Link } from 'react-router-dom'
import { useTheme } from '../../context/ThemeContext'
import { useUser } from '../../context/UserContext'
import { Button } from './Button'
import { Moon, Sun, LogOut } from 'lucide-react'

export function Navbar() {
  const { theme, toggleTheme } = useTheme()
  const { user, logout } = useUser()

  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-xl hover:opacity-80 transition-opacity">
          <div className="text-3xl">🎓</div>
          <span>Accessify</span>
        </Link>

        <div className="flex items-center gap-4">
          {user && (
            <div className="text-sm">
              Welcome, <span className="font-semibold">{user.name}</span>
            </div>
          )}

          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-primary/20 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          {user ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={logout}
              className="gap-2"
            >
              <LogOut size={16} />
              Logout
            </Button>
          ) : (
            <Link to="/login">
              <Button size="sm">Sign In</Button>
            </Link>
          )}
        </div>
      </div>
    </nav>
  )
}
