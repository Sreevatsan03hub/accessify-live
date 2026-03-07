import { Link, useNavigate } from 'react-router-dom';
import { useUser } from '../../context/UserContext';
import { Button } from './Button';
import { LayoutDashboard, History, Settings, Video, LogOut, GraduationCap } from 'lucide-react';

/**
 * Smart display name:
 *  - "Sreevatsan"           → "Sreevatsan"
 *  - "nithish kumar"        → "Nithish Kumar"
 *  - "sreesreevatsan55"     → "Sreesreevatsan" (email fallback, still clean)
 */
const displayName = (raw = '') => {
  const s = raw.trim().replace(/\d+$/, '').replace(/^\d+/, '');
  if (!s) return raw.slice(0, 12) || 'User';
  // If input contains a space it's a real full name — title-case every word
  if (s.includes(' ')) {
    return s.split(' ')
      .filter(Boolean)
      .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
      .join(' ');
  }
  // Single word: just capitalise first letter; keep as-is (user typed their name via Login)
  return s.charAt(0).toUpperCase() + s.slice(1);
};


export function Navbar() {
  const { user, logout } = useUser();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const navLinks = [
    { to: '/dashboard', label: 'Dashboard', Icon: LayoutDashboard },
    { to: '/history', label: 'History', Icon: History },
    { to: '/settings', label: 'Settings', Icon: Settings },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-bg-dark shadow-nav">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex justify-between items-center h-16">

          {/* ── Logo ─────────────────────────────────────── */}
          <Link
            to="/"
            className="flex items-center gap-2.5 hover:opacity-90 transition-opacity"
          >
            <div style={{
              width: 34, height: 34, borderRadius: 10,
              background: 'linear-gradient(135deg,#2563EB,#7C3AED)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(37,99,235,0.40)',
            }}>
              <GraduationCap size={18} className="text-white" />
            </div>
            <span style={{
              fontSize: 20, fontWeight: 800, letterSpacing: '-0.02em',
              background: 'linear-gradient(90deg,#60A5FA,#A78BFA)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>Accessify</span>
          </Link>

          {/* ── Nav links ────────────────────────────────── */}
          {user && (
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map(({ to, label, Icon }) => (
                <Link
                  key={to}
                  to={to}
                  className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium
                             text-blue-200 hover:text-white hover:bg-white/10
                             rounded-lg transition-all"
                >
                  <Icon size={15} />
                  {label}
                </Link>
              ))}

              {/* Upload — highlighted */}
              <Link
                to="/upload"
                className="flex items-center gap-1.5 ml-2 px-4 py-2 text-sm font-semibold
                           text-white rounded-lg transition-all shadow-sm"
                style={{
                  background: 'linear-gradient(135deg,#2563EB,#7C3AED)',
                  boxShadow: '0 4px 14px rgba(37,99,235,0.35)'
                }}
              >
                <Video size={15} />
                Upload Video
              </Link>
            </div>
          )}

          {/* ── User / auth ──────────────────────────────── */}
          <div className="flex items-center gap-3">
            {user ? (
              <div className="flex items-center gap-3">
                {/* Avatar */}
                <div className="w-9 h-9 rounded-full bg-primary flex items-center justify-center
                                text-white text-sm font-bold select-none">
                  {displayName(user.name).charAt(0).toUpperCase()}
                </div>
                <span className="hidden md:block text-sm font-semibold text-white">
                  {displayName(user.name)}
                </span>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium
                             text-blue-200 hover:text-white border border-white/20
                             hover:bg-white/10 rounded-lg transition-all"
                >
                  <LogOut size={14} />
                  Logout
                </button>
              </div>
            ) : (
              <div className="flex gap-2">
                <Link to="/login">
                  <button className="px-4 py-1.5 text-sm font-semibold text-white
                                     border border-white/30 rounded-lg hover:bg-white/10 transition-all">
                    Sign In
                  </button>
                </Link>
                <Link to="/register">
                  <button className="px-4 py-1.5 text-sm font-semibold bg-primary text-white
                                     rounded-lg hover:bg-blue-600 transition-all">
                    Register
                  </button>
                </Link>
              </div>
            )}
          </div>

        </div>
      </div>
    </nav>
  );
}
