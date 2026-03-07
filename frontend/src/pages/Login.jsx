import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { GraduationCap, LogIn } from 'lucide-react';
import { isConfigured } from '../firebase';

// ⚠️ Must be outside the component — defining inside causes focus to jump on every keystroke
const Field = ({ label, children }) => (
  <div>
    <label style={{
      display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
      marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em'
    }}>
      {label}
    </label>
    {children}
  </div>
);

export function Login() {
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('student');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login } = useUser();
  const navigate = useNavigate();

  const friendlyError = (code) => {
    const map = {
      'auth/user-not-found': 'No account found with that email. Please register first.',
      'auth/wrong-password': 'Incorrect password. Please try again.',
      'auth/invalid-email': 'Please enter a valid email address.',
      'auth/too-many-requests': 'Too many attempts. Please wait a moment and try again.',
      'auth/network-request-failed': 'Network error. Check your internet connection.',
    };
    return map[code] || 'Login failed. Please check your credentials.';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!email || !password) { setError('Please enter your email and password'); return; }
    try {
      setIsLoading(true);
      const name = displayName.trim() || email.split('@')[0];
      // Pass password so Firebase Auth can use it when configured
      await login(name, role, 'en', password);
      if (rememberMe) localStorage.setItem('rememberEmail', email);
      navigate('/dashboard');
    } catch (err) {
      setError(friendlyError(err.code));
    } finally {
      setIsLoading(false);
    }
  };






  return (
    <div className="min-h-screen flex" style={{ background: '#F1F5F9' }}>

      {/* ── Left brand panel ─── */}
      <div className="hidden lg:flex flex-col justify-between w-[44%] px-14 py-12"
        style={{ background: 'linear-gradient(135deg,#0F172A,#1E3A8A)' }}>
        <div className="flex items-center gap-2.5 text-white">
          <div className="w-9 h-9 bg-blue-500 rounded-xl flex items-center justify-center">
            <GraduationCap size={19} />
          </div>
          <span className="text-xl font-bold tracking-tight">Accessify</span>
        </div>

        <div>
          <h2 className="text-4xl font-extrabold text-white leading-snug mb-5">
            AI-Powered<br />Accessible<br />
            <span className="text-blue-400">Learning.</span>
          </h2>
          <p className="text-slate-300 text-[15px] leading-relaxed max-w-xs">
            Real-time captions, multilingual translation, and AI-enhanced
            understanding — making education accessible for everyone.
          </p>
          <div className="mt-9 grid grid-cols-2 gap-3">
            {[['Languages', '3+'], ['Real-time', '< 1s'], ['Accuracy', '97%'], ['Uptime', '24/7']].map(([l, v]) => (
              <div key={l} className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.08)' }}>
                <p className="text-2xl font-bold text-white">{v}</p>
                <p className="text-slate-400 text-sm">{l}</p>
              </div>
            ))}
          </div>
        </div>
        <p className="text-slate-500 text-xs">© {new Date().getFullYear()} Accessify. All rights reserved.</p>
      </div>

      {/* ── Right form panel ─── */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">

          <div className="flex items-center gap-2 mb-8 lg:hidden">
            <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center">
              <GraduationCap size={17} className="text-white" />
            </div>
            <span className="text-xl font-bold text-slate-800">Accessify</span>
          </div>

          <h1 className="text-[28px] font-bold text-slate-900 mb-1" style={{ letterSpacing: '-0.02em' }}>
            Sign in to Accessify
          </h1>
          <p className="text-slate-500 text-sm mb-7">
            Enter your details to access your dashboard
          </p>

          {error && (
            <div className="mb-5 p-3.5 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm">
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <Field label="Your Name">
              <input
                type="text"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                placeholder="Sreevatsan"
                className="input-field"
                disabled={isLoading}
              />
            </Field>

            <Field label="Email Address">
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="input-field"
                disabled={isLoading}
              />
            </Field>

            <Field label="Password">
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
                <a href="#" style={{ fontSize: 12, color: '#2563EB', fontWeight: 600 }}>Forgot password?</a>
              </div>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                className="input-field"
                disabled={isLoading}
              />
            </Field>

            <div className="flex items-center gap-2 pt-1">
              <input
                type="checkbox" id="remember" checked={rememberMe}
                onChange={e => setRememberMe(e.target.checked)}
                className="w-4 h-4 rounded border-slate-300 text-blue-600"
                disabled={isLoading}
              />
              <label htmlFor="remember" className="text-sm text-slate-500 cursor-pointer">
                Remember me for 30 days
              </label>
            </div>

            <button
              type="submit" disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-3 px-5 mt-2
                         text-white text-sm font-semibold rounded-xl transition-all
                         disabled:opacity-60 disabled:cursor-not-allowed"
              style={{
                background: isLoading ? '#3b82f6' : '#2563EB',
                boxShadow: '0 10px 25px rgba(37,99,235,0.30)'
              }}
              onMouseEnter={e => { if (!isLoading) e.currentTarget.style.background = '#1D4ED8'; }}
              onMouseLeave={e => { if (!isLoading) e.currentTarget.style.background = '#2563EB'; }}
            >
              {isLoading
                ? <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                : <LogIn size={16} />}
              {isLoading ? 'Signing in…' : 'Sign In'}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-slate-500">
            Don't have an account?{' '}
            <Link to="/register" className="text-blue-600 font-semibold hover:underline">
              Create one →
            </Link>
          </p>

          <div className="mt-5 p-3.5 bg-blue-50 border border-blue-100 rounded-xl text-blue-600 text-xs">
            ℹ️ Demo mode — any email + password will work to explore the platform.
          </div>
        </div>
      </div>
    </div>
  );
}
