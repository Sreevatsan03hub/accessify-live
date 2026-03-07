import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import { LanguageSelector } from '../components/settings/LanguageSelector';
import { GraduationCap, UserCheck } from 'lucide-react';

// ⚠️ Must be defined OUTSIDE the component to avoid re-creation on every render
// (re-creation causes focus to jump back to the first field on every keystroke)
const LabelInput = ({ label, children }) => (
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

export function Register() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState('student');
  const [language, setLanguage] = useState('en');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { register, login } = useUser();
  const navigate = useNavigate();

  const friendlyError = (code, message) => {
    const map = {
      'auth/email-already-in-use': 'An account with this email already exists. Try logging in.',
      'auth/weak-password': 'Password is too weak. Use at least 6 characters.',
      'auth/invalid-email': 'Please enter a valid email address.',
      'auth/network-request-failed': 'Network error. Check your internet connection.',
      'auth/operation-not-allowed': 'Email/Password sign-in is not enabled in Firebase Console.',
      'auth/api-key-not-valid.-please-pass-a-valid-api-key.': 'API key is invalid. Check .env.local',
    };
    return map[code] || `Error [${code}]: ${message}`;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!name || !email || !password || !confirmPassword) { setError('Please fill in all fields'); return; }
    if (password !== confirmPassword) { setError('Passwords do not match'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
    try {
      setIsLoading(true);
      // register() uses Firebase Auth when configured, localStorage fallback otherwise
      if (register) {
        await register(name.trim(), email.trim(), password, role, language);
      } else {
        await login(name.trim(), role, language);
      }
      navigate('/dashboard');
    } catch (err) {
      setError(friendlyError(err.code, err.message));
    } finally {
      setIsLoading(false);
    }
  };




  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12" style={{ background: '#F1F5F9' }}>
      <div className="w-full max-w-md">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-4"
            style={{ background: '#2563EB', boxShadow: '0 8px 20px rgba(37,99,235,0.35)' }}>
            <GraduationCap size={24} className="text-white" />
          </div>
          <h1 className="text-[28px] font-bold mb-1" style={{ color: '#0F172A', letterSpacing: '-0.02em' }}>
            Create your account
          </h1>
          <p className="text-sm" style={{ color: '#64748B' }}>
            Join Accessify to get started with accessible learning
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl p-8" style={{
          background: '#fff', border: '1px solid #E2E8F0',
          boxShadow: '0 8px 24px rgba(0,0,0,0.06)'
        }}>
          {error && (
            <div className="mb-5 p-3.5 rounded-xl text-sm" style={{ background: '#FEF2F2', border: '1px solid #FECACA', color: '#DC2626' }}>
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <LabelInput label="Full Name">
              <input
                type="text" value={name} onChange={e => setName(e.target.value)}
                placeholder="Sreevatsan" className="input-field"
                disabled={isLoading}
              />
            </LabelInput>

            <LabelInput label="Email Address">
              <input
                type="email" value={email} onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com" className="input-field"
                disabled={isLoading}
              />
            </LabelInput>

            <LabelInput label="Password">
              <input
                type="password" value={password} onChange={e => setPassword(e.target.value)}
                placeholder="Min. 6 characters" className="input-field"
                disabled={isLoading}
              />
            </LabelInput>

            <LabelInput label="Confirm Password">
              <input
                type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)}
                placeholder="••••••••" className="input-field"
                disabled={isLoading}
              />
            </LabelInput>

            {/* Role */}
            <div>
              <label className="block text-sm font-semibold mb-1.5" style={{ color: '#0F172A' }}>I am a</label>
              <div className="grid grid-cols-2 gap-3">
                {['student', 'teacher'].map(r => (
                  <button key={r} type="button" onClick={() => setRole(r)}
                    className="py-2.5 rounded-xl text-sm font-semibold border-2 transition-all capitalize"
                    style={{
                      borderColor: role === r ? '#2563EB' : '#E2E8F0',
                      background: role === r ? '#EFF6FF' : '#fff',
                      color: role === r ? '#2563EB' : '#64748B',
                    }}>
                    {r === 'student' ? '🎓' : '👩‍🏫'} {r.charAt(0).toUpperCase() + r.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Language */}
            <div>
              <label className="block text-sm font-semibold mb-1.5" style={{ color: '#0F172A' }}>Preferred Language</label>
              <LanguageSelector value={language} onChange={setLanguage} />
            </div>

            <button
              type="submit" disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-white text-sm font-semibold
                         transition-all mt-2 disabled:opacity-60"
              style={{ background: '#2563EB', boxShadow: '0 8px 20px rgba(37,99,235,0.30)' }}
              onMouseEnter={e => { if (!isLoading) e.currentTarget.style.background = '#1D4ED8'; }}
              onMouseLeave={e => { if (!isLoading) e.currentTarget.style.background = '#2563EB'; }}
            >
              {isLoading
                ? <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                : <UserCheck size={16} />}
              {isLoading ? 'Creating account…' : 'Create Account'}
            </button>
          </form>

          <p className="mt-5 text-center text-sm" style={{ color: '#64748B' }}>
            Already have an account?{' '}
            <Link to="/login" className="font-semibold" style={{ color: '#2563EB' }}>
              Sign in →
            </Link>
          </p>
        </div>

        <div className="mt-4 px-4 py-3 rounded-xl text-xs text-center" style={{ background: '#EFF6FF', color: '#3B82F6' }}>
          ℹ️ Demo mode — registrations are saved locally in your browser.
        </div>
      </div>
    </div>
  );
}
