import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { LanguageSelector } from '../components/settings/LanguageSelector';
import { Link2, User, Globe, LogIn } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export function JoinRoom() {
  const [code, setCode] = useState('');
  const [name, setName] = useState('');
  const [language, setLanguage] = useState('en');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  useEffect(() => {
    const c = searchParams.get('code');
    if (c) setCode(c.toUpperCase());
  }, [searchParams]);

  const handleCodeChange = (e) => setCode(e.target.value.toUpperCase().slice(0, 6));

  const handleJoin = async (e) => {
    e.preventDefault();
    setError('');
    if (!code || code.length !== 6) { setError('Please enter a valid 6-character room code'); return; }
    if (!name.trim()) { setError('Please enter your name'); return; }
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/api/v1/rooms/${code}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, language, role: 'student' }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Join failed: ${response.status}`);
      }
      const data = await response.json();
      navigate(`/room/${data.room_code}/student/${data.participant_id}`, { state: { language } });
    } catch (err) {
      console.error('Join error:', err);
      setError(err.message || 'Failed to join room. Please check the code and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh', background: '#F8FAFC', display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: '40px 16px'
    }}>
      <div style={{ width: '100%', maxWidth: 480 }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 16, background: '#DBEAFE', margin: '0 auto 16px',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Link2 size={26} color="#2563EB" />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em', marginBottom: 6 }}>
            Join Live Class
          </h1>
          <p style={{ fontSize: 15, color: '#64748B' }}>Enter the room code provided by your teacher</p>
        </div>

        {/* Card */}
        <div style={{
          background: '#fff', border: '1px solid #E2E8F0', borderRadius: 20, padding: 32,
          boxShadow: '0 4px 20px rgba(0,0,0,0.06)'
        }}>

          {error && (
            <div style={{
              marginBottom: 20, padding: 12, borderRadius: 10, background: '#FEF2F2',
              border: '1px solid #FECACA', color: '#DC2626', fontSize: 14
            }}>
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleJoin} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

            {/* Room code */}
            <div>
              <label style={{
                display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
                marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em'
              }}>
                Room Code
              </label>
              <input
                type="text" value={code} onChange={handleCodeChange}
                placeholder="e.g., ABC123" maxLength="6"
                disabled={isLoading}
                style={{
                  width: '100%', padding: '14px', textAlign: 'center',
                  fontSize: 28, fontFamily: 'monospace', fontWeight: 800,
                  letterSpacing: '0.25em', color: '#0F172A',
                  background: '#fff', border: '2px solid #E2E8F0', borderRadius: 12,
                  transition: 'border-color 0.2s', outline: 'none',
                }}
                onFocus={e => e.target.style.borderColor = '#2563EB'}
                onBlur={e => e.target.style.borderColor = '#E2E8F0'}
                autoFocus
              />
              <p style={{ fontSize: 12, color: '#94A3B8', marginTop: 6, textAlign: 'center' }}>
                6-character code from your teacher
              </p>
            </div>

            {/* Name */}
            <div>
              <label style={{
                display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
                marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em'
              }}>
                Your Name
              </label>
              <div style={{ position: 'relative' }}>
                <User size={15} color="#94A3B8"
                  style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
                <input
                  type="text" value={name} onChange={e => setName(e.target.value)}
                  placeholder="Enter your name" disabled={isLoading}
                  className="input-field" style={{ paddingLeft: 36 }}
                />
              </div>
            </div>

            {/* Language */}
            <div>
              <label style={{
                display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
                marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em'
              }}>
                Caption Language
              </label>
              <LanguageSelector value={language} onChange={setLanguage} />
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading || code.length !== 6 || !name.trim()}
              style={{
                width: '100%', padding: '13px', borderRadius: 12, border: 'none',
                background: isLoading ? '#93C5FD' : '#2563EB', color: '#fff',
                fontSize: 15, fontWeight: 700, cursor: isLoading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                boxShadow: '0 4px 14px rgba(37,99,235,0.30)', transition: 'background 0.2s',
              }}
              onMouseEnter={e => { if (!isLoading) e.currentTarget.style.background = '#1D4ED8'; }}
              onMouseLeave={e => { if (!isLoading) e.currentTarget.style.background = '#2563EB'; }}
            >
              {isLoading
                ? <span style={{
                  width: 16, height: 16, border: '2px solid #fff', borderTopColor: 'transparent',
                  borderRadius: '50%', display: 'inline-block', animation: 'spin 0.8s linear infinite'
                }} />
                : <LogIn size={16} />}
              {isLoading ? 'Joining…' : 'Join Class'}
            </button>
          </form>

          {/* Tip */}
          <div style={{
            marginTop: 20, padding: '12px 16px', borderRadius: 10, background: '#EFF6FF',
            borderLeft: '4px solid #2563EB'
          }}>
            <p style={{ fontSize: 13, color: '#1E293B', fontWeight: 600, marginBottom: 4 }}>ℹ️ How to join</p>
            <p style={{ fontSize: 13, color: '#475569' }}>
              Room codes are 6 characters (letters + numbers). Ask your teacher for the code.
            </p>
          </div>
        </div>
      </div>
      <style>{`@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}
