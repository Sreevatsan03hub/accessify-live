import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RoomCodeDisplay } from '../components/classroom/RoomCodeDisplay';
import { Video, User, Sparkles } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export function CreateRoom() {
  const [title, setTitle] = useState('');
  const [teacherName, setTeacherName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [roomCode, setRoomCode] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleCreateRoom = async (e) => {
    e.preventDefault();
    if (!title || !teacherName) { alert('Please fill in all fields'); return; }
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE}/api/v1/rooms/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, teacher_name: teacherName }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setRoomCode(data.room_code);
    } catch (err) {
      console.error('Create room error:', err);
      setError(err.message || 'Failed to create room. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  const wrapStyle = {
    minHeight: '100vh', background: '#F8FAFC',
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 16px',
  };
  const cardStyle = {
    width: '100%', maxWidth: 500,
    background: '#fff', border: '1px solid #E2E8F0', borderRadius: 20, padding: 32,
    boxShadow: '0 4px 20px rgba(0,0,0,0.06)',
  };
  const LabelStyle = {
    display: 'block', fontSize: 13, fontWeight: 700, color: '#0F172A',
    marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em'
  };

  /* ── Room created — show code ── */
  if (roomCode) {
    return (
      <div style={wrapStyle}>
        <div style={{ ...cardStyle, maxWidth: 560 }}>
          <RoomCodeDisplay code={roomCode} teacherName={teacherName} title={title} />
          <div style={{ marginTop: 24, display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              onClick={() => navigate(`/room/${roomCode}/teacher`)}
              style={{
                padding: '12px 24px', borderRadius: 12, border: 'none',
                background: '#2563EB', color: '#fff', fontWeight: 700, fontSize: 15,
                cursor: 'pointer', boxShadow: '0 4px 14px rgba(37,99,235,0.30)',
                display: 'flex', alignItems: 'center', gap: 8
              }}
            >
              <Video size={17} /> Start Broadcasting
            </button>
            <button
              onClick={() => setRoomCode(null)}
              style={{
                padding: '12px 20px', borderRadius: 12, border: '1px solid #E2E8F0',
                background: '#fff', color: '#374151', fontWeight: 600, fontSize: 14, cursor: 'pointer'
              }}
            >
              ← Create Another
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={wrapStyle}>
      <div style={{ width: '100%', maxWidth: 500 }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 16, background: '#DBEAFE',
            margin: '0 auto 14px', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Video size={26} color="#2563EB" />
          </div>
          <h1 style={{ fontSize: 32, fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em', marginBottom: 6 }}>
            Start a Live Class
          </h1>
          <p style={{ fontSize: 15, color: '#64748B' }}>Set up a new classroom session with real-time captions</p>
        </div>

        <div style={cardStyle}>
          {error && (
            <div style={{
              marginBottom: 20, padding: 12, borderRadius: 10, background: '#FEF2F2',
              border: '1px solid #FECACA', color: '#DC2626', fontSize: 14
            }}>
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleCreateRoom} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div>
              <label style={LabelStyle}>Class Title</label>
              <input
                type="text" value={title} onChange={e => setTitle(e.target.value)}
                placeholder="e.g., Introduction to Machine Learning"
                className="input-field" disabled={isLoading} autoFocus
              />
            </div>

            <div>
              <label style={LabelStyle}>Teacher Name</label>
              <div style={{ position: 'relative' }}>
                <User size={15} color="#94A3B8"
                  style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }} />
                <input
                  type="text" value={teacherName} onChange={e => setTeacherName(e.target.value)}
                  placeholder="Your name" className="input-field" style={{ paddingLeft: 36 }}
                  disabled={isLoading}
                />
              </div>
            </div>

            <button
              type="submit" disabled={isLoading}
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
                : <Sparkles size={16} />}
              {isLoading ? 'Creating room…' : 'Create Room'}
            </button>
          </form>

          {/* Tips */}
          <div style={{
            marginTop: 20, padding: '12px 16px', borderRadius: 10,
            background: '#EFF6FF', borderLeft: '4px solid #2563EB'
          }}>
            <p style={{ fontSize: 13, fontWeight: 700, color: '#1E293B', marginBottom: 6 }}>💡 Tips</p>
            <ul style={{ fontSize: 13, color: '#475569', lineHeight: 1.8, paddingLeft: 4, listStyle: 'none' }}>
              <li>• Share the room code with students to let them join</li>
              <li>• Each session gets a unique 6-character code</li>
              <li>• Students can select their own caption language</li>
            </ul>
          </div>
        </div>
      </div>
      <style>{`@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}
