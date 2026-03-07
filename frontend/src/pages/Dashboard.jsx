import { Link, useNavigate } from 'react-router-dom';
import { useUser } from '../context/UserContext';
import {
  Video, Link2, FolderOpen, PlayCircle, FileText, Settings,
  ArrowRight, TrendingUp, Clock, BookOpen, Zap
} from 'lucide-react';

const displayName = (raw = '') => {
  const s = raw.trim().replace(/\d+$/, '').replace(/^\d+/, '');
  if (!s) return raw.slice(0, 10) || 'User';
  if (s.includes(' ')) return s.split(' ').filter(Boolean)
    .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

/* ── Vibrant gradient icon configs ─────────────────────────── */
const ACTION_CARDS = [
  {
    id: 'create', to: '/room/create', Icon: Video,
    gradient: 'linear-gradient(135deg,#2563EB,#1D4ED8)',
    glow: 'rgba(37,99,235,0.30)',
    title: 'Start Live Class',
    desc: 'Launch a new classroom session with real-time captions',
    badge: '● Live', badgeBg: '#FEE2E2', badgeColor: '#DC2626',
  },
  {
    id: 'join', to: '/room/join', Icon: Link2,
    gradient: 'linear-gradient(135deg,#059669,#047857)',
    glow: 'rgba(5,150,105,0.28)',
    title: 'Join a Class',
    desc: 'Enter a room code to join as a student',
  },
  {
    id: 'upload', to: '/upload', Icon: FolderOpen,
    gradient: 'linear-gradient(135deg,#D97706,#B45309)',
    glow: 'rgba(217,119,6,0.28)',
    title: 'Upload Video',
    desc: 'Add captions to pre-recorded video lessons',
  },
  {
    id: 'replay', to: '/player', Icon: PlayCircle,
    gradient: 'linear-gradient(135deg,#7C3AED,#6D28D9)',
    glow: 'rgba(124,58,237,0.28)',
    title: 'Replay Sessions',
    desc: 'Watch past classes with AI-generated captions',
  },
  {
    id: 'history', to: '/history', Icon: FileText,
    gradient: 'linear-gradient(135deg,#0EA5E9,#0284C7)',
    glow: 'rgba(14,165,233,0.28)',
    title: 'Transcripts',
    desc: 'Browse and download session transcripts',
  },
  {
    id: 'settings', to: '/settings', Icon: Settings,
    gradient: 'linear-gradient(135deg,#64748B,#475569)',
    glow: 'rgba(100,116,139,0.22)',
    title: 'Preferences',
    desc: 'Customise caption size, language & accessibility',
  },
];

const MOCK_SESSIONS = [
  { id: 1, title: 'Introduction to Machine Learning', type: 'video', date: 'Mar 3, 2026', captions: 125 },
  { id: 2, title: 'Data Structures Class – Week 5', type: 'live', date: 'Mar 2, 2026', captions: 203 },
  { id: 3, title: 'Advanced Python Programming', type: 'video', date: 'Mar 1, 2026', captions: 456 },
];

/* ── Reusable styled section heading ───────────────────────── */
const SectionTitle = ({ accent, title, sub }) => (
  <div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
      <span style={{
        display: 'inline-block', width: 4, height: 22, borderRadius: 4,
        background: 'linear-gradient(180deg,#2563EB,#7C3AED)',
      }} />
      <h2 style={{ fontSize: 20, fontWeight: 800, color: '#0F172A', letterSpacing: '-0.01em' }}>
        {title}
      </h2>
      {accent && (
        <span style={{
          fontSize: 11, fontWeight: 700, padding: '2px 10px', borderRadius: 999,
          background: 'linear-gradient(90deg,#EFF6FF,#F5F3FF)',
          color: '#2563EB', border: '1px solid #BFDBFE'
        }}>
          {accent}
        </span>
      )}
    </div>
    {sub && <p style={{ fontSize: 13, color: '#64748B', marginLeft: 14 }}>{sub}</p>}
  </div>
);

export function Dashboard() {
  const { user } = useUser();
  const navigate = useNavigate();
  if (!user) { navigate('/login'); return null; }
  const dName = displayName(user.name);

  return (
    <div style={{ minHeight: '100vh', background: '#F8FAFC' }}>

      {/* ── Hero ─────────────────────────────────────────── */}
      <div style={{
        background: 'linear-gradient(135deg,#0F172A 0%,#1E3A8A 60%,#1D4ED8 100%)',
        position: 'relative', overflow: 'hidden'
      }}>
        {/* Subtle dot grid */}
        <div style={{
          position: 'absolute', inset: 0, opacity: 0.07,
          backgroundImage: 'radial-gradient(circle at 1px 1px,white 1px,transparent 0)',
          backgroundSize: '36px 36px'
        }} />
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '36px 24px', position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
            <div>
              <p style={{
                fontSize: 12, fontWeight: 700, color: '#60A5FA', letterSpacing: '0.08em',
                textTransform: 'uppercase', marginBottom: 6
              }}>
                {user.role === 'teacher' ? '👩‍🏫 Teacher Portal' : '🎓 Student Portal'}
              </p>
              <h1 style={{
                fontSize: 36, fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em',
                lineHeight: 1.15, marginBottom: 6
              }}>
                Welcome back, <span style={{
                  background: 'linear-gradient(90deg,#60A5FA,#A78BFA)',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                }}>{dName}!</span> 👋
              </h1>
              <p style={{ fontSize: 14, color: '#93C5FD' }}>
                {user.role === 'teacher' ? 'Ready to start a new session today?' : "Let's continue learning where you left off."}
              </p>
            </div>

            {/* Quick stats */}
            <div style={{ display: 'flex', gap: 10 }}>
              {[
                { label: 'Sessions', value: '12', Icon: TrendingUp, accent: '#60A5FA' },
                { label: 'Hours', value: '8.4', Icon: Clock, accent: '#34D399' },
                { label: 'Courses', value: '3', Icon: BookOpen, accent: '#FBBF24' },
              ].map(({ label, value, Icon, accent }) => (
                <div key={label} style={{
                  background: 'rgba(255,255,255,0.10)', border: '1px solid rgba(255,255,255,0.15)',
                  backdropFilter: 'blur(8px)', borderRadius: 14, padding: '12px 18px', textAlign: 'center', minWidth: 88,
                }}>
                  <Icon size={16} style={{ color: accent, margin: '0 auto 4px', display: 'block' }} />
                  <p style={{ fontSize: 22, fontWeight: 800, color: '#fff', lineHeight: 1 }}>{value}</p>
                  <p style={{ fontSize: 11, color: '#93C5FD', marginTop: 2 }}>{label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '36px 24px' }}>

        {/* ── Quick Actions ─────────────────────────────── */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <SectionTitle title="Quick Actions" accent="6 tools" sub="Everything you need in one place" />
            <span style={{
              display: 'flex', alignItems: 'center', gap: 5, fontSize: 12,
              color: '#64748B', fontWeight: 600
            }}>
              <Zap size={12} color="#F59E0B" /> AI-Powered
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 16 }}>
            {ACTION_CARDS.map(({ id, to, Icon, gradient, glow, title, desc, badge, badgeBg, badgeColor }) => (
              <Link key={id} to={to} style={{ textDecoration: 'none' }}>
                <div
                  style={{
                    background: '#fff', border: '1px solid #E2E8F0', borderRadius: 18,
                    padding: 22, height: '100%', display: 'flex', flexDirection: 'column',
                    transition: 'all 0.22s ease', cursor: 'pointer',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    e.currentTarget.style.boxShadow = `0 12px 32px ${glow}`;
                    e.currentTarget.style.borderColor = 'transparent';
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = '#E2E8F0';
                  }}
                >
                  {/* Icon row */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 14 }}>
                    <div style={{
                      width: 50, height: 50, borderRadius: 14, background: gradient,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      boxShadow: `0 6px 16px ${glow}`, flexShrink: 0,
                    }}>
                      <Icon size={22} color="#fff" />
                    </div>
                    {badge && (
                      <span style={{
                        fontSize: 11, fontWeight: 700, padding: '4px 10px', borderRadius: 999,
                        background: badgeBg, color: badgeColor, animation: 'pulse-badge 2s infinite',
                      }}>
                        {badge}
                      </span>
                    )}
                  </div>

                  {/* Text */}
                  <h3 style={{ fontSize: 16, fontWeight: 700, color: '#0F172A', marginBottom: 6 }}>{title}</h3>
                  <p style={{ fontSize: 13, color: '#64748B', lineHeight: 1.55, flex: 1 }}>{desc}</p>

                  {/* Open link */}
                  <div style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5, marginTop: 14,
                    fontSize: 13, fontWeight: 700,
                    background: 'linear-gradient(90deg,#2563EB,#7C3AED)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                  }}>
                    Open <ArrowRight size={13} style={{ color: '#2563EB' }} />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* ── Recent Sessions ───────────────────────────── */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <SectionTitle title="Recent Sessions" sub="Your latest classes and uploads" />
            <Link to="/history" style={{
              display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 13,
              fontWeight: 700, color: '#2563EB', textDecoration: 'none',
              padding: '6px 14px', borderRadius: 8, background: '#EFF6FF',
              border: '1px solid #BFDBFE', transition: 'all 0.15s',
            }}>
              View All <ArrowRight size={13} />
            </Link>
          </div>

          <div style={{
            background: '#fff', border: '1px solid #E2E8F0', borderRadius: 16,
            overflow: 'hidden', boxShadow: '0 2px 12px rgba(0,0,0,0.04)'
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'linear-gradient(90deg,#F8FAFC,#F1F5F9)', borderBottom: '1px solid #E2E8F0' }}>
                  {['Title', 'Type', 'Date', 'Captions', 'Actions'].map(h => (
                    <th key={h} style={{
                      textAlign: 'left', padding: '12px 20px', fontSize: 11,
                      fontWeight: 800, color: '#64748B', textTransform: 'uppercase',
                      letterSpacing: '0.07em',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MOCK_SESSIONS.map((s, i) => (
                  <tr key={s.id}
                    style={{
                      borderBottom: i < MOCK_SESSIONS.length - 1 ? '1px solid #F1F5F9' : 'none',
                      background: i % 2 === 0 ? '#fff' : '#FAFAFA',
                      transition: 'background 0.15s'
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = '#EFF6FF'}
                    onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? '#fff' : '#FAFAFA'}>
                    <td style={{ padding: '14px 20px', fontSize: 14, fontWeight: 600, color: '#0F172A' }}>{s.title}</td>
                    <td style={{ padding: '14px 20px' }}>
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', gap: 5,
                        padding: '4px 10px', borderRadius: 999, fontSize: 12, fontWeight: 700,
                        background: s.type === 'live' ? '#FEE2E2' : '#DBEAFE',
                        color: s.type === 'live' ? '#DC2626' : '#1D4ED8',
                      }}>
                        {s.type === 'live' ? '🔴' : '🎬'} {s.type.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '14px 20px', fontSize: 13, color: '#64748B' }}>{s.date}</td>
                    <td style={{ padding: '14px 20px', fontSize: 14, fontWeight: 700, color: '#0F172A' }}>
                      <span style={{ padding: '2px 8px', borderRadius: 6, background: '#F1F5F9', color: '#0F172A' }}>
                        {s.captions}
                      </span>
                    </td>
                    <td style={{ padding: '14px 20px' }}>
                      <div style={{ display: 'flex', gap: 8 }}>
                        <Link to="/player">
                          <button style={{
                            padding: '5px 12px', fontSize: 12, fontWeight: 700, borderRadius: 8,
                            background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE',
                            cursor: 'pointer', transition: 'all 0.15s',
                          }}
                            onMouseEnter={e => { e.currentTarget.style.background = '#2563EB'; e.currentTarget.style.color = '#fff'; }}
                            onMouseLeave={e => { e.currentTarget.style.background = '#EFF6FF'; e.currentTarget.style.color = '#2563EB'; }}>
                            👁 View
                          </button>
                        </Link>
                        <Link to="/history">
                          <button style={{
                            padding: '5px 12px', fontSize: 12, fontWeight: 700, borderRadius: 8,
                            background: '#F8FAFC', color: '#475569', border: '1px solid #E2E8F0',
                            cursor: 'pointer', transition: 'all 0.15s',
                          }}
                            onMouseEnter={e => { e.currentTarget.style.background = '#F1F5F9'; e.currentTarget.style.borderColor = '#2563EB'; }}>
                            ⬇ Download
                          </button>
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      <style>{`
        @keyframes pulse-badge {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.65; }
        }
      `}</style>
    </div>
  );
}
