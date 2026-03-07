import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Modal } from '../components/ui/Modal';
import { Button } from '../components/ui/Button';
import { Radio, Clock, MessageSquare, Download, Trash2, ChevronDown } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const S = {
  page: { background: '#F1F5F9', minHeight: '100vh', padding: '48px 0' },
  container: { maxWidth: 800, margin: '0 auto', padding: '0 24px' },
  heading: { fontSize: 34, fontWeight: 800, color: '#0F172A', letterSpacing: '-0.02em', marginBottom: 4 },
  sub: { fontSize: 15, color: '#64748B', marginBottom: 32 },
};

export function History() {
  const [sessions, setSessions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        setIsLoading(true);
        const res = await fetch(`${API_BASE}/api/v1/sessions/`);
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        const data = await res.json();
        const all = data.sessions || data || [];
        setSessions(all.filter(s => (s.session_type || s.type) === 'live'));
      } catch (err) {
        setError(err.message);
        setSessions([]);
      } finally {
        setIsLoading(false);
      }
    };
    fetchSessions();
  }, []);

  const formatDate = iso => !iso ? '—'
    : new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });

  const formatDuration = secs => {
    if (!secs) return null;
    return `${Math.floor(secs / 60)}m ${Math.floor(secs % 60)}s`;
  };

  const handleDownload = async (session, format) => {
    const id = session.session_id || session.id;
    try {
      const res = await fetch(`${API_BASE}/api/v1/export/${id}/${format}`);
      if (!res.ok) throw new Error('Download failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement('a'), { href: url, download: `${session.title || id}.${format}` });
      a.click(); URL.revokeObjectURL(url);
    } catch (err) { alert(`Download failed: ${err.message}`); }
  };

  const handleDelete = s => { setSessionToDelete(s); setIsDeleteModalOpen(true); };
  const confirmDelete = async () => {
    if (!sessionToDelete) return;
    try {
      await fetch(`${API_BASE}/api/v1/sessions/${sessionToDelete.session_id || sessionToDelete.id}`, { method: 'DELETE' });
      setSessions(prev => prev.filter(s => (s.session_id || s.id) !== (sessionToDelete.session_id || sessionToDelete.id)));
    } catch (err) { console.error(err); }
    finally { setIsDeleteModalOpen(false); setSessionToDelete(null); }
  };

  if (isLoading) return (
    <div style={{ ...S.page, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 40, marginBottom: 16, animation: 'spin 1.2s linear infinite' }}>📡</div>
        <p style={{ fontSize: 15, fontWeight: 600, color: '#64748B' }}>Loading live sessions…</p>
      </div>
      <style>{`@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}`}</style>
    </div>
  );

  return (
    <div style={S.page}>
      <div style={S.container}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 8 }}>
          <div style={{
            width: 48, height: 48, borderRadius: 14, background: '#DBEAFE',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Radio size={22} color="#2563EB" />
          </div>
          <div>
            <h1 style={S.heading}>Live Session History</h1>
          </div>
        </div>
        <p style={S.sub}>
          Real-time classroom transcripts — saved automatically.{' '}
          <Link to="/my-videos" style={{ color: '#2563EB', fontWeight: 600 }}>View My Videos →</Link>
        </p>

        {/* Error */}
        {error && (
          <div style={{
            marginBottom: 20, padding: 14, borderRadius: 10, background: '#FEF2F2',
            border: '1px solid #FECACA', color: '#DC2626', fontSize: 14
          }}>
            ⚠️ {error} — Make sure the backend is running at {API_BASE}
          </div>
        )}

        {/* Stats */}
        {sessions.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 24 }}>
            {[
              { Icon: Radio, color: '#2563EB', bg: '#DBEAFE', label: 'Live Sessions', value: sessions.length },
              { Icon: MessageSquare, color: '#059669', bg: '#D1FAE5', label: 'Total Captions', value: sessions.reduce((a, s) => a + (s.caption_count || 0), 0) },
              { Icon: Clock, color: '#D97706', bg: '#FEF3C7', label: 'Languages', value: [...new Set(sessions.map(s => (s.language || 'en').toUpperCase()))].join(', ') },
            ].map(({ Icon, color, bg, label, value }) => (
              <div key={label} style={{
                background: '#fff', border: '1px solid #E2E8F0', borderRadius: 12,
                padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 12
              }}>
                <div style={{
                  width: 36, height: 36, borderRadius: 10, background: bg,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                }}>
                  <Icon size={17} color={color} />
                </div>
                <div>
                  <p style={{ fontSize: 22, fontWeight: 800, color: '#0F172A', lineHeight: 1 }}>{value}</p>
                  <p style={{ fontSize: 12, color: '#64748B', marginTop: 2 }}>{label}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Sessions list */}
        {sessions.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {sessions.map(session => {
              const id = session.session_id || session.id;
              const isExpanded = expandedId === id;
              const dur = formatDuration(session.metadata?.duration || session.duration);

              return (
                <div key={id} style={{
                  background: '#fff', border: '1px solid #E2E8F0', borderRadius: 14,
                  overflow: 'hidden', boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
                }}>
                  {/* Row */}
                  <div
                    onClick={() => setExpandedId(isExpanded ? null : id)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 14, padding: '16px 20px',
                      cursor: 'pointer', transition: 'background 0.15s'
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = '#F8FAFC'}
                    onMouseLeave={e => e.currentTarget.style.background = '#fff'}
                  >
                    {/* Icon */}
                    <div style={{
                      width: 44, height: 44, borderRadius: 12, background: '#EFF6FF',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0
                    }}>
                      <Radio size={20} color="#2563EB" />
                    </div>

                    {/* Info */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{
                        fontSize: 15, fontWeight: 700, color: '#0F172A', whiteSpace: 'nowrap',
                        overflow: 'hidden', textOverflow: 'ellipsis'
                      }}>
                        {session.title || 'Untitled Session'}
                      </p>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 3, flexWrap: 'wrap' }}>
                        <span style={{ fontSize: 12, color: '#64748B' }}>📅 {formatDate(session.created_at)}</span>
                        {dur && <span style={{ fontSize: 12, color: '#64748B' }}>⏱ {dur}</span>}
                        <span style={{ fontSize: 12, color: '#64748B' }}>💬 {session.caption_count || 0} captions</span>
                        <span style={{
                          fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999,
                          background: '#DBEAFE', color: '#2563EB'
                        }}>LIVE</span>
                      </div>
                    </div>

                    <ChevronDown size={18} color="#94A3B8"
                      style={{ flexShrink: 0, transform: isExpanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                  </div>

                  {/* Expanded */}
                  {isExpanded && (
                    <div style={{ padding: '0 20px 20px', borderTop: '1px solid #F1F5F9' }}>
                      <p style={{ fontSize: 12, color: '#94A3B8', fontStyle: 'italic', margin: '12px 0 10px' }}>
                        No video file — this was a live session. Download the transcript below.
                      </p>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, marginBottom: 10 }}>
                        {['srt', 'vtt', 'txt', 'summary'].map(fmt => (
                          <button key={fmt} onClick={() => handleDownload(session, fmt)}
                            style={{
                              padding: '8px 0', borderRadius: 8, border: '1.5px solid #BFDBFE',
                              background: '#EFF6FF', color: '#2563EB', fontWeight: 700,
                              fontSize: 12, cursor: 'pointer', display: 'flex',
                              alignItems: 'center', justifyContent: 'center', gap: 4,
                              transition: 'all 0.15s'
                            }}
                            onMouseEnter={e => { e.currentTarget.style.background = '#2563EB'; e.currentTarget.style.color = '#fff'; }}
                            onMouseLeave={e => { e.currentTarget.style.background = '#EFF6FF'; e.currentTarget.style.color = '#2563EB'; }}>
                            <Download size={12} /> {fmt.toUpperCase()}
                          </button>
                        ))}
                      </div>
                      <button onClick={() => handleDelete(session)}
                        style={{
                          width: '100%', padding: '8px 16px', borderRadius: 8, border: '1.5px solid #FECACA',
                          background: '#FEF2F2', color: '#DC2626', fontWeight: 600,
                          fontSize: 13, cursor: 'pointer', display: 'flex',
                          alignItems: 'center', justifyContent: 'center', gap: 6
                        }}>
                        <Trash2 size={14} /> Delete Session
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          /* Empty state */
          <div style={{ background: '#fff', border: '1.5px dashed #BFDBFE', borderRadius: 16, padding: 64, textAlign: 'center' }}>
            <div style={{ fontSize: 56, marginBottom: 16 }}>📡</div>
            <h3 style={{ fontSize: 22, fontWeight: 700, color: '#0F172A', marginBottom: 8 }}>No live sessions yet</h3>
            <p style={{ fontSize: 14, color: '#64748B', marginBottom: 28, maxWidth: 380, margin: '0 auto 28px' }}>
              Start a live Accessify classroom and the full transcript will be saved here automatically.
            </p>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Link to="/room/create">
                <button style={{
                  padding: '10px 22px', borderRadius: 10, border: 'none',
                  background: '#2563EB', color: '#fff', fontWeight: 600,
                  fontSize: 14, cursor: 'pointer'
                }}>
                  🎥 Start a Live Class
                </button>
              </Link>
              <Link to="/my-videos">
                <button style={{
                  padding: '10px 22px', borderRadius: 10, border: '1.5px solid #E2E8F0',
                  background: '#fff', color: '#374151', fontWeight: 600,
                  fontSize: 14, cursor: 'pointer'
                }}>
                  📼 View My Videos
                </button>
              </Link>
            </div>
          </div>
        )}

        {/* Delete modal */}
        <Modal isOpen={isDeleteModalOpen} onClose={() => setIsDeleteModalOpen(false)}
          title="Delete Live Session?"
          footer={<>
            <Button variant="secondary" onClick={() => setIsDeleteModalOpen(false)}>Cancel</Button>
            <Button variant="danger" onClick={confirmDelete}>Delete</Button>
          </>}>
          <p style={{ fontSize: 14, color: '#374151' }}>
            Delete <strong style={{ color: '#0F172A' }}>"{sessionToDelete?.title}"</strong>?
            The transcript will be permanently removed.
          </p>
        </Modal>

      </div>
    </div>
  );
}
