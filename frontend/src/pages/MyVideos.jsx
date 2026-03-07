import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Modal } from '../components/ui/Modal';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export function MyVideos() {
    const [sessions, setSessions] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [sessionToDelete, setSessionToDelete] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchSessions = async () => {
            try {
                setIsLoading(true);
                const response = await fetch(`${API_BASE}/api/v1/sessions/`);
                if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`);
                const data = await response.json();
                const all = data.sessions || data || [];
                setSessions(all.filter((s) => (s.session_type || s.type) === 'video'));
            } catch (err) {
                setError(err.message);
                setSessions([]);
            } finally {
                setIsLoading(false);
            }
        };
        fetchSessions();
    }, []);

    const formatDate = (iso) => {
        if (!iso) return '—';
        return new Date(iso).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
        });
    };

    const formatDuration = (secs) => {
        if (!secs) return null;
        const m = Math.floor(secs / 60);
        const s = Math.floor(secs % 60);
        return `${m}:${String(s).padStart(2, '0')}`;
    };

    const handleDownload = async (session, format) => {
        const sessionId = session.session_id || session.id;
        try {
            const res = await fetch(`${API_BASE}/api/v1/export/${sessionId}/${format}`);
            if (!res.ok) throw new Error('Download failed');
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${session.title || sessionId}.${format}`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert(`Download failed: ${err.message}`);
        }
    };

    const handleDelete = (session) => {
        setSessionToDelete(session);
        setIsDeleteModalOpen(true);
    };

    const confirmDelete = async () => {
        if (!sessionToDelete) return;
        try {
            await fetch(`${API_BASE}/api/v1/sessions/${sessionToDelete.session_id || sessionToDelete.id}`, {
                method: 'DELETE',
            });
            setSessions(sessions.filter(
                (s) => (s.session_id || s.id) !== (sessionToDelete.session_id || sessionToDelete.id)
            ));
        } catch (err) {
            console.error('Delete error:', err);
        } finally {
            setIsDeleteModalOpen(false);
            setSessionToDelete(null);
        }
    };

    const goToPlayer = (session) => {
        navigate('/player', { state: { videoData: session } });
    };

    if (isLoading) {
        return (
            <div className="min-h-screen bg-bg-dark flex items-center justify-center">
                <div className="text-center">
                    <div className="text-5xl mb-4">🎬</div>
                    <p className="text-purple-400 font-semibold">Loading your videos...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen py-12" style={{ background: 'linear-gradient(135deg, #0F0F1A 0%, #160d2b 100%)' }}>
            <div className="max-w-7xl mx-auto px-4">

                {/* ── Header with purple accent (distinct from History teal) ── */}
                <div className="mb-10 flex items-start justify-between flex-wrap gap-4">
                    <div>
                        <div className="flex items-center gap-3 mb-3">
                            <div
                                className="flex items-center justify-center rounded-2xl text-3xl flex-shrink-0"
                                style={{ width: 56, height: 56, background: 'linear-gradient(135deg, #7c3aed, #6d28d9)' }}
                            >
                                📼
                            </div>
                            <div>
                                <h1 className="text-4xl font-bold text-white">My Videos</h1>
                                <p className="text-purple-400 text-sm font-medium mt-1">AI-captioned video library</p>
                            </div>
                        </div>
                        <p className="text-gray-400 text-sm ml-1">
                            Videos you uploaded and processed with Whisper AI captions. Click any video to play with live captions.
                            Want session transcripts?{' '}
                            <Link to="/history" className="text-teal-400 underline hover:text-teal-300 transition-colors">
                                Go to Live History →
                            </Link>
                        </p>
                    </div>
                    <Link to="/upload">
                        <button
                            className="px-5 py-2.5 rounded-xl font-bold text-sm text-white transition-all hover:scale-105"
                            style={{ background: 'linear-gradient(135deg, #7c3aed, #6d28d9)' }}
                        >
                            + Upload New Video
                        </button>
                    </Link>
                </div>

                {/* ── Purple accent rule ── */}
                <div className="mb-8 h-1 rounded-full" style={{ background: 'linear-gradient(90deg, #7c3aed, transparent)' }} />

                {/* ── Error ── */}
                {error && (
                    <div className="mb-6 p-4 rounded-xl border border-purple-700/50 bg-purple-900/20">
                        <p className="text-purple-300 text-sm">⚠️ {error}</p>
                    </div>
                )}

                {/* ── Stats bar ── */}
                {sessions.length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        {[
                            { icon: '🎬', label: 'Total Videos', value: sessions.length },
                            {
                                icon: '⏱️', label: 'Total Duration',
                                value: (() => {
                                    const total = sessions.reduce((a, s) => a + (s.metadata?.duration || s.duration || 0), 0);
                                    return total ? `${Math.floor(total / 60)} min` : '—';
                                })(),
                            },
                            { icon: '💬', label: 'Caption Cues', value: sessions.reduce((a, s) => a + (s.caption_count || 0), 0) },
                            {
                                icon: '🌐', label: 'Languages',
                                value: [...new Set(sessions.map((s) => (s.language || 'en').toUpperCase()))].join(', '),
                            },
                        ].map((stat) => (
                            <div
                                key={stat.label}
                                className="rounded-xl p-4 text-center"
                                style={{ background: 'rgba(124, 58, 237, 0.1)', border: '1px solid rgba(124, 58, 237, 0.25)' }}
                            >
                                <div className="text-2xl mb-1">{stat.icon}</div>
                                <div className="text-xl font-bold text-purple-300">{stat.value}</div>
                                <div className="text-xs text-gray-400">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                )}

                {/* ── Cinematic video card grid ── */}
                {sessions.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {sessions.map((session) => {
                            const sessionId = session.session_id || session.id;
                            const duration = session.metadata?.duration || session.duration || 0;
                            const durationStr = formatDuration(duration);
                            const filename = session.title || session.metadata?.filename || 'Video';

                            return (
                                <div
                                    key={sessionId}
                                    className="rounded-2xl overflow-hidden flex flex-col group"
                                    style={{ background: 'rgba(124, 58, 237, 0.07)', border: '1px solid rgba(124, 58, 237, 0.2)', transition: 'border-color 0.2s, transform 0.2s' }}
                                    onMouseEnter={(e) => e.currentTarget.style.borderColor = 'rgba(124,58,237,0.5)'}
                                    onMouseLeave={(e) => e.currentTarget.style.borderColor = 'rgba(124,58,237,0.2)'}
                                >
                                    {/* ── Cinematic thumbnail ── */}
                                    <div
                                        className="relative flex items-center justify-center cursor-pointer overflow-hidden"
                                        style={{
                                            height: 160,
                                            background: 'linear-gradient(135deg, #1e1040 0%, #0f0720 100%)',
                                        }}
                                        onClick={() => goToPlayer(session)}
                                    >
                                        {/* Film grain texture */}
                                        <div style={{
                                            position: 'absolute', inset: 0, opacity: 0.07,
                                            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.5) 2px, rgba(255,255,255,0.5) 4px)',
                                        }} />

                                        {/* Play button */}
                                        <div
                                            className="relative z-10 flex items-center justify-center rounded-full transition-transform group-hover:scale-105"
                                            style={{ width: 60, height: 60, background: 'rgba(124,58,237,0.85)', boxShadow: '0 0 30px rgba(124,58,237,0.5)' }}
                                        >
                                            <span className="text-white text-2xl ml-1">▶</span>
                                        </div>

                                        {/* Duration badge */}
                                        {durationStr && (
                                            <div
                                                className="absolute bottom-2 right-2 rounded px-2 py-0.5 text-xs font-bold text-white"
                                                style={{ background: 'rgba(0,0,0,0.75)' }}
                                            >
                                                {durationStr}
                                            </div>
                                        )}

                                        {/* CC badge */}
                                        <div
                                            className="absolute top-2 right-2 rounded px-2 py-0.5 text-xs font-bold"
                                            style={{ background: 'rgba(124,58,237,0.85)', color: '#e9d5ff' }}
                                        >
                                            CC
                                        </div>
                                    </div>

                                    {/* ── Info ── */}
                                    <div className="flex flex-col flex-grow p-4">
                                        <h3
                                            className="font-bold text-white mb-1 truncate cursor-pointer hover:text-purple-300 transition-colors"
                                            title={filename}
                                            onClick={() => goToPlayer(session)}
                                        >
                                            {filename}
                                        </h3>
                                        <div className="flex items-center gap-3 text-xs text-gray-400 mb-4">
                                            <span>📅 {formatDate(session.created_at)}</span>
                                            <span>💬 {session.caption_count || 0} cues</span>
                                            <span className="text-purple-400">{(session.language || 'en').toUpperCase()}</span>
                                        </div>

                                        {/* ── Actions ── */}
                                        <div className="mt-auto space-y-2">
                                            <button
                                                className="w-full py-2.5 rounded-xl text-sm font-bold text-white transition-all hover:opacity-90"
                                                style={{ background: 'linear-gradient(135deg, #7c3aed, #6d28d9)' }}
                                                onClick={() => goToPlayer(session)}
                                            >
                                                ▶ Play with Captions
                                            </button>

                                            <details className="w-full">
                                                <summary
                                                    className="cursor-pointer py-2 px-3 rounded-xl text-xs font-bold text-center transition-all list-none"
                                                    style={{ background: 'rgba(124,58,237,0.15)', color: '#c084fc', border: '1px solid rgba(124,58,237,0.3)' }}
                                                >
                                                    ⬇ Download Captions
                                                </summary>
                                                <div className="mt-2 grid grid-cols-2 gap-1">
                                                    {['srt', 'vtt', 'txt', 'summary'].map((fmt) => (
                                                        <button
                                                            key={fmt}
                                                            onClick={() => handleDownload(session, fmt)}
                                                            className="py-1.5 px-2 rounded-lg text-xs font-semibold transition-colors"
                                                            style={{ background: 'rgba(124,58,237,0.1)', color: '#a855f7', border: '1px solid rgba(124,58,237,0.2)' }}
                                                        >
                                                            {fmt.toUpperCase()}
                                                        </button>
                                                    ))}
                                                </div>
                                            </details>

                                            <button
                                                className="w-full py-2 rounded-xl text-xs font-bold text-red-400 border border-red-800/40 bg-red-900/10 hover:bg-red-900/20 transition-colors"
                                                onClick={() => handleDelete(session)}
                                            >
                                                🗑 Delete
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    /* ── Empty state ── */
                    <div
                        className="rounded-2xl p-16 text-center"
                        style={{ background: 'rgba(124,58,237,0.06)', border: '1px dashed rgba(124,58,237,0.3)' }}
                    >
                        <div className="text-7xl mb-5">🎬</div>
                        <h3 className="text-2xl font-bold text-white mb-3">No videos yet</h3>
                        <p className="text-gray-400 mb-8 max-w-md mx-auto">
                            Upload a video recording (.mp4, .mkv, .mov...) and Accessify will generate real-time AI captions using Whisper.
                        </p>
                        <Link to="/upload">
                            <button
                                className="px-8 py-3 rounded-xl font-bold text-white text-lg transition-all hover:scale-105"
                                style={{ background: 'linear-gradient(135deg, #7c3aed, #6d28d9)', boxShadow: '0 0 30px rgba(124,58,237,0.4)' }}
                            >
                                🚀 Upload Your First Video
                            </button>
                        </Link>
                    </div>
                )}

                {/* ── Delete Modal ── */}
                <Modal
                    isOpen={isDeleteModalOpen}
                    onClose={() => setIsDeleteModalOpen(false)}
                    title="Delete Video?"
                    size="md"
                    footer={
                        <>
                            <Button variant="secondary" onClick={() => setIsDeleteModalOpen(false)}>Cancel</Button>
                            <Button variant="danger" onClick={confirmDelete}>Delete</Button>
                        </>
                    }
                >
                    <p className="text-gray-400">
                        Delete <strong className="text-white">"{sessionToDelete?.title}"</strong>? This removes the session and all captions permanently.
                    </p>
                </Modal>
            </div>
        </div>
    );
}
