import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import Peer from 'peerjs';
import { CaptionPanel } from '../components/captions/CaptionPanel';
import { CaptionSizeControl } from '../components/settings/CaptionSizeControl';
import { LanguageSelector } from '../components/settings/LanguageSelector';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { SoundEventBanner } from '../components/captions/SoundEventBanner';
import { useWebSocket } from '../hooks/useWebSocket';
import { useCaptions } from '../context/CaptionContext';
import { useTheme } from '../context/ThemeContext';

// ─── PeerJS Cloud Broker Config ─────────────────────────────────────────────
// No host/port/path = uses PeerJS public cloud broker (0.peerjs.com)
// No local server needed!
const PEER_CONFIG = {
  debug: 1,
  config: {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
      { urls: 'stun:global.stun.twilio.com:3478' },
    ],
  },
};

export function StudentRoom() {
  const { code, participantId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  // Read language selected on the Join page (passed via navigation state)
  const joinedLanguage = location.state?.language || 'en';
  const [language, setLanguage] = useState(joinedLanguage);
  const [showTranslations, setShowTranslations] = useState(joinedLanguage !== 'en');
  const [soundEvent, setSoundEvent] = useState(null);
  const [liveCaption, setLiveCaption] = useState(null);   // Shown on-video like movie subs
  const [captionsVisible, setCaptionsVisible] = useState(true);
  const [roomError, setRoomError] = useState(null);  // inline error — never navigate away
  const [screenTimeRemaining, setScreenTimeRemaining] = useState(null); // Screen time tracking

  const liveCaptionTimer = useRef(null);
  const languageRef = useRef(language); // always reflects latest language in WS callback
  const { captions, addCaption, updateCaption, captionSize, setCaptionSize, clearCaptions } = useCaptions();
  const { isDark, toggleTheme } = useTheme();

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

  const teacherVideoRef = useRef(null);
  const selfVideoRef = useRef(null);
  const peerRef = useRef(null);
  const activeCallRef = useRef(null);
  const selfStreamRef = useRef(null);
  const wordRevealInterval = useRef(null); // for word-by-word animation

  const [isVideoConnected, setIsVideoConnected] = useState(false);
  const [videoError, setVideoError] = useState(null);
  const [teacherPeerId, setTeacherPeerId] = useState(null);
  const [peerReady, setPeerReady] = useState(false);
  const [selfCamOn, setSelfCamOn] = useState(false);

  // Streams in STATE so useEffects reliably sync them to video elements
  const [teacherStream, setTeacherStream] = useState(null);
  const [selfStream, setSelfStream] = useState(null);

  // Sync teacher stream → video element
  useEffect(() => {
    const video = teacherVideoRef.current;
    if (!video) return;
    video.srcObject = teacherStream || null;
    if (teacherStream) {
      video.play().catch((e) => console.warn('[Student] teacher play error:', e));
    }
  }, [teacherStream]);

  // Sync self stream → video element
  useEffect(() => {
    const video = selfVideoRef.current;
    if (!video) return;
    video.srcObject = selfStream || null;
    if (selfStream) {
      video.play().catch((e) => console.warn('[Student] self play error:', e));
    }
  }, [selfStream]);

  // ─── Word-by-word reveal (YouTube/movie caption style) ────────────────────
  const revealWordByWord = useCallback((text, msPerWord = 140) => {
    // Cancel any in-progress reveal
    if (wordRevealInterval.current) clearInterval(wordRevealInterval.current);
    if (liveCaptionTimer.current) clearTimeout(liveCaptionTimer.current);

    const words = text.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return;

    let idx = 1;
    setLiveCaption(words[0]); // show first word immediately

    wordRevealInterval.current = setInterval(() => {
      idx += 1;
      setLiveCaption(words.slice(0, idx).join(' '));
      if (idx >= words.length) {
        clearInterval(wordRevealInterval.current);
        wordRevealInterval.current = null;
        // Auto-clear 5s after last word
        liveCaptionTimer.current = setTimeout(() => setLiveCaption(null), 5000);
      }
    }, msPerWord);
  }, []);

  // ─── Get student's own camera ──────────────────────────────────────────────
  const startOwnCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,  // Simplest — let browser pick camera
        audio: false,
      });
      selfStreamRef.current = stream;
      setSelfStream(stream);
      setSelfCamOn(true);
      return stream;
    } catch (err) {
      console.warn('[Student] Could not access own camera:', err.name, err.message);
      return null;
    }
  }, []);


  // ─── Initialize PeerJS on LOCAL server ────────────────────────────────────
  useEffect(() => {
    const peer = new Peer(undefined, PEER_CONFIG);
    peerRef.current = peer;

    peer.on('open', (id) => {
      console.log('[Student] Peer ready, ID:', id);
      setPeerReady(true);
    });

    peer.on('error', (err) => {
      console.error('[Student] PeerJS error:', err.type, err);
      setVideoError(`PeerJS error: ${err.type}`);
    });

    // Camera starts ON DEMAND (startCall or Enable Cam button)
    // Do NOT auto-start here — it grabs the webcam and conflicts with teacher's camera

    return () => {
      peer.destroy();
      peerRef.current = null;
      // Stop own camera
      if (selfStreamRef.current) {
        selfStreamRef.current.getTracks().forEach((t) => t.stop());
        selfStreamRef.current = null;
      }
    };
  }, [startOwnCamera]);

  // ─── Call teacher with student's own camera ───────────────────────────────
  const startCall = useCallback(async () => {
    const peer = peerRef.current;
    if (!peer || !teacherPeerId) return;

    // Close any existing call first
    if (activeCallRef.current) {
      activeCallRef.current.close();
      activeCallRef.current = null;
    }

    console.log('[Student] Calling teacher peer:', teacherPeerId);
    setVideoError(null);
    setIsVideoConnected(false);

    // Get own camera stream (or use existing, or empty MediaStream as fallback)
    let streamToSend = selfStreamRef.current;
    if (!streamToSend) {
      streamToSend = await startOwnCamera();
    }
    // Fallback: use empty stream if camera not available
    if (!streamToSend) {
      streamToSend = new MediaStream();
    }

    const call = peer.call(teacherPeerId, streamToSend, {
      metadata: { participantId },
    });

    if (!call) {
      setVideoError('Failed to connect to teacher. Try refreshing.');
      return;
    }

    activeCallRef.current = call;

    // Teacher sends back their camera stream
    call.on('stream', (recvStream) => {
      console.log('[Student] Teacher stream received! Tracks:', recvStream.getTracks().length);
      setTeacherStream(recvStream);  // ← triggers useEffect to set srcObject safely
      setIsVideoConnected(true);
      setVideoError(null);
    });

    call.on('error', (err) => {
      console.error('[Student] Call error:', err);
      setVideoError(`Video error: ${err.type || err.message}`);
      setIsVideoConnected(false);
    });

    call.on('close', () => {
      console.log('[Student] Call closed by teacher');
      setIsVideoConnected(false);
      activeCallRef.current = null;
    });
  }, [teacherPeerId, participantId, startOwnCamera]);

  // ─── When peer is ready and we have teacher ID — call teacher ────────────
  useEffect(() => {
    if (!peerReady || !teacherPeerId) return;
    const peer = peerRef.current;
    if (!peer) return;

    if (peer.open) {
      startCall();
    } else {
      peer.once('open', startCall);
    }

    return () => {
      if (activeCallRef.current) {
        activeCallRef.current.close();
        activeCallRef.current = null;
      }
    };
  }, [peerReady, teacherPeerId, startCall]);

  // ─── WebSocket — receives captions + teacher peer ID ─────────────────────
  const { isConnected, isReconnecting, send } = useWebSocket(
    `/ws/room/${code}/student/${participantId}`,
    (message) => {
      if (message.type === 'caption') {
        const captionId = Date.now();
        const currentLang = languageRef.current;

        // Use backend translation if available; otherwise fetch client-side
        // (backend may miss translation due to timing/language-state races)
        const backendTranslation = message.translation || null;

        addCaption({
          id: captionId,
          text: message.text,
          simplified_text: message.simplified_text,
          keywords: message.keywords || [],
          tone: message.tone || {},
          translation: backendTranslation,
          sound_event: message.sound_event || null,
          timestamp: message.timestamp || Date.now(),
        });

        // If backend didn't translate, fetch translation client-side
        if (currentLang !== 'en' && !backendTranslation) {
          (async () => {
            try {
              const res = await fetch(`${API_BASE}/api/v1/translate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: message.text, target_lang: currentLang }),
              });
              const data = await res.json();
              if (data.translated) {
                updateCaption(captionId, {
                  translation: { text: data.translated, target_language: currentLang },
                });
              }
            } catch (err) {
              console.warn('[Translation fetch failed]', err);
            }
          })();
        }

        // ── Live in-video subtitle (cinema style) ──────────────────────────
        const displayText = (message.translation?.text) || message.text;
        revealWordByWord(displayText, 140);

        if (message.sound_event) {
          setSoundEvent(message.sound_event);
          setTimeout(() => setSoundEvent(null), 4000);
        }
      } else if (message.type === 'sound_event') {
        // ── Standalone non-speech sound event (clap, laugh, noise, door) ──
        const display = message.display || `${message.emoji || '🔊'} ${message.event || 'SOUND'}`;
        setSoundEvent(display);
        setTimeout(() => setSoundEvent(null), 4000);
        // Also push a visual card into the captions list
        addCaption({
          id: Date.now(),
          text: '',
          simplified_text: '',
          keywords: [],
          tone: {},
          translation: null,
          sound_event: display,
          timestamp: Date.now(),
        });
      } else if (message.type === 'teacher_peer_id') {
        console.log('[Student] Got teacher peer ID:', message.peer_id);
        setTeacherPeerId((prev) => (prev !== message.peer_id ? message.peer_id : prev));
      } else if (message.type === 'error') {
        console.warn('[WS] Server error:', message.message);
        setRoomError(message.message);
        // Auto-clear after 8s so transient errors don't stay forever
        setTimeout(() => setRoomError(null), 8000);
      } else if (message.type === 'screen_time_info' || message.type === 'screen_time_warning') {
        setScreenTimeRemaining(message.remaining_minutes);
      } else if (message.type === 'connected' || message.type === 'pong' || message.type === 'language_changed') {
        // On first connect: if student joined with a non-English language, tell backend now
        // so the very first caption is already translated
        if (message.type === 'connected' && joinedLanguage !== 'en') {
          send({ type: 'change_language', language: joinedLanguage });
        }
      }
    },
    true
  );

  const handleLanguageChange = (lang) => {
    setLanguage(lang);
    languageRef.current = lang;  // keep ref in sync for WS closure
    setShowTranslations(lang !== 'en');
    send({ type: 'change_language', language: lang });
  };

  return (
    <div className="min-h-screen bg-bg-dark">
      {/* Header */}
      <div className="bg-black border-b-2 border-primary/20 sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-xl font-bold text-white">📡 Live Class — {code}</h1>
              <div className="flex items-center gap-3">
                <p className={`text-sm font-medium ${isConnected ? 'text-accent' : isReconnecting ? 'text-yellow-400' : 'text-red-400'}`}>
                  {isReconnecting ? '🔄 Reconnecting...' : isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                </p>
                {screenTimeRemaining !== null && screenTimeRemaining <= 10 && (
                  <span className="text-sm font-bold text-orange-400 animate-pulse">
                    ⏳ {Math.ceil(screenTimeRemaining)}m remaining
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <Button variant="ghost" size="sm" onClick={toggleTheme}>
                {isDark ? '☀️ Light' : '🌙 Dark'}
              </Button>
              <Button variant="secondary" size="sm" onClick={() => navigate('/dashboard')}>
                🚪 Leave
              </Button>
            </div>
          </div>
        </div>
      </div>

      {soundEvent && (
        <div className="max-w-7xl mx-auto px-4 pt-3">
          <SoundEventBanner event={soundEvent} duration={4000} />
        </div>
      )}

      {roomError && (
        <div className="max-w-7xl mx-auto px-4 pt-2">
          <div className="bg-red-900/40 border border-red-500/50 rounded-lg px-4 py-2 text-red-300 text-sm flex items-center justify-between">
            <span>⚠️ {roomError}</span>
            <button onClick={() => setRoomError(null)} className="text-red-400 hover:text-white ml-4">✕</button>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* Main area */}
          <div className="lg:col-span-3 space-y-6">

            {/* Video area — Teacher + Own Camera side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

              {/* Teacher's video — always rendered, overlay shows when not connected */}
              <Card className="p-0 overflow-hidden">
                <div className="bg-gray-900 aspect-video relative overflow-hidden">
                  <video
                    ref={teacherVideoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  {!isVideoConnected && (
                    <div className="text-center absolute inset-0 flex flex-col items-center justify-center bg-gray-900">
                      <span className="text-5xl mb-3">👩‍🏫</span>
                      <p className="text-white font-semibold">Teacher</p>
                      <p className="text-xs mt-2">
                        {videoError ? (
                          <span className="text-red-400">{videoError}</span>
                        ) : teacherPeerId ? (
                          <span className="text-yellow-400">⏳ Connecting...</span>
                        ) : (
                          <span className="text-gray-500">Waiting for teacher...</span>
                        )}
                      </p>
                      {teacherPeerId && !isVideoConnected && peerReady && (
                        <Button variant="outline" size="sm" className="mt-3" onClick={startCall}>
                          🔄 Retry Video
                        </Button>
                      )}
                    </div>
                  )}

                  {/* ── Cinema-style subtitle overlay ──────────────────────── */}
                  {captionsVisible && liveCaption && (
                    <div
                      className="absolute bottom-8 left-0 right-0 flex justify-center px-6 pointer-events-none"
                      style={{ zIndex: 20 }}
                      key={liveCaption}
                    >
                      <div style={{
                        background: 'rgba(0,0,0,0.75)',
                        backdropFilter: 'blur(4px)',
                        borderRadius: '8px',
                        padding: '8px 18px',
                        maxWidth: '90%',
                        animation: 'subtitleFadeIn 0.2s ease-out',
                      }}>
                        <p style={{
                          color: '#fff',
                          fontSize: '1.15rem',
                          fontWeight: 600,
                          lineHeight: 1.5,
                          textAlign: 'center',
                          textShadow: '0 2px 8px rgba(0,0,0,0.9)',
                        }}>
                          {liveCaption}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* CSS animation */}
                  <style>{`
                  @keyframes subtitleFadeIn {
                    from { opacity: 0; transform: translateY(6px); }
                    to   { opacity: 1; transform: translateY(0); }
                  }
                `}</style>

                  {/* Top-left badge */}
                  <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
                    {isVideoConnected ? '🔴 LIVE' : '📡 Teacher'}
                  </div>

                  {/* CC toggle */}
                  <button
                    onClick={() => setCaptionsVisible(!captionsVisible)}
                    className={`absolute top-2 right-2 text-xs font-bold px-2 py-1 rounded border transition-all ${captionsVisible ? 'bg-white text-black border-white' : 'bg-black/60 text-gray-400 border-gray-600'
                      }`}
                    title="Toggle captions on video"
                  >
                    CC
                  </button>
                </div>
              </Card>

              {/* Student's own camera — always rendered, overlay when off */}
              <Card className="p-0 overflow-hidden">
                <div className="bg-gray-800 aspect-video relative overflow-hidden">
                  <video
                    ref={selfVideoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  {!selfCamOn && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 text-center">
                      <span className="text-4xl">🙍</span>
                      <p className="text-gray-500 text-sm mt-2">Your camera</p>
                      <Button variant="outline" size="sm" className="mt-2" onClick={startOwnCamera}>
                        📷 Enable Cam
                      </Button>
                    </div>
                  )}
                  <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
                    👤 You
                  </div>
                </div>
              </Card>
            </div>

            {/* Captions */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xl font-bold text-white">📝 Live Captions</h2>
                <div className="flex gap-2">
                  <Button variant="ghost" size="sm" onClick={() => setShowTranslations(!showTranslations)}>
                    {showTranslations ? '🌐 Hide Translation' : '🌐 Translate'}
                  </Button>
                  {captions.length > 0 && (
                    <Button variant="ghost" size="sm" onClick={clearCaptions}>🗑</Button>
                  )}
                </div>
              </div>
              {captions.length === 0 ? (
                <Card>
                  <div className="text-center py-12 text-gray-500">
                    <p className="text-4xl mb-3">👂</p>
                    <p className="text-lg">Waiting for teacher to speak...</p>
                  </div>
                </Card>
              ) : (
                <CaptionPanel
                  captions={captions}
                  showEmojis={true}
                  showTranslation={showTranslations}
                  language={language}
                  maxHeight="max-h-[400px]"
                />
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <h3 className="text-base font-bold text-white mb-3">🌐 Language</h3>
              <LanguageSelector value={language} onChange={handleLanguageChange} variant="pills" />
              <p className="text-xs text-gray-500 mt-2">Changes captions in real-time</p>
            </Card>

            <Card>
              <h3 className="text-base font-bold text-white mb-3">📏 Caption Size</h3>
              <CaptionSizeControl size={captionSize} onChange={setCaptionSize} />
            </Card>

            <Card className={isConnected ? 'border-l-4 border-accent' : 'border-l-4 border-red-500'}>
              <h3 className="text-base font-bold text-white mb-3">🔗 Status</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span>{isConnected ? '🟢' : isReconnecting ? '🟡' : '🔴'}</span>
                  <span className="text-gray-400">{isReconnecting ? 'Reconnecting...' : isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>📡</span><span className="text-gray-400">Room: {code}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>👤</span><span className="text-gray-400 text-xs break-all">{participantId}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span>📹</span>
                  <span className={isVideoConnected ? 'text-accent' : 'text-gray-400'}>
                    {isVideoConnected ? 'Teacher Video Live' : teacherPeerId ? 'Connecting...' : 'No Video Yet'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span>🙍</span>
                  <span className={selfCamOn ? 'text-accent' : 'text-gray-400'}>
                    {selfCamOn ? 'Your Cam Active' : 'Cam Off'}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
