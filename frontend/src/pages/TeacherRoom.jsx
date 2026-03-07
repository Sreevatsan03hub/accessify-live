import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Peer from 'peerjs';
import { CaptionPanel } from '../components/captions/CaptionPanel';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { useMicrophone } from '../hooks/useMicrophone';
import { useWebSocket } from '../hooks/useWebSocket';
import { useCaptions } from '../context/CaptionContext';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

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

export function TeacherRoom() {
  const { code } = useParams();
  const navigate = useNavigate();
  const [isLive, setIsLive] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [studentCount, setStudentCount] = useState(0);
  const [wsPath, setWsPath] = useState(null);
  const [wsError, setWsError] = useState(null);  // inline error — never navigate

  // Student video tiles: { [peerId]: MediaStream }
  const [studentStreams, setStudentStreams] = useState({});
  // cameraStream in STATE so useEffect reliably syncs it to the video element
  const [cameraStream, setCameraStream] = useState(null);

  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const pendingCallsRef = useRef([]);
  const myPeerIdRef = useRef(null);
  const activeCalls = useRef({}); // Track calls per student peerId

  const [myPeerId, setMyPeerId] = useState(null);
  const { captions, addCaption, clearCaptions } = useCaptions();

  // Safety settings
  const [safetySettings, setSafetySettings] = useState({
    screen_time_limit_minutes: 0,
    profanity_filter_enabled: true,
  });

  // Load safety settings on mount
  useEffect(() => {
    if (code) {
      fetch(`${API_BASE}/api/v1/rooms/${code}/safety`)
        .then(res => res.json())
        .then(data => setSafetySettings(data))
        .catch(err => console.warn("[Teacher] Failed to load safety settings", err));
    }
  }, [code]);

  const updateSafety = async (newSettings) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/rooms/${code}/safety`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...safetySettings, ...newSettings }),
      });
      const data = await res.json();
      setSafetySettings(data);
    } catch (err) {
      console.error("[Teacher] Failed to update safety settings", err);
    }
  };

  // ─── Sync camera stream to video element (React-safe pattern) ─────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = cameraStream || null;
    if (cameraStream) {
      video.play().catch((e) => console.warn('[Teacher] play error:', e));
    }
  }, [cameraStream]);

  // ─── Initialize PeerJS on LOCAL server ───────────────────────────────────────
  useEffect(() => {
    if (!code) return;

    const peerInstance = new Peer(undefined, PEER_CONFIG);

    peerInstance.on('open', (id) => {
      console.log('[Teacher] Peer ready, ID:', id);
      myPeerIdRef.current = id;
      setMyPeerId(id);
    });

    peerInstance.on('error', (err) => {
      console.error('[Teacher] PeerJS error:', err.type, err);
    });

    // Receive call from a student (student sends their camera → teacher)
    peerInstance.on('call', (call) => {
      console.log('[Teacher] Incoming video call from student peer:', call.peer);

      // Set up stream/close/error handlers BEFORE answering
      call.on('stream', (studentStream) => {
        console.log('[Teacher] Received student stream from:', call.peer,
          'tracks:', studentStream.getTracks().length);
        setStudentStreams((prev) => ({ ...prev, [call.peer]: studentStream }));
        activeCalls.current[call.peer] = call;
      });

      call.on('close', () => {
        console.log('[Teacher] Student call closed:', call.peer);
        setStudentStreams((prev) => {
          const next = { ...prev };
          delete next[call.peer];
          return next;
        });
        delete activeCalls.current[call.peer];
      });

      call.on('error', (err) => {
        console.error('[Teacher] Student call error:', err);
      });

      // Answer with camera stream if ready, otherwise queue (do NOT answer with empty stream)
      if (cameraStreamRef.current) {
        call.answer(cameraStreamRef.current);
      } else {
        console.log('[Teacher] Camera not ready yet, queuing call from:', call.peer);
        pendingCallsRef.current.push(call);
      }
    });

    return () => {
      peerInstance.destroy();
      myPeerIdRef.current = null;
    };
  }, [code]);

  // ─── Audio chunk → WebSocket ──────────────────────────────────────────────────
  const handleAudioChunk = useCallback((base64Audio, sampleRate) => {
    send({ type: 'audio_chunk', data: base64Audio, sample_rate: sampleRate });
  }, []);

  const { isActive: micOn, isLoading: micLoading, error: micError, volume, toggleMicrophone } =
    useMicrophone(false, handleAudioChunk);

  // ─── WebSocket ────────────────────────────────────────────────────────────────
  const { isConnected, isReconnecting, send } = useWebSocket(
    wsPath,
    (message) => {
      if (message.type === 'caption_sent') {
        addCaption({
          id: Date.now(),
          text: message.text,
          simplified_text: message.simplified_text,
          keywords: message.keywords || [],
          tone: message.tone || {},
          sound_event: message.sound_event || null,
          translation: null,
          timestamp: Date.now(),
        });
        setStudentCount(message.students_reached || 0);
      } else if (message.type === 'error') {
        console.warn('[WS] Server error:', message.message);
        setWsError(message.message);
        setTimeout(() => setWsError(null), 8000);
      } else if (message.type === 'connected' || message.type === 'pong') {
        // Control messages — no action needed
      }
    },
    !!wsPath
  );

  // ─── Send peer ID to backend whenever WS connects/reconnects ─────────────────
  useEffect(() => {
    if (isConnected && myPeerIdRef.current) {
      send({ type: 'teacher_peer_id', peer_id: myPeerIdRef.current });
    }
  }, [isConnected]);

  // ─── Camera controls ────────────────────────────────────────────────────────────
  const [cameraError, setCameraError] = useState(null);

  const startCamera = async (retryCount = 0) => {
    setCameraError(null);
    try {
      // First try: standard constraints (no facingMode — not supported on Windows)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      }).catch(() =>
        // Immediate fallback: bare video:true if resolution constraints cause timeout
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      );
      cameraStreamRef.current = stream;
      setCameraStream(stream);
      setCameraOn(true);

      if (pendingCallsRef.current.length > 0) {
        pendingCallsRef.current.forEach((call) => {
          try { call.answer(stream); } catch (e) { console.error('[Teacher] pending answer error:', e); }
        });
        pendingCallsRef.current = [];
      }
    } catch (err) {
      console.error('[Teacher] Camera error:', err.name, err.message);

      // NotFoundError, NotReadableError, or AbortError (timeout): camera still releasing or slow to init
      // Retry up to 3 times with 1.5s delay
      if ((err.name === 'NotFoundError' || err.name === 'NotReadableError' || err.name === 'AbortError') && retryCount < 3) {
        console.warn(`[Teacher] Camera not ready (${err.name}), retrying in 1.5s... (attempt ${retryCount + 1}/3)`);
        setTimeout(() => startCamera(retryCount + 1), 1500);
        return;
      }

      // Final fallback: enumerate all video devices and try each one
      if (retryCount >= 3) {
        try {
          console.warn('[Teacher] Trying device enumeration fallback...');
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter(d => d.kind === 'videoinput');
          console.log('[Teacher] Found video devices:', videoDevices.map(d => d.label || d.deviceId));

          for (const device of videoDevices) {
            try {
              const stream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: device.deviceId } },
                audio: false,
              });
              cameraStreamRef.current = stream;
              setCameraStream(stream);
              setCameraOn(true);
              console.log('[Teacher] Camera started via fallback device:', device.label || device.deviceId);
              return;
            } catch (devErr) {
              console.warn('[Teacher] Device failed:', device.label, devErr.message);
            }
          }
        } catch (enumErr) {
          console.error('[Teacher] Device enumeration failed:', enumErr);
        }
      }

      // All retries exhausted — show inline error
      const msg = err.name === 'NotFoundError'
        ? 'Camera not found. Close Teams/Zoom/any other app using your camera, then click "Camera OFF" to retry.'
        : err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access in your browser settings.'
          : `Camera error: ${err.message}`;
      setCameraError(msg);
      setCameraOn(false);
    }
  };

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      cameraStreamRef.current = null;
    }
    setCameraStream(null);  // ← triggers useEffect to clear srcObject
    setCameraOn(false);
    setCameraError(null);
  };

  const handleGoLive = async () => {
    setIsLive(true);
    setWsPath(`/ws/room/${code}/teacher`);
    await startCamera();
  };

  const handleStopBroadcasting = async () => {
    setIsLive(false);
    setWsPath(null);
    if (micOn) toggleMicrophone();
    stopCamera();
    clearCaptions();
    setStudentStreams({});
    try {
      await fetch(`${API_BASE}/api/v1/rooms/${code}/close`, { method: 'POST' });
    } catch (e) { console.warn(e); }
  };

  useEffect(() => {
    return () => { stopCamera(); };
  }, []);

  const studentEntries = Object.entries(studentStreams);

  return (
    <div className="min-h-screen bg-bg-dark">
      {/* Header */}
      <div className="bg-black border-b-2 border-primary/20 sticky top-16 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-2xl font-bold text-white">🎥 Broadcasting: {code}</h1>
              <div className="flex items-center gap-3 mt-1">
                <span className={`text-sm font-semibold ${isLive ? 'text-red-400' : 'text-gray-400'}`}>
                  {isLive ? '🔴 LIVE' : '⚫ Offline'}
                </span>
                {isLive && (
                  <span className={`text-sm ${isConnected ? 'text-accent' : 'text-yellow-400'}`}>
                    {isReconnecting ? '🔄 Reconnecting...' : isConnected ? '🟢 Connected' : '🔴 Disconnected'}
                  </span>
                )}
              </div>
            </div>
            <div className="flex gap-3 flex-wrap">
              {!isLive ? (
                <Button variant="primary" size="lg" onClick={handleGoLive}
                  className="bg-red-600 hover:bg-red-700">🎙️ Go Live
                </Button>
              ) : (
                <Button variant="danger" size="lg" onClick={handleStopBroadcasting}>
                  ⏹️ Stop Broadcasting
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      {wsError && (
        <div className="max-w-7xl mx-auto px-4 pt-2">
          <div className="bg-red-900/40 border border-red-500/50 rounded-lg px-4 py-2 text-red-300 text-sm flex items-center justify-between">
            <span>⚠️ {wsError}</span>
            <button onClick={() => setWsError(null)} className="text-red-400 hover:text-white ml-4">✕</button>
          </div>
        </div>
      )}

      {/* ── Main content ── */}
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">

        {/* ── Full-width camera row ────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">

          {/* Camera — takes 3 of 4 columns for a wide landscape view */}
          <div className="lg:col-span-3">
            <Card className="p-0 overflow-hidden">
              {/* Widescreen 16:9 video */}
              <div className="relative bg-black w-full" style={{ aspectRatio: '16/9' }}>
                <video ref={videoRef} autoPlay muted playsInline
                  className="w-full h-full object-cover" />

                {/* Camera-off overlay */}
                {!cameraOn && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black">
                    {!isLive ? (
                      <>
                        <div className="text-6xl mb-3 opacity-30">🎥</div>
                        <p className="text-gray-500 text-sm">Click Go Live to start</p>
                      </>
                    ) : (
                      <>
                        <div className="text-6xl mb-3 opacity-30">📷</div>
                        <p className="text-gray-500 text-sm">Camera is off</p>
                      </>
                    )}
                  </div>
                )}

                {/* LIVE badge */}
                {isLive && cameraOn && (
                  <div className="absolute top-3 left-3 bg-red-600 text-white text-xs font-bold px-3 py-1 rounded-full animate-pulse shadow-lg">
                    🔴 LIVE
                  </div>
                )}

                {/* Connection badge */}
                {isLive && (
                  <div className="absolute top-3 right-3 bg-black/60 text-xs px-3 py-1 rounded-full">
                    <span className={isConnected ? 'text-green-400' : 'text-yellow-400'}>
                      {isReconnecting ? '🔄 Reconnecting' : isConnected ? '🟢 Connected' : '⚪ Offline'}
                    </span>
                  </div>
                )}
              </div>

              {/* Controls row below video */}
              <div className="p-4 flex items-center gap-3 flex-wrap bg-gray-900/80">
                <Button variant={micOn ? 'danger' : 'primary'} onClick={toggleMicrophone}
                  disabled={!isLive || micLoading} size="sm">
                  {micLoading ? '⏳ Starting...' : micOn ? '🔴 Mic ON' : '🔇 Mic OFF'}
                </Button>
                <Button variant={cameraOn ? 'danger' : 'primary'}
                  onClick={cameraOn ? stopCamera : startCamera}
                  disabled={!isLive} size="sm">
                  {cameraOn ? '🔴 Camera ON' : '📷 Camera OFF'}
                </Button>

                {/* Volume bar inline */}
                {isLive && micOn && (
                  <div className="flex items-center gap-2 ml-auto flex-1 max-w-xs">
                    <span className="text-xs text-accent whitespace-nowrap">🎙️ {Math.round(volume)}%</span>
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div className="bg-accent h-2 rounded-full transition-all duration-100"
                        style={{ width: `${volume}%` }} />
                    </div>
                  </div>
                )}
              </div>

              {micError && (
                <div className="px-4 pb-3">
                  <p className="text-red-400 text-sm">⚠️ {micError}</p>
                </div>
              )}
              {cameraError && (
                <div className="px-4 pb-3 flex items-start justify-between gap-2">
                  <p className="text-red-400 text-sm">📷 {cameraError}</p>
                  <button onClick={() => startCamera(0)}
                    className="text-xs bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded whitespace-nowrap">
                    🔄 Retry
                  </button>
                </div>
              )}
            </Card>
          </div>

          {/* Sidebar — stats + share code */}
          <div className="space-y-4">
            <Card className={isLive ? 'border-l-4 border-red-500' : ''}>
              <h3 className="text-base font-bold text-white mb-3">📡 Stats</h3>
              <div className="space-y-2 text-sm">
                {[
                  ['Status', isLive ? <span className="text-red-400 font-bold">🔴 LIVE</span> : <span className="text-gray-500">⚫ Offline</span>],
                  ['WebSocket', <span className={isConnected ? 'text-accent font-bold' : 'text-red-400 font-bold'}>{isConnected ? '🟢 OK' : '🔴 Off'}</span>],
                  ['Video Peer', <span className={myPeerId ? 'text-accent' : 'text-gray-500'}>{myPeerId ? '🟢 Ready' : '⏳...'}</span>],
                  ['Students', <span className="text-2xl font-bold text-accent">{studentCount}</span>],
                  ['Video Feeds', <span className="text-2xl font-bold text-primary">{studentEntries.length}</span>],
                  ['Captions', <span className="font-bold text-accent">{captions.length}</span>],
                ].map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-gray-400">{label}</span>
                    {value}
                  </div>
                ))}
              </div>
            </Card>

            <Card className="bg-primary/10 border-l-4 border-primary">
              <h3 className="text-base font-bold text-primary mb-2">🔗 Share Code</h3>
              <div className="bg-bg-dark p-3 rounded-lg border-2 border-primary mb-3">
                <p className="text-center text-2xl font-mono font-bold text-primary tracking-widest">{code}</p>
              </div>
              <Button variant="primary" size="sm" className="w-full"
                onClick={() => { navigator.clipboard.writeText(code); alert('Copied!'); }}>
                📋 Copy Code
              </Button>
            </Card>

            {/* ── Safety Controls ── */}
            <SafetyControls settings={safetySettings} onUpdate={updateSafety} />
          </div>
        </div>

        {/* ── Student cameras — horizontal grid ───────────────────────── */}
        {studentEntries.length > 0 && (
          <Card>
            <h2 className="text-lg font-bold text-white mb-3">👥 Students ({studentEntries.length})</h2>
            <div className={`grid gap-4 ${studentEntries.length === 1 ? 'grid-cols-1 max-w-sm' :
              studentEntries.length === 2 ? 'grid-cols-2' :
                'grid-cols-3'
              }`}>
              {studentEntries.map(([peerId, stream], index) => (
                <StudentVideoTile key={peerId} stream={stream} label={`Student ${index + 1}`} />
              ))}
            </div>
          </Card>
        )}

        {/* ── Captions ────────────────────────────────────────────────── */}
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-bold text-white">📝 Live Captions</h2>
            {captions.length > 0 && (
              <Button variant="secondary" size="sm" onClick={clearCaptions}>🗑 Clear</Button>
            )}
          </div>
          {captions.length === 0 ? (
            <div className="text-center py-6 text-gray-500">
              <p className="text-3xl mb-2">🎙️</p>
              <p>Enable mic to see captions here</p>
            </div>
          ) : (
            <CaptionPanel captions={captions} showEmojis={true} showTranslation={false} maxHeight="max-h-64" />
          )}
        </Card>
      </div>
    </div>
  );
}


// ─── Student video tile component ────────────────────────────────────────────
function StudentVideoTile({ stream, label }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => { });
    }
  }, [stream]);

  return (
    <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video">
      <video ref={videoRef} autoPlay playsInline
        className="w-full h-full object-cover" />
      <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
        {label}
      </div>
    </div>
  );
}

// ─── Safety Controls Component ───────────────────────────────────────────────
function SafetyControls({ settings, onUpdate }) {
  const [localLimit, setLocalLimit] = useState(settings.screen_time_limit_minutes || 0);

  // Sync local state if prop changes remotely (e.g. initial load)
  useEffect(() => {
    setLocalLimit(settings.screen_time_limit_minutes || 0);
  }, [settings.screen_time_limit_minutes]);

  const commitLimit = () => {
    const val = parseInt(localLimit) || 0;
    if (val !== settings.screen_time_limit_minutes) {
      onUpdate({ screen_time_limit_minutes: val });
    }
  };

  return (
    <Card className="border-l-4 border-yellow-500 mt-4">
      <h3 className="text-base font-bold text-yellow-500 mb-3">🛡️ Safety Controls</h3>

      <div className="space-y-4">
        {/* Screen Time Limit */}
        <div>
          <label className="text-xs text-gray-400 uppercase font-semibold block mb-1">
            Screen Time Limit (min)
          </label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min="0"
              max="480"
              value={localLimit}
              onChange={(e) => setLocalLimit(e.target.value)}
              onBlur={commitLimit}
              onKeyDown={(e) => e.key === 'Enter' && commitLimit()}
              className="bg-black/40 border border-gray-700 rounded px-2 py-1 w-20 text-white text-sm focus:border-yellow-500 outline-none"
            />
            <span className="text-xs text-gray-500">
              {localLimit == 0 ? "(Unlimited)" : "minutes"}
            </span>
          </div>
        </div>

        {/* Profanity Filter Toggle */}
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400 uppercase font-semibold">
            Profanity Filter
          </label>
          <button
            onClick={() => onUpdate({ profanity_filter_enabled: !settings.profanity_filter_enabled })}
            className={`px-3 py-1 rounded text-xs font-bold transition-all border ${settings.profanity_filter_enabled
              ? "bg-green-500/20 text-green-400 border-green-500/50"
              : "bg-red-500/20 text-red-400 border-red-500/50"
              }`}
          >
            {settings.profanity_filter_enabled ? "ON" : "OFF"}
          </button>
        </div>
      </div>
    </Card>
  );
}
