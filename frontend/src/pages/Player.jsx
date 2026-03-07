import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

/** Parse a VTT file into [{start, end, text}] segments */
function parseVTT(vttText) {
  const cues = [];
  const blocks = vttText.replace(/\r\n/g, '\n').split(/\n\n+/);
  const timeRe = /(\d+:\d+:\d+[\.,]\d+)\s+-->\s+(\d+:\d+:\d+[\.,]\d+)/;

  function toSeconds(ts) {
    const [h, m, rest] = ts.replace(',', '.').split(':');
    return parseFloat(h) * 3600 + parseFloat(m) * 60 + parseFloat(rest);
  }

  for (const block of blocks) {
    const lines = block.trim().split('\n');
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(timeRe);
      if (m) {
        // Strip any HTML tags Whisper may add (e.g. <c>word</c>)
        const text = lines.slice(i + 1).join(' ').replace(/<[^>]+>/g, '').trim();
        if (text) {
          cues.push({ start: toSeconds(m[1]), end: toSeconds(m[2]), text });
        }
        break;
      }
    }
  }
  return cues;
}

/**
 * Split each VTT cue into word-level sub-cues so captions pop in
 * one small chunk at a time — identical to YouTube's live caption pacing.
 * wordsPerChunk = 3 gives a fast, natural feel.
 */
function splitIntoSubCues(cues, wordsPerChunk = 3) {
  const result = [];
  for (const cue of cues) {
    const words = cue.text.split(/\s+/).filter(Boolean);
    if (!words.length) continue;
    const numChunks = Math.max(1, Math.ceil(words.length / wordsPerChunk));
    const span = (cue.end - cue.start) / numChunks;
    for (let i = 0; i < numChunks; i++) {
      result.push({
        start: cue.start + i * span,
        end: cue.start + (i + 1) * span,
        text: words.slice(i * wordsPerChunk, (i + 1) * wordsPerChunk).join(' '),
      });
    }
  }
  return result;
}

export function Player() {
  const location = useLocation();
  const navigate = useNavigate();
  const [videoData, setVideoData] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);
  const [captionsVisible, setCaptionsVisible] = useState(true);

  // VTT cues for in-video captions
  const [cues, setCues] = useState([]);
  const [cuesLoading, setCuesLoading] = useState(false);
  const [activeCue, setActiveCue] = useState(null);
  // Track the last cue text so we can keep it visible briefly after it ends
  const [displayedCue, setDisplayedCue] = useState(null);
  const hideTimerRef = useRef(null);

  const videoRef = useRef(null);

  useEffect(() => {
    if (location.state?.videoData) {
      setVideoData(location.state.videoData);
    }
  }, [location.state]);

  // Resolve video URL — checks all possible locations in priority order
  const resolvedVideoUrl = videoData?.video_url
    || videoData?.metadata?.video_url
    || (videoData?.metadata?.filename
      ? `/api/v1/video/stream/${encodeURIComponent(videoData.metadata.filename.replace(/ /g, '_'))}`
      : null)
    || (videoData?.filename
      ? `/api/v1/video/stream/${encodeURIComponent(videoData.filename.replace(/ /g, '_'))}`
      : null)
    // Last resort: derive from title (title = original filename for video sessions)
    || (videoData?.title && (videoData?.session_type || videoData?.type) === 'video'
      ? `/api/v1/video/stream/${encodeURIComponent(videoData.title.replace(/ /g, '_'))}`
      : null);

  const resolvedFilename = videoData?.filename
    || videoData?.metadata?.filename
    || videoData?.title
    || 'Video';

  // ─── Load VTT: use embedded VTT in response first, fallback to session endpoint ──
  useEffect(() => {
    if (!videoData) return;

    function processVTT(text) {
      console.log('[Player] VTT text length:', text.length);
      console.log('[Player] VTT preview:', text.substring(0, 200));
      const rawCues = parseVTT(text);
      const subCues = splitIntoSubCues(rawCues, 3);
      console.log('[Player] raw cues:', rawCues.length, '→ sub-cues:', subCues.length);
      if (subCues.length > 0) console.log('[Player] First sub-cue:', subCues[0]);
      setCues(subCues);
      setCuesLoading(false);
    }

    // Primary: VTT is already embedded in the upload API response
    const embeddedVTT = videoData?.transcription?.vtt;
    if (embeddedVTT && embeddedVTT.startsWith('WEBVTT')) {
      console.log('[Player] Using embedded VTT from upload response');
      processVTT(embeddedVTT);
      return;
    }

    // Fallback: fetch from session export endpoint (used for history playback)
    if (!videoData?.session_id) {
      console.warn('[Player] No VTT source available — no embedded VTT and no session_id');
      setCuesLoading(false);
      return;
    }

    setCuesLoading(true);
    const vttUrl = `${API_BASE}/api/v1/export/${videoData.session_id}/vtt`;
    console.log('[Player] Fetching VTT from:', vttUrl);
    fetch(vttUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then(processVTT)
      .catch((err) => {
        console.warn('[Player] Could not load VTT:', err);
        setCuesLoading(false);
      });
  }, [videoData]);

  // ─── Pick active cue based on current time ────────────────────────────────
  // Shows ONLY the cue that matches the current playback time.
  // Nothing is shown before the first word is spoken or after the last one ends.
  useEffect(() => {
    if (!cues.length) {
      // No cues loaded — show nothing (do NOT fall back to full transcript)
      setActiveCue(null);
      return;
    }
    const found = cues.find((c) => currentTime >= c.start && currentTime < c.end);
    setActiveCue(found || null);
  }, [currentTime, cues]);

  // ─── Keep caption visible for 400ms after it ends (avoid flicker) ─────────
  useEffect(() => {
    if (activeCue) {
      // New cue came in — cancel any pending hide and show immediately
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
      setDisplayedCue(activeCue);
    } else {
      // Cue ended — linger briefly so it doesn't flash off abruptly
      hideTimerRef.current = setTimeout(() => setDisplayedCue(null), 400);
    }
    return () => {
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    };
  }, [activeCue]);

  const formatTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const handleDownload = async (format) => {
    if (!videoData?.session_id) return;
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/export/${videoData.session_id}/${format}${format === 'vtt' ? '?download=true' : ''}`
      );
      if (!response.ok) throw new Error(`Export failed: ${response.status}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${videoData.filename?.replace(/\.[^/.]+$/, '') || 'captions'}.${format === 'summary' ? 'txt' : format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Download failed: ${err.message}`);
    }
  };

  if (!videoData) {
    return (
      <div className="min-h-screen bg-bg-dark py-12">
        <div className="max-w-4xl mx-auto px-4">
          <Card>
            <div className="text-center py-16">
              <div className="text-5xl mb-4">🎬</div>
              <p className="text-gray-400 text-lg mb-4">No video loaded</p>
              <Button variant="primary" onClick={() => navigate('/upload')}>Upload Video</Button>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-dark py-8">
      <div className="max-w-5xl mx-auto px-4 space-y-6">

        {/* ── Cinema Video Player ─────────────────────────────────────────── */}
        <div
          className="relative bg-black rounded-2xl overflow-hidden shadow-2xl"
          style={{ aspectRatio: '16/9' }}
        >
          {resolvedVideoUrl ? (
            <video
              ref={videoRef}
              src={`${API_BASE}${resolvedVideoUrl}`}
              className="w-full h-full object-contain"
              onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
              onDurationChange={(e) => setDuration(e.target.duration)}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              crossOrigin="anonymous"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <div className="text-6xl mb-4">⏳</div>
                <p>Video processing...</p>
              </div>
            </div>
          )}

          {/* ── Live Caption Overlay (YouTube-style) ──────────────────────── */}
          {captionsVisible && (
            <div
              className="absolute bottom-16 left-0 right-0 flex justify-center px-8 pointer-events-none"
              style={{ zIndex: 20, minHeight: '3.5rem' }}
            >
              {displayedCue ? (
                <div
                  key={displayedCue.text}
                  style={{
                    background: 'rgba(0,0,0,0.80)',
                    backdropFilter: 'blur(4px)',
                    borderRadius: '6px',
                    padding: '8px 18px',
                    maxWidth: '75%',
                    animation: 'captionPop 0.15s ease-out',
                  }}
                >
                  <p
                    style={{
                      color: '#fff',
                      fontSize: '1.3rem',
                      fontWeight: 700,
                      lineHeight: 1.4,
                      textAlign: 'center',
                      textShadow: '0 1px 6px rgba(0,0,0,0.95)',
                      fontFamily: "'Inter', 'Segoe UI', sans-serif",
                      letterSpacing: '0.01em',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {displayedCue.text}
                  </p>
                </div>
              ) : cuesLoading ? (
                <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.8rem' }}>
                  Loading captions…
                </span>
              ) : null}
            </div>
          )}

          {/* ── Gradient bar at bottom for controls ───────────────────── */}
          <div
            className="absolute bottom-0 left-0 right-0 px-4 pb-3 pt-8"
            style={{
              background: 'linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 100%)',
              zIndex: 10,
            }}
          >
            {/* Seek bar */}
            <div className="mb-2">
              <input
                type="range"
                min="0"
                max={duration || 100}
                step="0.1"
                value={currentTime}
                onChange={(e) => {
                  const t = parseFloat(e.target.value);
                  setCurrentTime(t);
                  if (videoRef.current) videoRef.current.currentTime = t;
                }}
                className="w-full h-1 rounded cursor-pointer"
                style={{ accentColor: '#8B5CF6' }}
              />
            </div>

            {/* Controls row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {/* Play/Pause */}
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      if (isPlaying) videoRef.current.pause();
                      else videoRef.current.play();
                    }
                  }}
                  className="text-white text-2xl hover:text-primary transition-colors"
                >
                  {isPlaying ? '⏸' : '▶️'}
                </button>
                {/* Restart */}
                <button
                  onClick={() => {
                    if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); }
                  }}
                  className="text-white text-lg hover:text-primary transition-colors"
                >
                  ⏮
                </button>
                {/* Time */}
                <span className="text-white text-xs font-mono">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
              </div>

              <div className="flex items-center gap-3">
                {/* CC toggle */}
                <button
                  onClick={() => setCaptionsVisible(!captionsVisible)}
                  className={`text-xs font-bold px-2 py-1 rounded border transition-all ${captionsVisible
                    ? 'bg-white text-black border-white'
                    : 'bg-transparent text-gray-400 border-gray-600'
                    }`}
                  title="Toggle captions"
                >
                  CC
                </button>
                {/* Translate toggle */}
                <button
                  onClick={() => setShowTranslation(!showTranslation)}
                  className="text-white text-sm hover:text-primary transition-colors"
                  title="Toggle translation"
                >
                  🌐
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ── CSS animation ─────────────────────────────────────────────────── */}
        <style>{`
          @keyframes captionPop {
            from { opacity: 0; transform: translateY(4px) scale(0.97); }
            to   { opacity: 1; transform: translateY(0)   scale(1);    }
          }
        `}</style>

        {/* ── Below video: info + download in 2 cols ────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Translation panel (if active) */}
          {showTranslation && videoData.translation?.text && (
            <Card className="md:col-span-2">
              <h3 className="text-base font-bold text-primary mb-2">🌐 Translation</h3>
              <p className="text-gray-300 leading-relaxed text-sm">
                {videoData.translation.text}
              </p>
            </Card>
          )}

          {/* Video info */}
          <Card>
            <h3 className="text-lg font-bold text-white mb-4">📄 Video Info</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Filename</span>
                <span className="text-white font-semibold truncate max-w-[60%]">{resolvedFilename}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Duration</span>
                <span className="text-white font-semibold">
                  {duration ? formatTime(duration) : videoData.duration ? formatTime(videoData.duration) : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">File Size</span>
                <span className="text-white font-semibold">
                  {videoData.file_size ? `${(videoData.file_size / (1024 * 1024)).toFixed(1)} MB` : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Caption Cues</span>
                <span className="text-accent font-semibold">
                  {cuesLoading ? '⏳ loading…' : cues.length > 0 ? `${cues.length} cues` : 'none'}
                </span>
              </div>
            </div>
          </Card>

          {/* Downloads */}
          <Card>
            <h3 className="text-lg font-bold text-white mb-4">⬇️ Download</h3>
            <div className="grid grid-cols-2 gap-2">
              {['srt', 'vtt', 'txt', 'summary'].map((fmt) => (
                <Button key={fmt} variant="secondary" size="sm" onClick={() => handleDownload(fmt)}>
                  📥 {fmt.toUpperCase()}
                </Button>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
