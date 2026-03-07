"""
Non-Speech Sound Detection Service (v4 — speech-filtered, accurate)
Pure numpy. No TensorFlow required.

Detects:  👏 APPLAUSE  😂 LAUGHTER  🔊 BACKGROUND NOISE  🚪 DOOR OPENS

v4 key improvement: Speech-activity guard.
If the audio is speech-like (person talking), we return None immediately.
This prevents random false triggers while the teacher is speaking.
"""
import numpy as np
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

COOLDOWN_SEC   = 5.0     # seconds between same-event repeats
MIN_AUDIO_SECS = 0.8     # ignore buffers shorter than this

SOUND_META = {
    "APPLAUSE":         {"emoji": "👏"},
    "LAUGHTER":         {"emoji": "😂"},
    "BACKGROUND NOISE": {"emoji": "🔊"},
    "DOOR OPENS":       {"emoji": "🚪"},
}


def _features(audio: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Zero-crossing rate per second
    signs = np.where(audio >= 0, 1, -1)
    zcr = float(np.sum(np.abs(np.diff(signs))) / 2 / (len(audio) / sr))

    # Short-time energy coefficient of variation (25 ms frames)
    fsz = max(int(0.025 * sr), 1)
    n   = len(audio) // fsz
    if n >= 2:
        frms  = audio[:n * fsz].reshape(n, fsz)
        fe    = np.sqrt(np.mean(frms ** 2, axis=1))
        e_cv  = float(np.std(fe) / (np.mean(fe) + 1e-8))
    else:
        e_cv = 0.0

    # Spectral bands
    mag   = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = float(np.sum(mag)) + 1e-8
    low_r  = float(np.sum(mag[freqs <  300])                    / total)
    mid_r  = float(np.sum(mag[(freqs >= 300) & (freqs < 3500)]) / total)
    high_r = float(np.sum(mag[freqs >= 3500])                   / total)

    return dict(rms=rms, zcr=zcr, e_cv=e_cv,
                low_r=low_r, mid_r=mid_r, high_r=high_r)


def _is_speech(f: dict) -> bool:
    """
    True if the audio looks like someone talking.
    Speech characteristics:
      • ZCR 80–900 /s  (consistent with vocal cord vibration)
      • mid-band dominant  (voice frequencies: 300–3500 Hz)
      • moderate energy variance — rhythmic but not extreme
      • NOT dominated by high frequencies (≠ clapping)
    If all four match, assume speech and skip sound detection.
    """
    return (
        80  < f["zcr"]  < 900   and   # vocal ZCR window
        f["mid_r"]       > 0.40  and   # voice-band dominant
        0.08 < f["e_cv"] < 0.80  and   # regular speech rhythm
        f["high_r"]      < 0.40        # not clap-broadband
    )


class SoundDetector:

    def __init__(self):
        self._last: dict = {}
        logger.info("SoundDetector v4 (speech-filtered) initialised.")

    def detect_sound(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[dict]:
        if len(audio) / sample_rate < MIN_AUDIO_SECS:
            return None
        if np.max(np.abs(audio)) < 0.003:
            return None

        f = _features(audio, sample_rate)

        # ── Speech guard ────────────────────────────────────────────────────
        # If this sounds like a person talking, don't report a sound event.
        if _is_speech(f):
            logger.debug("[SoundDetect] skipped — sounds like speech "
                         f"(zcr={f['zcr']:.0f} mid_r={f['mid_r']:.2f} e_cv={f['e_cv']:.2f})")
            return None

        logger.debug(f"[SoundDetect] rms={f['rms']:.4f} zcr={f['zcr']:.0f} "
                     f"e_cv={f['e_cv']:.2f} high_r={f['high_r']:.2f} "
                     f"mid_r={f['mid_r']:.2f} low_r={f['low_r']:.2f}")

        scores: dict = {}

        # ── 👏 APPLAUSE ─────────────────────────────────────────────────────
        # Very high ZCR (clap transients), broadband (high_r significant),
        # must NOT be pure speech (already filtered above)
        if (f["zcr"]   >= 500 and
            f["high_r"] >= 0.12 and
            f["high_r"] >= f["mid_r"] * 0.5):
            score = (min(f["zcr"] / 2500, 1.0) * 0.55 +
                     min(f["high_r"] / 0.35, 1.0) * 0.30 +
                     min(f["e_cv"] / 1.0, 1.0) * 0.15)
            scores["APPLAUSE"] = score

        # ── 😂 LAUGHTER ─────────────────────────────────────────────────────
        # Rhythmic bursts (high e_cv), lower ZCR than clapping,
        # voice-frequency band, but more bursty than plain speech
        if (f["e_cv"]  >= 0.80 and          # very bursty = ha-ha pattern
            f["zcr"]   <  500 and           # not clapping speeds
            f["mid_r"] >= 0.35 and
            f["mid_r"] >  f["high_r"]):     # voice > crisp
            score = (min(f["e_cv"] / 2.0, 1.0) * 0.60 +
                     min(f["mid_r"] / 0.6, 1.0)  * 0.40)
            scores["LAUGHTER"] = score

        # ── 🔊 BACKGROUND NOISE ─────────────────────────────────────────────
        # Sustained, VERY low variance, not likely speech (speech guard passed)
        if (f["e_cv"] <= 0.20 and
            f["rms"]  >= 0.004):
            score = ((1.0 - f["e_cv"] / 0.20) * 0.60 +
                     min(f["rms"] / 0.05, 1.0)   * 0.40)
            scores["BACKGROUND NOISE"] = score

        # ── 🚪 DOOR OPENS ───────────────────────────────────────────────────
        # Very sharp spike (extreme e_cv), low-frequency cue (thud)
        if (f["e_cv"]  >= 1.20 and          # very high transient
            f["low_r"] >= 0.20 and
            f["rms"]   >= 0.006):
            score = (min(f["e_cv"] / 3.0, 1.0)  * 0.55 +
                     min(f["low_r"] / 0.50, 1.0) * 0.30 +
                     min(f["rms"] / 0.06, 1.0)   * 0.15)
            scores["DOOR OPENS"] = score

        if not scores:
            return None

        event_name = max(scores, key=scores.__getitem__)
        confidence = scores[event_name]

        if confidence < 0.35:
            return None

        now = time.time()
        if now - self._last.get(event_name, 0) < COOLDOWN_SEC:
            return None
        self._last[event_name] = now

        meta    = SOUND_META[event_name]
        display = f"{meta['emoji']} {event_name}"
        logger.info(f"[SoundDetector] *** {display} *** "
                    f"conf={confidence:.2f} zcr={f['zcr']:.0f} "
                    f"e_cv={f['e_cv']:.2f} high_r={f['high_r']:.2f}")
        return {"event": event_name, "emoji": meta["emoji"],
                "confidence": round(confidence, 2), "display": display}


_detector: Optional[SoundDetector] = None


def get_sound_detector() -> SoundDetector:
    global _detector
    if _detector is None:
        _detector = SoundDetector()
    return _detector
