"""
Screen Time Control Service — Parental Control Feature
Tracks session durations per participant and enforces time limits.
Works alongside the existing room/participant system (no auth DB needed).
"""
import json
import logging
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SAFETY_DIR = Path("data/rooms")
SAFETY_DIR.mkdir(parents=True, exist_ok=True)

# ─── In-memory session tracker ────────────────────────────────────────────────
# { participant_id: session_start_epoch_float }
_active_sessions: dict[str, float] = {}


# ─── Safety settings helpers ──────────────────────────────────────────────────

def _settings_path(room_code: str) -> Path:
    return SAFETY_DIR / f"{room_code}_safety.json"


def load_safety_settings(room_code: str) -> dict:
    """Load room safety settings from disk. Returns defaults if not found."""
    path = _settings_path(room_code)
    defaults = {
        "screen_time_limit_minutes": 0,   # 0 = unlimited
        "allowed_start_time": None,        # "HH:MM" or null
        "allowed_end_time": None,
        "profanity_filter_enabled": True,
        "profanity_action": "redact",      # "redact" | "block"
    }
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**defaults, **data}
        except Exception as e:
            logger.warning(f"Failed to load safety settings for {room_code}: {e}")
    return defaults


def save_safety_settings(room_code: str, settings: dict) -> None:
    """Save room safety settings to disk."""
    path = _settings_path(room_code)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save safety settings for {room_code}: {e}")


# ─── Session tracking ─────────────────────────────────────────────────────────

def start_session(participant_id: str) -> None:
    """Record session start time for a participant."""
    _active_sessions[participant_id] = time.time()
    logger.debug(f"[ScreenTime] Session started: {participant_id}")


def end_session(participant_id: str) -> Optional[float]:
    """End a session. Returns duration in minutes (or None if not tracked)."""
    start = _active_sessions.pop(participant_id, None)
    if start is None:
        return None
    duration = (time.time() - start) / 60.0
    logger.debug(f"[ScreenTime] Session ended: {participant_id} ({duration:.1f} min)")
    return duration


def get_elapsed_minutes(participant_id: str) -> float:
    """Return how many minutes this participant has been in session."""
    start = _active_sessions.get(participant_id)
    if start is None:
        return 0.0
    return (time.time() - start) / 60.0


# ─── Limit enforcement ────────────────────────────────────────────────────────

class ScreenTimeStatus:
    __slots__ = ("allowed", "remaining_minutes", "reason")

    def __init__(self, allowed: bool, remaining_minutes: float, reason: str = ""):
        self.allowed = allowed
        self.remaining_minutes = remaining_minutes
        self.reason = reason


def check_screen_time(participant_id: str, room_code: str) -> ScreenTimeStatus:
    """
    Check whether this participant is within their allowed screen time.

    Returns:
        ScreenTimeStatus with .allowed (bool) and .remaining_minutes (float).
    """
    settings = load_safety_settings(room_code)
    limit_minutes: int = settings.get("screen_time_limit_minutes", 0)
    start_str: Optional[str] = settings.get("allowed_start_time")
    end_str: Optional[str] = settings.get("allowed_end_time")

    # ── Allowed hours check ───────────────────────────────────────────────────
    if start_str and end_str:
        try:
            now_time = datetime.now().time()
            start_t = dtime.fromisoformat(start_str)
            end_t   = dtime.fromisoformat(end_str)
            in_window = (
                start_t <= now_time <= end_t
                if start_t <= end_t
                else (now_time >= start_t or now_time <= end_t)   # overnight window
            )
            if not in_window:
                return ScreenTimeStatus(
                    allowed=False,
                    remaining_minutes=0,
                    reason=f"Outside allowed hours ({start_str}–{end_str})"
                )
        except Exception as e:
            logger.warning(f"[ScreenTime] Could not parse allowed hours: {e}")

    # ── Screen time limit check ───────────────────────────────────────────────
    if limit_minutes <= 0:
        return ScreenTimeStatus(allowed=True, remaining_minutes=float("inf"))

    elapsed = get_elapsed_minutes(participant_id)
    remaining = limit_minutes - elapsed

    if remaining <= 0:
        return ScreenTimeStatus(
            allowed=False,
            remaining_minutes=0,
            reason=f"Screen time limit of {limit_minutes} min reached"
        )

    return ScreenTimeStatus(allowed=True, remaining_minutes=remaining)
