"""
Session Service — Caption Storage & History (Feature 9)
Stores live and video session transcripts for replay, revision, and export.
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Storage directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions")


class Session:
    """Represents a single captioning session (live or video upload)."""
    
    def __init__(self, session_type: str = "live", title: str = "", language: str = "en"):
        self.session_id = str(uuid.uuid4())[:8]
        self.session_type = session_type  # "live" or "video"
        self.title = title or f"{session_type.capitalize()} Session"
        self.language = language
        self.created_at = datetime.now().isoformat()
        self.ended_at = None
        self.is_active = True
        self.captions: List[Dict] = []  # List of caption entries
        self.metadata: Dict = {}  # Extra info (duration, file name, etc.)
    
    def add_caption(self, caption_data: dict):
        """Add a caption entry with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "index": len(self.captions),
            **caption_data
        }
        self.captions.append(entry)
        return entry
    
    def end_session(self):
        """Mark session as ended."""
        self.ended_at = datetime.now().isoformat()
        self.is_active = False
    
    def to_dict(self) -> dict:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "title": self.title,
            "language": self.language,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "is_active": self.is_active,
            "caption_count": len(self.captions),
            "captions": self.captions,
            "metadata": self.metadata
        }
    
    def summary(self) -> dict:
        """Return a lightweight summary (no captions, for listing)."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "title": self.title,
            "language": self.language,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "is_active": self.is_active,
            "caption_count": len(self.captions),
        }


class SessionService:
    """
    Manages session lifecycle: create, store captions, save, retrieve, delete.
    Sessions are stored as JSON files in data/sessions/ directory.
    """
    
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.active_sessions: Dict[str, Session] = {}  # In-memory active sessions
        logger.info(f"Session service initialized. Storage: {DATA_DIR}")
    
    def create_session(self, session_type: str = "live", title: str = "", language: str = "en") -> Session:
        """Create and register a new session."""
        session = Session(session_type=session_type, title=title, language=language)
        self.active_sessions[session.session_id] = session
        logger.info(f"Session created: {session.session_id} ({session_type})")
        return session
    
    def add_caption(self, session_id: str, caption_data: dict) -> Optional[dict]:
        """Add a caption to an active session."""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found (may be ended)")
            return None
        return session.add_caption(caption_data)
    
    def end_session(self, session_id: str) -> Optional[dict]:
        """End a session, save to disk, and remove from active memory."""
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session.end_session()
        
        # Save to disk
        self._save_to_disk(session)
        
        # Remove from active memory
        del self.active_sessions[session_id]
        logger.info(f"Session ended and saved: {session_id} ({len(session.captions)} captions)")
        return session.to_dict()
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data (checks active memory first, then disk)."""
        # Check active sessions
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].to_dict()
        
        # Check disk
        return self._load_from_disk(session_id)
    
    def list_sessions(self) -> List[dict]:
        """List all sessions (active + saved)."""
        sessions = []
        
        # Active sessions
        for session in self.active_sessions.values():
            sessions.append(session.summary())
        
        # Saved sessions from disk
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith(".json"):
                    sid = filename.replace(".json", "")
                    if sid not in self.active_sessions:
                        data = self._load_from_disk(sid)
                        if data:
                            meta = data.get("metadata", {})
                            sessions.append({
                                "session_id": data["session_id"],
                                "session_type": data["session_type"],
                                "title": data["title"],
                                "language": data["language"],
                                "created_at": data["created_at"],
                                "ended_at": data["ended_at"],
                                "is_active": data["is_active"],
                                "caption_count": data["caption_count"],
                                # Include video playback fields for My Videos page
                                "video_url": meta.get("video_url"),
                                "filename": meta.get("filename"),
                                "duration": meta.get("duration"),
                            })
        
        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk."""
        filepath = os.path.join(DATA_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Session deleted: {session_id}")
            return True
        
        # Also remove from active if exists
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        
        return False
    
    def save_video_session(self, title: str, language: str, transcription: dict, 
                            translation: dict = None, enrichment: dict = None,
                            tone: dict = None, duration: float = 0,
                            video_url: str = None, filename: str = None) -> dict:
        """
        Convenience method: Create + save a complete video upload session in one call.
        Used by video_routes.py after processing a video.
        """
        session = self.create_session(session_type="video", title=title, language=language)
        
        # Add the full transcription as a single caption entry
        session.add_caption({
            "text": transcription.get("text", ""),
            "language": transcription.get("language", language),
            "processing_time": transcription.get("processing_time", 0),
            "enrichment": enrichment,
            "translation": translation,
            "tone": tone,
        })
        
        session.metadata = {
            "duration": duration,
            "vtt": transcription.get("vtt"),
            "segments": transcription.get("segments", []),
            "video_url": video_url,
            "filename": filename,
        }
        
        # End and save immediately
        return self.end_session(session.session_id)
    
    def _save_to_disk(self, session: Session):
        """Write session JSON to data/sessions/."""
        filepath = os.path.join(DATA_DIR, f"{session.session_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Session saved to: {filepath}")
    
    def _load_from_disk(self, session_id: str) -> Optional[dict]:
        """Load session JSON from disk."""
        filepath = os.path.join(DATA_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None


# Singleton
_session_service = None

def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
