"""
Room Service — Live Caption Broadcasting (Feature 10)
Manages classroom rooms: creation, joining, leaving, participant tracking.
Rooms are persisted to disk so server restarts don't break active sessions.
"""
import uuid
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger(__name__)

ROOM_DATA_DIR = Path("data/rooms")
ROOM_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Participant:
    """A single participant in a room."""
    participant_id: str
    name: str
    role: str                   # "teacher" or "student"
    language: str               # Preferred caption language: "en", "hi", "ta", "te"
    websocket: Optional[WebSocket] = None
    joined_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_connected: bool = True

    def to_dict(self) -> dict:
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "role": self.role,
            "language": self.language,
            "joined_at": self.joined_at,
            "is_connected": self.is_connected,
        }

    @staticmethod
    def from_dict(data: dict) -> "Participant":
        return Participant(
            participant_id=data["participant_id"],
            name=data["name"],
            role=data["role"],
            language=data.get("language", "en"),
            joined_at=data.get("joined_at", datetime.now().isoformat()),
            is_connected=False,  # Not connected on load; WS reconnects
        )


class Room:
    """
    A live classroom room.
    - One teacher broadcasts audio → AI generates captions
    - All students receive captions in their preferred language
    """

    def __init__(self, room_code: str, title: str, created_by: str,
                 created_at: str = None, is_active: bool = True):
        self.room_code = room_code
        self.title = title
        self.created_by = created_by
        self.created_at = created_at or datetime.now().isoformat()
        self.is_active = is_active
        self.participants: Dict[str, Participant] = {}
        # WebSocket connections: participant_id → WebSocket (NOT persisted)
        self.connections: Dict[str, WebSocket] = {}
        self.caption_count = 0
        self.teacher_peer_id: Optional[str] = None  # Dynamic PeerID for WebRTC video

    def add_participant(self, name: str, role: str, language: str = "en") -> Participant:
        """Add a new participant to the room."""
        participant_id = str(uuid.uuid4())[:8]
        participant = Participant(
            participant_id=participant_id,
            name=name,
            role=role,
            language=language,
        )
        self.participants[participant_id] = participant
        logger.info(f"Room {self.room_code}: {role} '{name}' joined (lang={language})")
        return participant

    def remove_participant(self, participant_id: str):
        """Remove a participant from the room."""
        if participant_id in self.participants:
            name = self.participants[participant_id].name
            del self.participants[participant_id]
            self.connections.pop(participant_id, None)
            logger.info(f"Room {self.room_code}: '{name}' left")

    def register_websocket(self, participant_id: str, websocket: WebSocket):
        """Register a WebSocket connection for a participant."""
        if participant_id in self.participants:
            self.connections[participant_id] = websocket
            self.participants[participant_id].is_connected = True

    def unregister_websocket(self, participant_id: str):
        """Unregister a WebSocket connection."""
        self.connections.pop(participant_id, None)
        if participant_id in self.participants:
            self.participants[participant_id].is_connected = False

    def get_teacher(self) -> Optional[Participant]:
        """Get the teacher participant."""
        for p in self.participants.values():
            if p.role == "teacher":
                return p
        return None

    def get_students(self) -> List[Participant]:
        """Get all student participants."""
        return [p for p in self.participants.values() if p.role == "student"]

    def get_connected_students(self) -> List[Participant]:
        """Get students with active WebSocket connections."""
        return [
            p for p in self.participants.values()
            if p.role == "student" and p.participant_id in self.connections
        ]

    async def broadcast_caption(self, caption_data: dict):
        """
        Broadcast a caption to all connected students.
        Each student gets the caption in their preferred language.
        """
        self.caption_count += 1
        disconnected = []
        connected_students = self.get_connected_students()
        logger.info(f"[Broadcast] {len(connected_students)} student(s) connected in room {self.room_code}")

        for student in connected_students:
            ws = self.connections.get(student.participant_id)
            if ws is None:
                continue
            try:
                payload = {
                    "type": "caption",
                    "room_code": self.room_code,
                    "caption_number": self.caption_count,
                    "text": caption_data.get("text", ""),
                    "simplified_text": caption_data.get("simplified_text", ""),
                    "keywords": self._get_keywords_for(caption_data, student.language),
                    "tone": caption_data.get("tone", {}),
                    "sound_event": caption_data.get("sound_event"),
                    "timestamp": caption_data.get("timestamp", time.time()),
                    "translation": self._get_translation_for(caption_data, student.language),
                    "language": student.language,
                }
                await ws.send_json(payload)
                logger.info(f"[Broadcast] Caption sent to '{student.name}' (lang={student.language}, "
                            f"translated={'yes' if payload['translation'] else 'no'})")
            except Exception as e:
                logger.warning(f"Failed to send caption to {student.name}: {e}")
                disconnected.append(student.participant_id)

        for pid in disconnected:
            self.unregister_websocket(pid)

    def _get_translation_for(self, caption_data: dict, language: str) -> Optional[dict]:
        """Get the right translation for a student's language preference."""
        if language == "en":
            return None
        translations = caption_data.get("translations", {})
        if language in translations:
            return {"text": translations[language], "target_language": language}
        return None

    def _get_keywords_for(self, caption_data: dict, language: str) -> list:
        """Get translated keywords for a student's language (fallback to English)."""
        if language == "en":
            return caption_data.get("keywords", [])
        kw_translations = caption_data.get("keyword_translations", {})
        # Return translated keywords if available, else fall back to English
        return kw_translations.get(language, caption_data.get("keywords", []))

    async def broadcast_sound_event(self, sound_event: dict):
        """
        Broadcast a standalone non-speech sound event to all connected students.
        Called when clapping/laughter is detected with no accompanying speech.
        """
        disconnected = []
        for student in self.get_connected_students():
            ws = self.connections.get(student.participant_id)
            if ws is None:
                continue
            try:
                await ws.send_json({
                    "type":    "sound_event",
                    "event":   sound_event.get("event", "SOUND"),
                    "emoji":   sound_event.get("emoji", "🔊"),
                    "display": sound_event.get("display", "🔊 SOUND"),
                    "confidence": sound_event.get("confidence", 1.0),
                })
            except Exception as e:
                logger.warning(f"Failed to send sound event to {student.name}: {e}")
                disconnected.append(student.participant_id)

        for pid in disconnected:
            self.unregister_websocket(pid)



    def to_dict(self) -> dict:
        return {
            "room_code": self.room_code,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "participant_count": len(self.participants),
            "connected_count": len(self.connections),
            "caption_count": self.caption_count,
            "teacher_peer_id": self.teacher_peer_id,
            "participants": [p.to_dict() for p in self.participants.values()],
        }

    def to_disk_dict(self) -> dict:
        """Serializable form for disk persistence (no WS objects)."""
        return {
            "room_code": self.room_code,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "participants": {
                pid: p.to_dict()
                for pid, p in self.participants.items()
            },
        }

    @staticmethod
    def from_disk_dict(data: dict) -> "Room":
        room = Room(
            room_code=data["room_code"],
            title=data["title"],
            created_by=data["created_by"],
            created_at=data.get("created_at"),
            is_active=data.get("is_active", True),
        )
        for pid, pdata in data.get("participants", {}).items():
            participant = Participant.from_dict(pdata)
            room.participants[pid] = participant
        return room


class RoomService:
    """
    Manages all active classroom rooms.
    Rooms are persisted to disk — server restarts do NOT lose room state.
    """

    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self._load_all_rooms()
        logger.info(f"Room service initialized ({len(self.rooms)} rooms loaded from disk)")

    # ─── Persistence helpers ──────────────────────────────────────────────────

    def _save_to_disk(self, room: Room):
        """Save room JSON to data/rooms/{code}.json"""
        filepath = ROOM_DATA_DIR / f"{room.room_code}.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(room.to_disk_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save room {room.room_code}: {e}")

    def _delete_from_disk(self, room_code: str):
        """Remove room JSON from disk."""
        filepath = ROOM_DATA_DIR / f"{room_code}.json"
        try:
            if filepath.exists():
                filepath.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete room {room_code} from disk: {e}")

    def _load_all_rooms(self):
        """Load all persisted rooms from disk on startup."""
        if not ROOM_DATA_DIR.exists():
            return
        for filepath in ROOM_DATA_DIR.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("is_active", True):
                    room = Room.from_disk_dict(data)
                    self.rooms[room.room_code] = room
                    logger.info(f"Loaded room from disk: {room.room_code} "
                                f"({len(room.participants)} participants)")
            except Exception as e:
                logger.error(f"Failed to load room from {filepath}: {e}")

    # ─── Room management ──────────────────────────────────────────────────────

    def create_room(self, title: str, created_by: str) -> Room:
        """Create a new room with a unique 6-character code."""
        while True:
            code = str(uuid.uuid4()).upper().replace("-", "")[:6]
            if code not in self.rooms:
                break

        room = Room(room_code=code, title=title, created_by=created_by)
        self.rooms[code] = room
        self._save_to_disk(room)
        logger.info(f"Room created: {code} — '{title}' by {created_by}")
        return room

    def get_room(self, room_code: str) -> Optional[Room]:
        """Get a room by its code."""
        return self.rooms.get(room_code.upper())

    def add_participant(self, room_code: str, name: str, role: str, language: str = "en") -> Optional[Participant]:
        """Add a participant and persist the change."""
        room = self.get_room(room_code)
        if not room:
            return None
        participant = room.add_participant(name=name, role=role, language=language)
        self._save_to_disk(room)
        return participant

    def close_room(self, room_code: str):
        """Close a room (teacher stop broadcast)."""
        room = self.rooms.get(room_code.upper())
        if room:
            room.is_active = False
            self._save_to_disk(room)
            logger.info(f"Room {room_code} closed")

    def delete_room(self, room_code: str):
        """Delete a room from memory and disk."""
        self.rooms.pop(room_code.upper(), None)
        self._delete_from_disk(room_code.upper())

    def list_rooms(self) -> List[dict]:
        """List all active rooms."""
        return [r.to_dict() for r in self.rooms.values() if r.is_active]

    def get_all_rooms(self) -> List[dict]:
        """List all rooms including inactive."""
        return [r.to_dict() for r in self.rooms.values()]


# Singleton
_room_service = None

def get_room_service() -> RoomService:
    global _room_service
    if _room_service is None:
        _room_service = RoomService()
    return _room_service
