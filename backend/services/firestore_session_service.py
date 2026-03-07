"""
Firestore-backed Session Service.
Drop-in replacement for session_service.py when Firebase is configured.
Falls back to the local JSON file service when Firebase is NOT configured.
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

SESSIONS_COLLECTION = "sessions"


class FirestoreSessionService:
    """
    Stores sessions in Firestore instead of local JSON files.
    Collection structure: sessions/{session_id} → full session document.
    """

    def __init__(self, db):
        self._db = db
        self._col = db.collection(SESSIONS_COLLECTION)
        logger.info("✅ FirestoreSessionService ready")

    # ── helpers ──────────────────────────────────────────────────────
    def _ref(self, session_id: str):
        return self._col.document(session_id)

    # ── public API (mirrors SessionService) ──────────────────────────
    def create_session(self, session_type="live", title="", language="en") -> dict:
        import uuid
        session_id = str(uuid.uuid4())[:8]
        data = {
            "session_id":   session_id,
            "session_type": session_type,
            "title":        title or f"{session_type.capitalize()} Session",
            "language":     language,
            "created_at":   datetime.now().isoformat(),
            "ended_at":     None,
            "is_active":    True,
            "captions":     [],
            "caption_count": 0,
            "metadata":     {},
        }
        self._ref(session_id).set(data)
        logger.info(f"Firestore session created: {session_id}")
        return data

    def add_caption(self, session_id: str, caption_data: dict) -> Optional[dict]:
        ref = self._ref(session_id)
        snap = ref.get()
        if not snap.exists:
            logger.warning(f"Firestore: session {session_id} not found")
            return None
        doc = snap.to_dict()
        entry = {"timestamp": datetime.now().isoformat(),
                 "index": doc.get("caption_count", 0),
                 **caption_data}
        doc["captions"].append(entry)
        doc["caption_count"] = len(doc["captions"])
        ref.update({"captions": doc["captions"], "caption_count": doc["caption_count"]})
        return entry

    def end_session(self, session_id: str) -> Optional[dict]:
        ref = self._ref(session_id)
        snap = ref.get()
        if not snap.exists:
            return None
        ref.update({"is_active": False, "ended_at": datetime.now().isoformat()})
        return ref.get().to_dict()

    def get_session(self, session_id: str) -> Optional[dict]:
        snap = self._ref(session_id).get()
        return snap.to_dict() if snap.exists else None

    def list_sessions(self) -> List[dict]:
        docs = self._col.order_by("created_at", direction="DESCENDING").stream()
        return [d.to_dict() for d in docs]

    def delete_session(self, session_id: str) -> bool:
        ref = self._ref(session_id)
        if ref.get().exists:
            ref.delete()
            logger.info(f"Firestore session deleted: {session_id}")
            return True
        return False

    def save_video_session(self, title, language, transcription,
                           translation=None, enrichment=None,
                           tone=None, duration=0,
                           video_url=None, filename=None) -> dict:
        sess = self.create_session(session_type="video", title=title, language=language)
        session_id = sess["session_id"]
        caption = {
            "text": transcription.get("text", ""),
            "language": transcription.get("language", language),
            "processing_time": transcription.get("processing_time", 0),
            "enrichment": enrichment,
            "translation": translation,
            "tone": tone,
        }
        metadata = {
            "duration":  duration,
            "vtt":       transcription.get("vtt"),
            "segments":  transcription.get("segments", []),
            "video_url": video_url,
            "filename":  filename,
        }
        self._ref(session_id).update({
            "captions":      [caption],
            "caption_count": 1,
            "metadata":      metadata,
            "is_active":     False,
            "ended_at":      datetime.now().isoformat(),
        })
        return self._ref(session_id).get().to_dict()
