"""
Session Routes — Caption Storage & History API (Feature 9)
REST endpoints for managing captioning sessions.
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services.session_service import get_session_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


# --- Request Models ---

class CreateSessionRequest(BaseModel):
    session_type: str = "live"  # "live" or "video"
    title: str = ""
    language: str = "en"

class AddCaptionRequest(BaseModel):
    text: str
    language: str = "en"
    simplified_text: Optional[str] = None
    enriched_text: Optional[str] = None
    translated_text: Optional[str] = None
    target_language: Optional[str] = None
    keywords: Optional[list] = None
    tone: Optional[dict] = None
    sound_event: Optional[dict] = None


# --- Endpoints ---

@router.post("/create")
async def create_session(req: CreateSessionRequest):
    """Create a new captioning session."""
    service = get_session_service()
    session = service.create_session(
        session_type=req.session_type,
        title=req.title,
        language=req.language
    )
    return {
        "status": "created",
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at
    }


@router.post("/{session_id}/caption")
async def add_caption(session_id: str, req: AddCaptionRequest):
    """Add a caption entry to an active session."""
    service = get_session_service()
    entry = service.add_caption(session_id, req.dict())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or already ended")
    return {"status": "added", "index": entry["index"]}


@router.post("/{session_id}/end")
async def end_session(session_id: str):
    """End an active session and save to disk."""
    service = get_session_service()
    result = service.end_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {
        "status": "ended",
        "session_id": result["session_id"],
        "caption_count": result["caption_count"],
        "ended_at": result["ended_at"]
    }


@router.get("/")
async def list_sessions():
    """List all sessions (active + saved)."""
    service = get_session_service()
    sessions = service.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get full session data including all captions."""
    service = get_session_service()
    session = service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a saved session."""
    service = get_session_service()
    deleted = service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"status": "deleted", "session_id": session_id}
