"""
Safety Routes — Screen Time & Content Filter Settings
Provides REST endpoints for a teacher to configure room-level safety settings.
No auth required — settings are scoped to room_code (only teacher knows it).
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from services.screen_time_service import load_safety_settings, save_safety_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["safety"])


class SafetySettingsRequest(BaseModel):
    screen_time_limit_minutes: int = Field(
        default=0, ge=0, le=480,
        description="Session length limit in minutes. 0 = unlimited."
    )
    allowed_start_time: Optional[str] = Field(
        default=None,
        description="HH:MM — earliest allowed join time (24h). null = no restriction."
    )
    allowed_end_time: Optional[str] = Field(
        default=None,
        description="HH:MM — latest allowed join time (24h). null = no restriction."
    )
    profanity_filter_enabled: bool = Field(
        default=True,
        description="If true, captions pass through the content moderation filter."
    )
    profanity_action: str = Field(
        default="redact",
        description="'redact' replaces bad words with ****. 'block' suppresses the whole caption."
    )


@router.get("/api/v1/rooms/{room_code}/safety")
async def get_safety_settings(room_code: str):
    """
    Get the current safety settings for a room.
    Returns defaults if no settings have been saved yet.
    """
    settings = load_safety_settings(room_code.upper())
    return {"room_code": room_code.upper(), **settings}


@router.put("/api/v1/rooms/{room_code}/safety")
async def update_safety_settings(room_code: str, req: SafetySettingsRequest):
    """
    Create or update safety settings for a room.
    Call this from the teacher dashboard before starting a session.
    """
    code = room_code.upper()
    if req.profanity_action not in ("redact", "block"):
        raise HTTPException(status_code=400, detail="profanity_action must be 'redact' or 'block'")

    settings = req.model_dump()
    save_safety_settings(code, settings)
    logger.info(f"[Safety] Room {code} settings updated: limit={req.screen_time_limit_minutes}min "
                f"filter={req.profanity_filter_enabled}/{req.profanity_action}")
    return {"room_code": code, "status": "updated", **settings}


@router.delete("/api/v1/rooms/{room_code}/safety")
async def reset_safety_settings(room_code: str):
    """Reset safety settings to defaults (removes the settings file)."""
    from pathlib import Path
    path = Path("data/rooms") / f"{room_code.upper()}_safety.json"
    if path.exists():
        path.unlink()
    return {"room_code": room_code.upper(), "status": "reset to defaults"}
