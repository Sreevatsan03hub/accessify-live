"""
Export Routes — Post-Processed Downloads API (Feature 10)
Download captions from saved sessions as SRT, VTT, TXT, or Summary.
"""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from services.session_service import get_session_service
from services.export_service import get_export_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/export", tags=["export"])


def _get_session_data(session_id: str) -> dict:
    """Helper: Get session data or raise 404."""
    data = get_session_service().get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return data


@router.get("/{session_id}/srt")
async def download_srt(session_id: str):
    """Download captions as SRT subtitle file."""
    session_data = _get_session_data(session_id)
    export = get_export_service()
    content = export.generate_srt(session_data)
    
    if not content:
        raise HTTPException(status_code=400, detail="No caption data available for export")
    
    filename = f"{session_data.get('title', 'captions').replace(' ', '_')}.srt"
    return PlainTextResponse(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/{session_id}/vtt")
async def download_vtt(session_id: str, download: bool = False):
    """
    Download captions as WebVTT subtitle file.
    Args:
        download: If True, forces file download (Content-Disposition: attachment).
                  If False, returns inline for <track> usage.
    """
    session_data = _get_session_data(session_id)
    export = get_export_service()
    content = export.generate_vtt(session_data)
    
    filename = f"{session_data.get('title', 'captions').replace(' ', '_')}.vtt"
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    
    return PlainTextResponse(
        content=content,
        media_type="text/vtt",
        headers=headers
    )


@router.get("/{session_id}/txt")
async def download_txt(session_id: str):
    """Download full transcript as plain text (includes enrichment data)."""
    session_data = _get_session_data(session_id)
    export = get_export_service()
    content = export.generate_txt(session_data)
    
    filename = f"{session_data.get('title', 'captions').replace(' ', '_')}.txt"
    return PlainTextResponse(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/{session_id}/summary")
async def download_summary(session_id: str):
    """Download a concise session summary for note-making."""
    session_data = _get_session_data(session_id)
    export = get_export_service()
    content = export.generate_summary(session_data)
    
    filename = f"{session_data.get('title', 'captions').replace(' ', '_')}_summary.txt"
    return PlainTextResponse(
        content=content,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/{session_id}/all")
async def get_all_formats(session_id: str):
    """Preview all export formats for a session (JSON response, not download)."""
    session_data = _get_session_data(session_id)
    export = get_export_service()
    
    return {
        "session_id": session_id,
        "title": session_data.get("title", ""),
        "formats": {
            "srt": export.generate_srt(session_data)[:500] + "..." if len(export.generate_srt(session_data)) > 500 else export.generate_srt(session_data),
            "vtt": export.generate_vtt(session_data)[:500] + "..." if len(export.generate_vtt(session_data)) > 500 else export.generate_vtt(session_data),
            "txt": export.generate_txt(session_data)[:500] + "..." if len(export.generate_txt(session_data)) > 500 else export.generate_txt(session_data),
            "summary": export.generate_summary(session_data)[:500] + "..." if len(export.generate_summary(session_data)) > 500 else export.generate_summary(session_data),
        },
        "download_links": {
            "srt": f"/api/v1/export/{session_id}/srt",
            "vtt": f"/api/v1/export/{session_id}/vtt",
            "txt": f"/api/v1/export/{session_id}/txt",
            "summary": f"/api/v1/export/{session_id}/summary",
        }
    }
