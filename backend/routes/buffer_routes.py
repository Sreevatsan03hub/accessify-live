"""
Buffer Routes — Low-Internet Resilience API (Feature 11)
Endpoints for client registration, batch processing, and reconnection.
"""
import logging
import base64
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.buffer_service import get_buffer_service
from services.speech_to_text import get_stt_service
from services.text_simplification_service import get_simplifier
from services.keyword_detection_service import get_keyword_detector
from services.tone_analysis_service import get_tone_service
from services.translation_service import get_translator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/buffer", tags=["buffer"])


# --- Request Models ---

class RegisterRequest(BaseModel):
    session_id: Optional[str] = None

class ReconnectRequest(BaseModel):
    client_id: str

class BufferedChunk(BaseModel):
    """A single buffered audio chunk from the client."""
    audio_base64: str          # Base64-encoded audio data
    timestamp: float           # Client-side timestamp when recorded
    sequence: int              # Sequence number for ordering

class BatchProcessRequest(BaseModel):
    """Batch of buffered chunks sent after reconnection."""
    client_id: str
    chunks: List[BufferedChunk]
    translate_to: Optional[str] = None  # "hi", "ta", "te"


# --- Endpoints ---

@router.post("/register")
async def register_client(req: RegisterRequest):
    """Register a new client and get a client_id for reconnection."""
    service = get_buffer_service()
    client_id = service.register_client(session_id=req.session_id)
    return {
        "client_id": client_id,
        "status": "registered",
        "message": "Use this client_id to reconnect if connection drops"
    }


@router.post("/reconnect")
async def reconnect_client(req: ReconnectRequest):
    """Reconnect a previously registered client."""
    service = get_buffer_service()
    client = service.reconnect_client(req.client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return {
        "status": "reconnected",
        "client_id": req.client_id,
        "reconnect_count": client.reconnect_count,
        "pending_chunks": len(client.pending_chunks)
    }


@router.post("/process-batch")
async def process_batch(req: BatchProcessRequest):
    """
    Process a batch of buffered audio chunks.
    Called by client after reconnecting with buffered audio.
    Returns all captions at once for the buffered period.
    """
    service = get_buffer_service()
    
    if not req.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")
    
    # Sort chunks by sequence number
    sorted_chunks = sorted(req.chunks, key=lambda c: c.sequence)
    
    results = []
    stt = get_stt_service()
    
    for chunk in sorted_chunks:
        try:
            # Decode audio
            audio_bytes = base64.b64decode(chunk.audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Process through AI pipeline
            transcription = stt.transcribe_audio(audio_array)
            
            if not transcription or not transcription.text.strip():
                continue
            
            # Simplification
            simplified = get_simplifier().simplify(transcription.text)
            
            # Keywords
            keywords = get_keyword_detector().extract_keywords(transcription.text)
            enriched = get_keyword_detector().enrich_text(transcription.text)
            
            # Tone
            tone = get_tone_service().analyze_tone(transcription.text)
            
            # Translation
            translation_text = None
            if req.translate_to:
                try:
                    translation_text = get_translator().translate(
                        transcription.text, req.translate_to
                    )
                except Exception:
                    pass
            
            results.append({
                "sequence": chunk.sequence,
                "timestamp": chunk.timestamp,
                "text": transcription.text,
                "simplified_text": simplified,
                "enriched_text": enriched,
                "keywords": keywords,
                "tone": tone,
                "translation": {
                    "text": translation_text,
                    "target_language": req.translate_to
                } if translation_text else None,
                "language": transcription.language
            })
            
        except Exception as e:
            logger.warning(f"Failed to process buffered chunk {chunk.sequence}: {e}")
            continue
    
    # Mark chunks as processed
    service.mark_chunks_processed(req.client_id, len(sorted_chunks))
    
    logger.info(f"Batch processed: {len(results)}/{len(sorted_chunks)} chunks for client {req.client_id}")
    
    return {
        "client_id": req.client_id,
        "processed": len(results),
        "total_chunks": len(sorted_chunks),
        "results": results
    }


@router.get("/status/{client_id}")
async def get_buffer_status(client_id: str):
    """Get buffer status for a client."""
    service = get_buffer_service()
    status = service.get_status(client_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@router.get("/clients")
async def list_clients():
    """List all active client buffers."""
    service = get_buffer_service()
    clients = service.get_all_clients()
    return {"clients": clients, "total": len(clients)}


@router.delete("/{client_id}")
async def cleanup_client(client_id: str):
    """Remove a client buffer."""
    service = get_buffer_service()
    service.cleanup_client(client_id)
    return {"status": "cleaned", "client_id": client_id}
