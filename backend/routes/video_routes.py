"""
Video Upload Routes
Handles video file uploads and processing.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from services.video_service import get_video_processor
from services.speech_to_text import get_stt_service
from services.translation_service import get_translator
from services.text_simplification_service import get_simplifier
from services.keyword_detection_service import get_keyword_detector
from services.tone_analysis_service import get_tone_service
from services.session_service import get_session_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Persistent storage for video playback
VIDEO_STORAGE_DIR = Path("uploads/video")
VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    language: str = Form("en"),
    translate_to: Optional[str] = Form(None)
):
    """
    Upload video file and process through STT/Translation pipeline.
    
    Args:
        file: Video file (mp4, mkv, avi, mov, webm)
        language: Source language for transcription (default: en)
        translate_to: Target language for translation (hi, ta, te)
        
    Returns:
        JSON with transcription, translation, and metadata
    """
    video_processor = get_video_processor()
    # Use 'tiny' model for fast processing — still accurate for English
    stt_service = get_stt_service(model_size="tiny")
    
    # Create temp file for uploaded video
    temp_video = tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False)
    temp_video_path = temp_video.name
    
    try:
        # Save uploaded file permanently for playback
        video_filename = file.filename.replace(" ", "_") # Sanitize filename
        saved_video_path = VIDEO_STORAGE_DIR / video_filename
        
        logger.info(f"Receiving video upload: {file.filename}")
        content = await file.read()
        
        # Write to persistent storage
        with open(saved_video_path, "wb") as f:
            f.write(content)
            
        # Write to temp file for processing (or reuse saved file)
        temp_video.write(content)
        temp_video.close()
        
        # Extract audio from video
        logger.info("Extracting audio from video...")
        result = video_processor.extract_audio(temp_video_path)
        
        if not result.success:
            logger.error(f"Extraction failed: {result.error}")
            raise HTTPException(status_code=400, detail=f"Audio extraction failed: {result.error}")
        
        audio_path = result.audio_path
        logger.info(f"Audio extracted: {audio_path} (duration: {result.duration:.2f}s)")
        
        # Transcribe audio
        logger.info("Transcribing audio...")
        transcription = stt_service.transcribe_audio(
            audio_path=audio_path,
            language=language
        )
        
        if translate_to:
            logger.info(f"Translating to: {translate_to}")
            translator = get_translator()

            # Translate full text block (for display below video)
            translation_result = translator.translate(transcription.text, translate_to)
            translated_text = translation_result.translated_text

            # Translate segments for VTT captions
            if transcription.segments:
                logger.info(f"Translating {len(transcription.segments)} segments for VTT...")
                translated_segments = []
                for seg in transcription.segments:
                    new_seg = seg.copy()
                    seg_result = translator.translate(seg['text'].strip(), translate_to)
                    new_seg['text'] = get_keyword_detector().enrich_text(
                        seg_result.translated_text, format="vtt"
                    )
                    translated_segments.append(new_seg)
                vtt_content = stt_service.generate_vtt(translated_segments)
            else:
                vtt_content = None
        else:
            translated_text = None
            # Generate English VTT with enrichment
            if transcription.segments:
                # Create enriched segments for VTT
                enriched_segments = []
                for seg in transcription.segments:
                    new_seg = seg.copy()
                    new_seg['text'] = get_keyword_detector().enrich_text(seg['text'], format="vtt")
                    enriched_segments.append(new_seg)
                    
                vtt_content = stt_service.generate_vtt(enriched_segments)
            else:
                vtt_content = None
        
        # Clean up temp files
        try:
            os.unlink(audio_path)
        except:
            pass
        
        
        # Auto-save session for history (Feature 9)
        session_id = None
        try:
            session = get_session_service().save_video_session(
                title=file.filename,
                language=transcription.language,
                transcription={
                    "text": transcription.text,
                    "language": transcription.language,
                    "processing_time": transcription.processing_time,
                    "vtt": vtt_content
                },
                translation={"text": translated_text, "target_language": translate_to} if translate_to else None,
                enrichment={"keywords": get_keyword_detector().extract_keywords(transcription.text)},
                tone=get_tone_service().analyze_tone(transcription.text),
                duration=result.duration,
                video_url=f"/api/v1/video/stream/{video_filename}",
                filename=file.filename,
            )
            session_id = session.session_id
            logger.info(f"Video session saved to history: {file.filename} (ID: {session_id})")
        except Exception as e:
            logger.warning(f"Failed to save video session: {e}")
        
        # Return results
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "video_url": f"/api/v1/video/stream/{video_filename}",
            "duration": result.duration,
            "transcription": {
                "text": transcription.text,
                "language": transcription.language,
                "processing_time": transcription.processing_time,
                "vtt": vtt_content  # Return VTT (Translated or English)
            },
            "enrichment": {
                "keywords": get_keyword_detector().extract_keywords(transcription.text),
                "text": get_keyword_detector().enrich_text(transcription.text)
            },
            "tone": get_tone_service().analyze_tone(transcription.text),
            "translation": {
                "text": get_keyword_detector().enrich_text(translated_text) if translated_text else None,
                "target_language": translate_to
            } if translate_to else None
        })        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up uploaded video
        try:
            os.unlink(temp_video_path)
        except:
            pass





@router.get("/info")
def get_supported_formats():
    """Get list of supported video formats."""
    return {
        "supported_formats": [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"],
        "max_file_size": "500MB",
        "supported_languages": {
            "transcription": ["en"],
            "translation": ["hi", "ta", "te"]
        }
    }


@router.get("/stream/{filename}")
async def stream_video(filename: str):
    """Stream a video file."""
    file_path = VIDEO_STORAGE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(file_path, media_type="video/mp4")
