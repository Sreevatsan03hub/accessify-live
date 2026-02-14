"""
Audio Routes
API endpoints for audio capture and processing.
Supports both live microphone and video file audio extraction.
"""
import os
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, WebSocket, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime

from services.unified_audio_pipeline import get_pipeline, UnifiedAudioPipeline
from utils.video_to_audio import ExtractionResult

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"
AUDIO_DIR = "uploads/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


# ==================== RESPONSE MODELS ====================

class AudioDeviceResponse(BaseModel):
    """Response for audio device listing"""
    devices: list
    count: int


class ExtractionResponse(BaseModel):
    """Response for audio extraction"""
    success: bool
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    error: Optional[str] = None


class CaptureStatusResponse(BaseModel):
    """Response for capture status"""
    is_streaming: bool
    source_type: Optional[str] = None
    sample_rate: int
    channels: int


class PipelineInfoResponse(BaseModel):
    """Response for pipeline information"""
    pipeline_info: dict


class VideoInfoResponse(BaseModel):
    """Response for video information"""
    duration: float
    width: int
    height: int
    fps: float
    audio_codec: str
    sample_rate: int
    channels: int
    bitrate: int
    file_size: int
    format: str


class ModelInfoResponse(BaseModel):
    """Response for model information"""
    model: str
    device: str
    language: str
    status: str


# ==================== SYNCHRONOUS ENDPOINTS ====================

@router.get("/devices", response_model=AudioDeviceResponse)
async def list_audio_devices():
    """
    List available audio input devices (microphones).
    """
    pipeline = get_pipeline()
    devices = pipeline.get_microphone_devices()
    
    return AudioDeviceResponse(
        devices=devices,
        count=len(devices)
    )


@router.get("/status", response_model=CaptureStatusResponse)
async def get_capture_status():
    """
    Get current audio capture status.
    """
    pipeline = get_pipeline()
    info = pipeline.get_pipeline_info()
    
    return CaptureStatusResponse(
        is_streaming=info.get('is_streaming', False),
        source_type=info.get('current_source'),
        sample_rate=info.get('sample_rate', 16000),
        channels=info.get('channels', 1)
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get Whisper model information.
    """
    try:
        from services.speech_to_text import get_stt_service
        stt = get_stt_service()
        return ModelInfoResponse(
            model=stt.model_name,
            device=str(stt.device),
            language="en",
            status="ready"
        )
    except Exception as e:
        logger.warning(f"Model info request (non-critical): {e}")
        # Return a default response instead of erroring
        return ModelInfoResponse(
            model="whisper-base",
            device="cpu",
            language="en",
            status="lazy-load"
        )


@router.post("/capture/start")
async def start_live_capture(
    device_id: Optional[int] = Query(None, description="Device index to use"),
    normalize: bool = Query(True, description="Apply audio normalization"),
    noise_reduction: bool = Query(True, description="Apply noise reduction")
):
    """
    Start live microphone audio capture.
    """
    pipeline = get_pipeline()
    
    success = pipeline.start_live_capture(
        device_id=device_id,
        apply_normalization=normalize,
        apply_noise_reduction=noise_reduction
    )
    
    if success:
        return {"status": "started", "message": "Microphone capture started"}
    else:
        raise HTTPException(status_code=500, detail="Failed to start microphone capture")


@router.post("/capture/stop")
async def stop_live_capture():
    """
    Stop live microphone audio capture.
    """
    pipeline = get_pipeline()
    pipeline.stop_live_capture()
    
    return {"status": "stopped", "message": "Microphone capture stopped"}


@router.post("/capture/segment", response_model=dict)
async def capture_segment(
    duration: float = Query(5.0, ge=1.0, le=300.0, description="Capture duration in seconds")
):
    """
    Capture a segment of live audio and save to file.
    """
    pipeline = get_pipeline()
    
    audio_data = pipeline.capture_live_segment(duration_seconds=duration)
    
    if audio_data:
        saved_path = pipeline.save_audio_data(audio_data)
        return {
            "success": True,
            "saved_path": saved_path,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to capture audio segment")


@router.post("/extract/video", response_model=ExtractionResponse)
async def extract_audio_from_video(
    video_file: UploadFile = File(..., description="Video file to extract audio from"),
    start_time: Optional[float] = Query(None, ge=0, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, ge=0, description="End time in seconds"),
    output_format: str = Query("wav", description="Output audio format (wav, mp3, flac)")
):
    """
    Extract audio from uploaded video file.
    """
    # Save uploaded video temporarily
    video_filename = f"{uuid.uuid4()}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    
    try:
        content = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Extract audio
        pipeline = get_pipeline()
        result = pipeline.extract_from_video(
            video_path=video_path,
            output_format=output_format,
            start_time=start_time,
            end_time=end_time
        )
        
        # Clean up video file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return ExtractionResponse(
            success=result.success,
            audio_path=result.audio_path,
            duration=result.duration,
            sample_rate=result.sample_rate,
            channels=result.channels,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/audio", response_model=ExtractionResponse)
async def extract_audio_from_audio_file(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    convert_to_mono: bool = Query(True, description="Convert stereo to mono"),
    target_sample_rate: Optional[int] = Query(16000, description="Target sample rate")
):
    """
    Process uploaded audio file (convert format, normalize, etc.).
    """
    pipeline = get_pipeline()
    
    try:
        content = await audio_file.read()
        
        # Save temporarily
        temp_filename = f"{uuid.uuid4()}_{audio_file.filename}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load and save with correct format
        audio_data = pipeline.load_audio_file(temp_path)
        
        if audio_data:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            saved_path = pipeline.save_audio_data(audio_data)
            
            return ExtractionResponse(
                success=True,
                audio_path=saved_path,
                duration=audio_data.duration,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to process audio file")
            
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{session_id}")
async def get_audio_info(session_id: str):
    """
    Get information about an audio session.
    """
    audio_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio session not found")
    
    pipeline = get_pipeline()
    audio_data = pipeline.load_audio_file(audio_path)
    
    if audio_data:
        return {
            "session_id": session_id,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate,
            "channels": audio_data.channels,
            "source_type": audio_data.source.source_type.value
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to read audio file")


@router.get("/download/{session_id}")
async def download_audio(session_id: str):
    """
    Download an audio file.
    """
    audio_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=audio_path,
        filename=f"accessify_{session_id}.wav",
        media_type="audio/wav"
    )


@router.post("/upload/{session_id}/audio")
async def upload_audio_file(
    session_id: str,
    audio_file: UploadFile = File(..., description="Audio file to upload")
):
    """
    Upload an audio file for a session.
    """
    pipeline = get_pipeline()
    
    try:
        content = await audio_file.read()
        
        # Save to audio directory
        output_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
        
        with open(output_path, "wb") as f:
            f.write(content)
        
        return {
            "success": True,
            "session_id": session_id,
            "audio_path": output_path
        }
        
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info", response_model=PipelineInfoResponse)
async def get_pipeline_info():
    """
    Get detailed information about the audio pipeline.
    """
    pipeline = get_pipeline()
    
    return PipelineInfoResponse(
        pipeline_info=pipeline.get_pipeline_info()
    )


# ==================== WEBSOCKET ENDPOINTS ====================

@router.websocket("/stream/live")
async def websocket_live_audio(websocket: WebSocket):
    """
    WebSocket endpoint for live microphone audio streaming.
    Streams audio chunks in real-time for speech-to-text processing.
    """
    await websocket.accept()
    
    pipeline = get_pipeline()
    
    # Get parameters from query string
    device_id = websocket.query_params.get("device_id")
    normalize = websocket.query_params.get("normalize", "true").lower() == "true"
    noise_reduction = websocket.query_params.get("noise_reduction", "true").lower() == "true"
    
    if device_id:
        device_id = int(device_id)
    
    # Start capture
    success = pipeline.start_live_capture(
        device_id=device_id,
        apply_normalization=normalize,
        apply_noise_reduction=noise_reduction
    )
    
    if not success:
        await websocket.send_json({"error": "Failed to start microphone capture"})
        await websocket.close()
        return
    
    logger.info("WebSocket live audio streaming started")
    
    try:
        while True:
            audio_data = pipeline.get_live_audio_chunk(timeout=1.0)
            
            if audio_data is not None:
                # Send audio data as base64 encoded
                import base64
                import numpy as np
                
                # Convert to 16-bit PCM
                audio_int16 = (audio_data.data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": audio_b64,
                    "sample_rate": audio_data.sample_rate,
                    "channels": audio_data.channels,
                    "duration": audio_data.duration,
                    "timestamp": audio_data.timestamp
                })
            else:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        pipeline.stop_live_capture()
        await websocket.close()
        logger.info("WebSocket live audio streaming stopped")


@router.websocket("/stream/video/{session_id}")
async def websocket_video_audio(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming extracted video audio.
    """
    await websocket.accept()
    
    audio_path = os.path.join(AUDIO_DIR, f"{session_id}.wav")
    
    if not os.path.exists(audio_path):
        await websocket.send_json({"error": "Audio session not found"})
        await websocket.close()
        return
    
    try:
        import wave
        import base64
        
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            # Send audio info
            await websocket.send_json({
                "type": "audio_info",
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width
            })
            
            # Stream audio chunks
            chunk_size = 4096
            
            while True:
                frames = wf.readframes(chunk_size)
                
                if not frames:
                    break
                
                audio_b64 = base64.b64encode(frames).decode('utf-8')
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "data": audio_b64
                })
        
        await websocket.send_json({"type": "end"})
        
    except Exception as e:
        logger.error(f"WebSocket video streaming error: {e}")
        await websocket.send_json({"error": str(e)})
    
    finally:
        await websocket.close()


# ==================== SPEECH-TO-TEXT ENDPOINTS ====================

class TranscriptionResponse(BaseModel):
    """Response for speech-to-text transcription"""
    success: bool
    text: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None
    words: Optional[list] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class TranscriptionConfigRequest(BaseModel):
    """Configuration for speech-to-text"""
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None
    word_timestamps: bool = True


@router.post("/transcribe/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Query(None, description="Language code (en, hi, ta, etc.)"),
    model_size: str = Query("base", description="Whisper model size"),
    word_timestamps: bool = Query(True, description="Include word-level timestamps")
):
    """
    Transcribe an audio file to text using OpenAI Whisper.
    """
    try:
        from services.speech_to_text import get_stt_service
        import tempfile
        import os
        
        # Save uploaded file temporarily
        temp_filename = f"{uuid.uuid4()}_{audio_file.filename}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        content = await audio_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        try:
            # Get STT service and transcribe
            stt = get_stt_service()
            logger.info(f"Transcribing {audio_file.filename} with model {model_size}")
            
            result = stt.transcribe_audio(
                audio_path=temp_path,
                language=language,
                model_size=model_size,
                word_timestamps=word_timestamps
            )
            
            logger.info(f"Transcription success: {result.text[:100]}")
            
            return TranscriptionResponse(
                success=True,
                text=result.text,
                language=result.language,
                duration=result.duration,
                confidence=result.confidence,
                words=result.words,
                processing_time=result.processing_time
            )
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/transcribe/video", response_model=TranscriptionResponse)
async def transcribe_video_file(
    video_file: UploadFile = File(..., description="Video file to transcribe"),
    language: Optional[str] = Query(None, description="Language code (en, hi, ta, etc.)"),
    model_size: str = Query("base", description="Whisper model size"),
    start_time: Optional[float] = Query(0, ge=0, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, ge=0, description="End time in seconds")
):
    """
    Transcribe a video file to text using OpenAI Whisper.
    Extracts audio from video and transcribes it.
    """
    try:
        from services.speech_to_text import get_stt_service
        
        # Save uploaded video temporarily
        video_filename = f"{uuid.uuid4()}_{video_file.filename}"
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        
        content = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        try:
            # Transcribe video
            stt = get_stt_service()
            result = stt.transcribe_video(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
                language=language
            )
            
            return TranscriptionResponse(
                success=True,
                text=result.text,
                language=result.language,
                duration=result.duration,
                confidence=result.confidence,
                words=result.words,
                processing_time=result.processing_time
            )
        finally:
            # Clean up temp file
            if os.path.exists(video_path):
                os.remove(video_path)
                
    except ImportError as e:
        logger.error(f"Speech-to-text service not available: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text service not available. Install openai-whisper.")
    except Exception as e:
        logger.error(f"Video transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transcribe/model-info")
async def get_stt_model_info():
    """
    Get information about the speech-to-text model.
    """
    try:
        from services.speech_to_text import get_stt_service
        stt = get_stt_service()
        return stt.get_model_info()
    except ImportError:
        return {
            "status": "not_available",
            "message": "Install openai-whisper to enable speech-to-text"
        }


@router.websocket("/transcribe/stream")
async def websocket_transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription.
    Streams transcribed text as audio chunks are received.
    """
    await websocket.accept()
    
    pipeline = get_pipeline()
    session_id = str(uuid.uuid4())
    
    try:
        from services.speech_to_text import get_stt_service
        import numpy as np
        import base64
        
        stt = get_stt_service()
        
        # Get parameters
        language = websocket.query_params.get("language", "en")
        sample_rate = int(websocket.query_params.get("sample_rate", 16000))
        
        logger.info(f"Starting real-time transcription stream (language: {language})")
        
        # Start pipeline session
        pipeline.start_stream_capture(
            session_id=session_id,
            sample_rate=sample_rate,
            channels=1
        )
        
        audio_buffer = []
        
        while True:
            try:
                message = await websocket.receive_json()
                
                if message.get("type") == "audio_chunk":
                    # Decode base64 audio
                    audio_b64 = message.get("data")
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Push to pipeline for processing (normalization, etc.)
                    processed_audio = pipeline.push_audio_chunk(
                        audio_data=audio_bytes,
                        sample_rate=sample_rate
                    )
                    
                    if processed_audio is not None and len(processed_audio.data) > 0:
                        audio_buffer.append(processed_audio.data)
                        
                        # Just debug logging here
                        # logger.debug(f"Pushed chunk duration: {processed_audio.duration}s")
                    
                elif message.get("type") == "end":
                    # Transcribe accumulated audio
                    if audio_buffer:
                        import torch
                        audio_combined = np.concatenate(audio_buffer)
                        
                        result = stt.transcribe_realtime_audio(
                            audio_data=audio_combined,
                            sample_rate=16000,
                            language=language
                        )
                        
                        await websocket.send_json({
                            "type": "transcription",
                            "text": result.text,
                            "language": result.language
                        })
                    
                    break
                    
            except Exception as e:
                logger.error(f"Streaming transcription error: {e}")
                await websocket.send_json({"error": str(e)})
                break
        
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
    
    finally:
        pipeline.stop_stream_capture()
        await websocket.close()
