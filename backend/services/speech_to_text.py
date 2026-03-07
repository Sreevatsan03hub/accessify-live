"""
Speech-to-Text Service
Wrapper around OpenAI Whisper for audio transcription.
"""

import logging
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Singleton instance
_stt_service =  None


@dataclass
class TranscriptionResult:
    """Standardized result from STT service."""
    text: str
    language: str
    duration: float
    confidence: float
    words: Optional[list] = None
    segments: Optional[list] = None  # Added for VTT generation
    processing_time: float = 0.0


class WhisperSTT:
    """Whisper-based speech-to-text service."""
    
    def __init__(self, model_size: str = "small", device: str = None):
        """
        Initialize Whisper STT service.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda, mps)
        """
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            logger.error("whisper library not installed. Install with: pip install openai-whisper")
            self.whisper = None
        
        self.model_size = model_size
        self._device = device or "cpu"
        self.model = None
        self._model_loaded = False
    
    @property
    def model_name(self):
        """Get the model name."""
        return f"whisper-{self.model_size}"
    
    @property
    def device(self):
        """Get the device being used."""
        return self._device
    
    def _load_model(self):
        """Load the Whisper model lazily."""
        if self._model_loaded or self.model is not None:
            return
        
        if self.whisper is None:
            raise RuntimeError("Whisper library not available")
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self._device}")
            self.model = self.whisper.load_model(self.model_size, device=self._device)
            self._model_loaded = True
            logger.info(f"Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self._model_loaded = False
            raise
    
    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        word_timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'hi', 'ta')
            model_size: Override model size for this transcription
            word_timestamps: Include word-level timestamps
            
        Returns:
            TranscriptionResult with transcribed text
        """
        try:
            import librosa
            import numpy as np
            
            self._load_model()
            
            start_time = datetime.now()
            
            # Use provided model size or default
            if model_size and model_size != self.model_size:
                model = self.whisper.load_model(model_size, device=self._device)
            else:
                model = self.model
            
            # Load audio using librosa (doesn't require ffmpeg)
            logger.info(f"Loading audio from {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Safety check for empty/short audio
            if len(audio) < 1600: # < 0.1s
                logger.warning("Audio too short for transcription, skipping.")
                return TranscriptionResult(
                    text="",
                    language="en",
                    duration=0.0,
                    confidence=0.0,
                    words=[],
                    segments=[]
                )

            # Convert to float32 and normalize
            audio = audio.astype(np.float32)
            
            # Transcribe using the loaded audio array
            logger.info("Starting transcription...")
            
            result = model.transcribe(
                audio,
                language=language,
                verbose=False,
                condition_on_previous_text=False, # Reduces hallucinations
                temperature=0.0, # Deterministic
                initial_prompt="Classroom lecture. Clear speech. Sreevatsan. Accessify. Student. Teacher."
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Filter hallucinations based on no_speech_prob and logprob if segments available
            final_text = ""
            if "segments" in result:
                valid_segments = []
                for segment in result["segments"]:
                    # Filter out segments with high probability of no speech (silence/noise)
                    no_speech_prob = segment.get("no_speech_prob", 0.0)
                    avg_logprob = segment.get("avg_logprob", 0.0)
                    compression_ratio = segment.get("compression_ratio", 0.0)
                    
                    logger.info(f"[STT-Segment] '{segment['text'].strip()}' "
                               f"(no_speech={no_speech_prob:.4f}, logprob={avg_logprob:.4f}, "
                               f"compress={compression_ratio:.4f})")

                    # Thresholds (stricter to reduce hallucinations):
                    # no_speech > 0.65 → likely silence/noise (Relaxed from 0.45)
                    # logprob  < -0.8  → low confidence (Relaxed from -0.6)
                    # compression > 2.0 → repetitive loop
                    if (no_speech_prob < 0.65 and 
                        avg_logprob > -0.8 and 
                        compression_ratio < 2.0):
                        valid_segments.append(segment["text"])
                    else:
                        logger.info(f"[STT-Filter] Dropped: '{segment['text'].strip()}'")
                
                final_text = "".join(valid_segments).strip()
            else:
                final_text = result.get("text", "").strip()
            
            # Extract words if available
            words = None
            if word_timestamps and "words" in result:
                words = result["words"]
            
            logger.info(f"Transcription complete in {processing_time:.2f}s: '{final_text}'")
            
            return TranscriptionResult(
                text=final_text,
                language=result.get("language", language or "en"),
                duration=len(audio) / sr,
                confidence=float(result.get("confidence", 0.0)),
                words=result.get("words"),
                segments=result.get("segments"),
                processing_time=processing_time
            )
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            # Return empty result instead of crashing
            return TranscriptionResult(
                text="",
                language="en",
                duration=0.0,
                confidence=0.0,
                words=[],
                segments=[]
            )

    def generate_vtt(self, segments: list) -> str:
        """
        Generate WebVTT formatted string from segments.
        
        Args:
            segments: List of segment dictionaries from Whisper
            
        Returns:
            VTT formatted string
        """
        vtt = "WEBVTT\n\n"
        
        for i, segment in enumerate(segments):
            start = self._format_timestamp(segment["start"])
            end = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            vtt += f"{i+1}\n"
            vtt += f"{start} --> {end}\n"
            vtt += f"{text}\n\n"
            
        return vtt

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into MM:SS.mmm for VTT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def transcribe_video(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio extracted from a video file.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            language: Language code
            
        Returns:
            TranscriptionResult with transcribed text
        """
        try:
            from utils.video_to_audio import VideoToAudioExtractor
            import tempfile
            import os
            
            # Extract audio from video
            extractor = VideoToAudioExtractor()
            
            # Create temporary file for extracted audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio_path = tmp.name
            
            try:
                result = extractor.extract_audio(
                    video_path=video_path,
                    output_path=tmp_audio_path,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if result.success:
                    # Transcribe extracted audio
                    transcription = self.transcribe_audio(
                        audio_path=tmp_audio_path,
                        language=language
                    )
                    
                    return transcription
                else:
                    raise Exception(f"Failed to extract audio: {result.error}")
            
            finally:
                # Clean up temporary audio file
                if os.path.exists(tmp_audio_path):
                    try:
                        os.remove(tmp_audio_path)
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Video transcription failed: {e}")
            raise
    
    def transcribe_realtime_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio data from a numpy array (used for real-time stream).
        """
        try:
            import soundfile as sf
            import os
            import uuid
            
            # Basic validation
            if audio_data is None or len(audio_data) < 1600:
                return TranscriptionResult(
                    text="",
                    language="en",
                    duration=0.0,
                    confidence=0.0,
                    words=[],
                    segments=[]
                )

            # Create temporary file
            tmp_filename = f"live_{uuid.uuid4()}.wav"
            tmp_audio_path = os.path.join(os.getcwd(), "tmp", tmp_filename)
            
            # Ensure tmp directory exists
            os.makedirs(os.path.join(os.getcwd(), "tmp"), exist_ok=True)
            
            try:
                # Write audio data
                sf.write(tmp_audio_path, audio_data, 16000) # Assuming 16kHz for real-time
                
                # Transcribe
                transcription = self.transcribe_audio(
                    audio_path=tmp_audio_path,
                    language=language
                )
                
                return transcription
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_audio_path):
                    try:
                        os.remove(tmp_audio_path)
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Real-time transcription failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model_name,
            "size": self.model_size,
            "device": self.device,
            "language": "multilingual",
            "status": "ready" if self.whisper else "not_available"
        }


def get_stt_service(model_size: str = "small", device: str = None) -> WhisperSTT:
    """
    Get or create the singleton STT service.
    
    Args:
        model_size: Model size (default: small)
        device: Device to use (default: cpu)
        
    Returns:
        WhisperSTT instance
    """
    global _stt_service
    
    if _stt_service is None:
        _stt_service = WhisperSTT(model_size=model_size, device=device)
    
    return _stt_service


def reset_stt_service():
    """Reset the singleton STT service."""
    global _stt_service
    _stt_service = None
