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
_stt_service = None


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    language: str
    duration: float
    confidence: float = 0.0
    words: Optional[List[dict]] = None
    processing_time: float = 0.0


class WhisperSTT:
    """Whisper-based speech-to-text service."""
    
    def __init__(self, model_size: str = "base", device: str = None):
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
            
            # Convert to float32 and normalize
            audio = audio.astype(np.float32)
            
            # Transcribe using the loaded audio array
            logger.info("Starting transcription...")
            result = model.transcribe(
                audio,
                language=language,
                verbose=False
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract words if available
            words = None
            if word_timestamps and "words" in result:
                words = result["words"]
            
            logger.info(f"Transcription complete: {result.get('text', '')[:100]}")
            
            return TranscriptionResult(
                text=result.get("text", ""),
                language=result.get("language", language or "en"),
                duration=len(audio) / sr,
                confidence=result.get("confidence", 0.0),
                words=words,
                processing_time=processing_time
            )
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
    
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
        Transcribe audio data in real-time.
        
        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            TranscriptionResult with transcribed text
        """
        try:
            import tempfile
            import soundfile as sf
            import os
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio_path = tmp.name
            
            try:
                # Write audio data
                sf.write(tmp_audio_path, audio_data, sample_rate)
                
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


def get_stt_service(model_size: str = "base", device: str = None) -> WhisperSTT:
    """
    Get or create the singleton STT service.
    
    Args:
        model_size: Model size (default: base)
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
