"""
Unified Audio Pipeline
Combines live microphone capture and video-to-audio extraction
into a single interface for speech-to-text processing.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Union, Generator, Dict, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import asyncio

from services.audio_service import (
    MicrophoneCapture,
    AudioBuffer,
    AudioConfig,
    normalize_audio,
    apply_noise_reduction
)
from utils.video_to_audio import (
    VideoToAudioExtractor,
    ExtractionResult,
    VideoInfo
)

logger = logging.getLogger(__name__)


class AudioSourceType(Enum):
    """Type of audio source"""
    MICROPHONE = "microphone"
    VIDEO_FILE = "video_file"
    AUDIO_FILE = "audio_file"


@dataclass
class AudioSource:
    """Represents an audio source"""
    source_type: AudioSourceType
    path: Optional[str] = None
    device_id: Optional[int] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AudioData:
    """Container for audio data"""
    data: any  # numpy array or bytes
    sample_rate: int
    channels: int
    duration: float
    source: AudioSource
    timestamp: float = 0.0


class UnifiedAudioPipeline:
    """
    Unified interface for audio input from multiple sources.
    Supports both live microphone and video file audio extraction.
    """
    
    def __init__(
        self,
        output_dir: str = "uploads/audio",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024
    ):
        """
        Initialize the unified audio pipeline.
        
        Args:
            output_dir: Directory for extracted audio files
            sample_rate: Sample rate for audio processing
            channels: Number of audio channels
            chunk_size: Chunk size for streaming
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.mic_config = AudioConfig(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size
        )
        self.microphone = MicrophoneCapture(self.mic_config)
        self.video_extractor = VideoToAudioExtractor(
            output_dir=str(self.output_dir),
            sample_rate=sample_rate,
            channels=channels
        )
        
        # Buffer for real-time processing
        self.audio_buffer = AudioBuffer(max_samples=sample_rate * 5)  # 5 seconds buffer
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # State
        self._is_streaming = False
        self._current_source: Optional[AudioSource] = None
    
    # ==================== MICROPHONE METHODS ====================
    
    def get_microphone_devices(self) -> list:
        """
        Get list of available microphone devices.
        
        Returns:
            List of device dictionaries
        """
        return self.microphone.list_devices()
    
    def start_live_capture(
        self,
        device_id: Optional[int] = None,
        apply_normalization: bool = True,
        apply_noise_reduction: bool = True
    ) -> bool:
        """
        Start live microphone capture.
        
        Args:
            device_id: Specific device index, or None for default
            apply_normalization: Whether to normalize audio
            apply_noise_reduction: Whether to apply noise reduction
            
        Returns:
            True if started successfully
        """
        success = self.microphone.start(device_id=device_id)
        
        if success:
            self._current_source = AudioSource(
                source_type=AudioSourceType.MICROPHONE,
                device_id=device_id,
                metadata={
                    'normalization': apply_normalization,
                    'noise_reduction': apply_noise_reduction
                }
            )
            self._is_streaming = True
        
        return success
    
    def stop_live_capture(self):
        """Stop live microphone capture."""
        self.microphone.stop()
        self._is_streaming = False
        self._current_source = None
    
    def get_live_audio_chunk(
        self,
        timeout: float = 1.0,
        apply_processing: bool = True
    ) -> Optional[AudioData]:
        """
        Get a chunk of live audio from microphone.
        
        Args:
            timeout: Maximum wait time for chunk
            apply_processing: Whether to apply normalization/noise reduction
            
        Returns:
            AudioData object or None
        """
        chunk = self.microphone.get_audio_chunk(timeout=timeout)
        
        if chunk is None:
            return None
        
        # Apply processing if enabled
        if apply_processing and self._current_source:
            metadata = self._current_source.metadata or {}
            
            if metadata.get('normalization', True):
                chunk = normalize_audio(chunk)
            
            if metadata.get('noise_reduction', True):
                chunk = apply_noise_reduction(chunk)
        
        # Calculate duration
        duration = len(chunk) / self.mic_config.sample_rate
        
        return AudioData(
            data=chunk,
            sample_rate=self.mic_config.sample_rate,
            channels=self.mic_config.channels,
            duration=duration,
            source=self._current_source or AudioSource(source_type=AudioSourceType.MICROPHONE),
            timestamp=0.0
        )
    
    def stream_live_audio(
        self,
        duration_seconds: float = None,
        chunk_callback=None
    ) -> Generator[AudioData, None, None]:
        """
        Stream live audio for a duration or continuously.
        
        Args:
            duration_seconds: Stream duration (None for indefinite)
            chunk_callback: Optional callback for each chunk
            
        Yields:
            AudioData objects
        """
        self.start_live_capture()
        
        import time
        start_time = time.time()
        
        while self._is_streaming:
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break
            
            audio_data = self.get_live_audio_chunk(timeout=0.5)
            
            if audio_data:
                if chunk_callback:
                    chunk_callback(audio_data)
                
                yield audio_data
        
        self.stop_live_capture()
    
    def capture_live_segment(self, duration_seconds: float = 5.0) -> Optional[AudioData]:
        """
        Capture a complete segment of live audio.
        
        Args:
            duration_seconds: Duration to capture
            
        Returns:
            AudioData object or None
        """
        audio_segments = []
        total_duration = 0.0
        
        for audio_data in self.stream_live_audio(duration_seconds):
            audio_segments.append(audio_data.data)
            total_duration = audio_data.duration
        
        if audio_segments:
            import numpy as np
            combined = np.concatenate(audio_segments, axis=0)
            
            return AudioData(
                data=combined,
                sample_rate=self.mic_config.sample_rate,
                channels=self.mic_config.channels,
                duration=total_duration,
                source=self._current_source or AudioSource(source_type=AudioSourceType.MICROPHONE)
            )
        
        return None
    
    # ==================== VIDEO/AUDIO FILE METHODS ====================
    
    def extract_from_video(
        self,
        video_path: str,
        output_format: str = "wav",
        start_time: float = None,
        end_time: float = None
    ) -> ExtractionResult:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_format: Output audio format
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            ExtractionResult object
        """
        return self.video_extractor.extract_audio(
            video_path=video_path,
            format=output_format,
            start_time=start_time,
            end_time=end_time
        )
    
    def extract_from_uploaded_file(
        self,
        file_content: bytes,
        filename: str
    ) -> ExtractionResult:
        """
        Extract audio from uploaded file content.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            ExtractionResult object
        """
        return self.video_extractor.extract_audio_from_upload(
            file_content=file_content,
            filename=filename,
            output_dir=str(self.output_dir),
            sample_rate=self.mic_config.sample_rate
        )
    
    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object or None
        """
        return self.video_extractor._probe_video(video_path)
    
    def extract_video_chunks(
        self,
        video_path: str,
        chunk_duration: int = 60,
        format: str = "wav"
    ) -> Generator[ExtractionResult, None, None]:
        """
        Extract audio from video in chunks.
        
        Args:
            video_path: Path to video file
            chunk_duration: Duration of each chunk
            format: Output format
            
        Yields:
            ExtractionResult objects
        """
        return self.video_extractor.extract_audio_chunks(
            video_path=video_path,
            chunk_duration=chunk_duration,
            format=format
        )
    
    # ==================== UTILITY METHODS ====================
    
    def save_audio_data(
        self,
        audio_data: AudioData,
        filename: Optional[str] = None,
        format: str = "wav"
    ) -> str:
        """
        Save audio data to file.
        
        Args:
            audio_data: AudioData object to save
            filename: Output filename (auto-generated if None)
            format: Audio format
            
        Returns:
            Path to saved file
        """
        import wave
        import numpy as np
        
        if filename is None:
            filename = f"{uuid.uuid4()}.{format}"
        
        output_path = self.output_dir / filename
        
        # Convert float to 16-bit PCM
        audio_int16 = (audio_data.data * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(audio_data.channels)
            wf.setsampwidth(2)  # 2 bytes for 16-bit
            wf.setframerate(audio_data.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return str(output_path)
    
    def load_audio_file(self, audio_path: str) -> Optional[AudioData]:
        """
        Load audio from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            AudioData object or None
        """
        import wave
        import numpy as np
        
        if not os.path.exists(audio_path):
            return None
        
        try:
            with wave.open(audio_path, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                frames = wf.readframes(n_frames)
                
                # Convert to numpy array
                dtype = np.int16 if sample_width == 2 else np.float32
                audio_data = np.frombuffer(frames, dtype=dtype)
                
                # Convert to float
                if dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                
                # Convert to mono if stereo
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels).mean(axis=1)
                
                duration = len(audio_data) / sample_rate
                
                return AudioData(
                    data=audio_data,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration=duration,
                    source=AudioSource(
                        source_type=AudioSourceType.AUDIO_FILE,
                        path=audio_path
                    )
                )
                
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline state.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'is_streaming': self._is_streaming,
            'current_source': self._current_source.source_type.value if self._current_source else None,
            'sample_rate': self.mic_config.sample_rate,
            'channels': self.mic_config.channels,
            'chunk_size': self.mic_config.chunk_size,
            'buffer_size': self.audio_buffer.get_available_samples(),
            'microphone_devices': self.microphone.list_devices()
        }
    
    def shutdown(self):
        """Shutdown the pipeline and release resources."""
        if self._is_streaming:
            self.stop_live_capture()
        
        self.executor.shutdown(wait=False)
        logger.info("Audio pipeline shutdown complete")


# Singleton instance for easy access
_pipeline: Optional[UnifiedAudioPipeline] = None


def get_pipeline() -> UnifiedAudioPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    
    if _pipeline is None:
        _pipeline = UnifiedAudioPipeline()
    
    return _pipeline


def init_pipeline(
    output_dir: str = "uploads/audio",
    sample_rate: int = 16000,
    channels: int = 1
) -> UnifiedAudioPipeline:
    """Initialize the global pipeline instance."""
    global _pipeline
    
    _pipeline = UnifiedAudioPipeline(
        output_dir=output_dir,
        sample_rate=sample_rate,
        channels=channels
    )
    
    return _pipeline
