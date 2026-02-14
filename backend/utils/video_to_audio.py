"""
Video to Audio Extraction Utility
Extracts audio from video files for speech-to-text processing.
Supports various video formats and provides both sync and async extraction.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple, Generator
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class AudioCodec(Enum):
    """Supported audio codecs for extraction"""
    PCM_16 = "pcm_s16le"
    PCM_32 = "pcm_s32le"
    MP3 = "libmp3lame"
    AAC = "aac"
    OPUS = "libopus"
    FLAC = "flac"


class VideoFormat(Enum):
    """Supported video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MKV = "mkv"
    MOV = "mov"
    WEBM = "webm"
    WMV = "wmv"
    FLV = "flv"


@dataclass
class VideoInfo:
    """Metadata about a video file"""
    duration: float  # Duration in seconds
    width: int  # Video width
    height: int  # Video height
    fps: float  # Frames per second
    audio_codec: str  # Audio codec name
    sample_rate: int  # Audio sample rate
    channels: int  # Number of audio channels
    bitrate: int  # Video bitrate in kbps
    file_size: int  # File size in bytes
    format: str  # Container format


@dataclass
class ExtractionResult:
    """Result of audio extraction"""
    success: bool
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    error: Optional[str] = None


class VideoToAudioExtractor:
    """
    Video to audio extraction class.
    Uses FFmpeg for audio extraction from video files.
    """
    
    # Default output settings optimized for speech recognition
    DEFAULT_OUTPUT_FORMAT = "wav"
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_CODEC = AudioCodec.PCM_16
    
    # Supported input formats
    SUPPORTED_FORMATS = {
        VideoFormat.MP4: [".mp4", ".m4v"],
        VideoFormat.AVI: [".avi"],
        VideoFormat.MKV: [".mkv"],
        VideoFormat.MOV: [".mov"],
        VideoFormat.WEBM: [".webm"],
        VideoFormat.WMV: [".wmv"],
        VideoFormat.FLV: [".flv"],
    }
    
    def __init__(
        self,
        output_dir: str = "uploads/audio",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        codec: AudioCodec = DEFAULT_CODEC
    ):
        """
        Initialize the extractor.
        
        Args:
            output_dir: Directory for extracted audio files
            sample_rate: Output sample rate (16000 for speech recognition)
            channels: Number of output channels (1 for mono)
            codec: Audio codec for output
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.codec = codec
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_ffmpeg_path(self) -> str:
        """Get FFmpeg executable path."""
        return "ffmpeg"
    
    def _validate_video_file(self, video_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the file is a supported video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(video_path):
            return False, f"File not found: {video_path}"
        
        ext = Path(video_path).suffix.lower()
        supported = False
        
        for video_format in self.SUPPORTED_FORMATS.values():
            if ext in video_format:
                supported = True
                break
        
        if not supported:
            return False, f"Unsupported video format: {ext}"
        
        return True, None
    
    def _probe_video(self, video_path: str) -> Optional[VideoInfo]:
        """
        Get video metadata using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo object or None on error
        """
        try:
            import subprocess
            
            ffprobe_path = ffprobe_path = "ffprobe"
            
            cmd = [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"ffprobe error: {result.stderr}")
                return None
            
            import json
            info = json.loads(result.stdout)
            
            # Find audio stream
            audio_stream = None
            video_stream = None
            
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                elif stream.get("codec_type") == "video":
                    video_stream = stream
            
            format_info = info.get("format", {})
            
            duration = float(format_info.get("duration", 0))
            
            if video_stream:
                width = int(video_stream.get("width", 0))
                height = int(video_stream.get("height", 0))
                fps_str = video_stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    fps_parts = fps_str.split("/")
                    fps = float(fps_parts[0]) / float(fps_parts[1]) if fps_parts[1] != "0" else 0
                else:
                    fps = float(fps_str)
            else:
                width, height, fps = 0, 0, 0
            
            if audio_stream:
                audio_codec = audio_stream.get("codec_name", "unknown")
                sample_rate = int(audio_stream.get("sample_rate", self.sample_rate))
                channels = int(audio_stream.get("channels", self.channels))
            else:
                audio_codec = "none"
                sample_rate = self.sample_rate
                channels = self.channels
            
            bitrate = int(format_info.get("bit_rate", 0)) // 1000
            file_size = int(format_info.get("size", 0))
            format_name = format_info.get("format_name", "unknown")
            
            return VideoInfo(
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                audio_codec=audio_codec,
                sample_rate=sample_rate,
                channels=channels,
                bitrate=bitrate,
                file_size=file_size,
                format=format_name
            )
            
        except subprocess.TimeoutExpired:
            logger.error("ffprobe timeout")
            return None
        except Exception as e:
            logger.error(f"Error probing video: {e}")
            return None
    
    def extract_audio(
        self,
        video_path: str,
        output_filename: Optional[str] = None,
        format: str = "wav",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        volume: float = 1.0
    ) -> ExtractionResult:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_filename: Output filename (auto-generated if None)
            format: Output audio format (wav, mp3, flac, etc.)
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)
            volume: Volume multiplier (1.0 = original)
            
        Returns:
            ExtractionResult object
        """
        # Validate input
        valid, error = self._validate_video_file(video_path)
        if not valid:
            return ExtractionResult(success=False, error=error)
        
        try:
            # Generate output filename if not provided
            if output_filename is None:
                output_filename = f"{uuid.uuid4()}.{format}"
            
            output_path = self.output_dir / output_filename
            
            # Get FFmpeg command
            cmd = self._build_ffmpeg_cmd(
                video_path=str(output_path),
                format=format,
                start_time=start_time,
                end_time=end_time,
                volume=volume
            )
            
            # Execute FFmpeg
            import subprocess
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max for large files
            )
            
            if result.returncode != 0:
                return ExtractionResult(
                    success=False,
                    error=f"FFmpeg error: {result.stderr}"
                )
            
            if not output_path.exists():
                return ExtractionResult(
                    success=False,
                    error="Audio file was not created"
                )
            
            # Get audio info
            info = self._get_audio_info(str(output_path))
            
            return ExtractionResult(
                success=True,
                audio_path=str(output_path),
                duration=info.get("duration"),
                sample_rate=info.get("sample_rate"),
                channels=info.get("channels")
            )
            
        except subprocess.TimeoutExpired:
            return ExtractionResult(
                success=False,
                error="Audio extraction timed out"
            )
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return ExtractionResult(
                success=False,
                error=str(e)
            )
    
    async def extract_audio_async(
        self,
        video_path: str,
        output_filename: Optional[str] = None,
        format: str = "wav",
        progress_callback=None
    ) -> ExtractionResult:
        """
        Extract audio from video file asynchronously.
        
        Args:
            video_path: Path to input video file
            output_filename: Output filename (auto-generated if None)
            format: Output audio format
            progress_callback: Optional callback for progress updates
            
        Returns:
            ExtractionResult object
        """
        loop = asyncio.get_event_loop()
        
        def run_extraction():
            return self.extract_audio(
                video_path,
                output_filename,
                format
            )
        
        result = await loop.run_in_executor(None, run_extraction)
        
        if progress_callback:
            progress_callback(100)  # 100% complete
        
        return result
    
    def _build_ffmpeg_cmd(
        self,
        video_path: str,
        format: str,
        start_time: Optional[float],
        end_time: Optional[float],
        volume: float
    ) -> list:
        """
        Build FFmpeg command for audio extraction.
        
        Args:
            video_path: Path to input video
            format: Output format
            start_time: Start time in seconds
            end_time: End time in seconds
            volume: Volume multiplier
            
        Returns:
            List of command arguments
        """
        cmd = [
            self._get_ffmpeg_path(),
            "-i", video_path,
            "-vn",  # No video output
        ]
        
        # Add audio filters
        filters = []
        
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])
        
        if volume != 1.0:
            filters.append(f"volume={volume}")
        
        if filters:
            cmd.extend(["-af", ",".join(filters)])
        
        # Output settings
        cmd.extend([
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-y"  # Overwrite output
        ])
        
        # Add codec based on format
        if format == "wav":
            cmd.extend(["-acodec", "pcm_s16le"])
        elif format == "mp3":
            cmd.extend(["-acodec", "libmp3lame", "-q:a", "2"])
        elif format == "flac":
            cmd.extend(["-acodec", "flac"])
        elif format == "aac":
            cmd.extend(["-acodec", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-acodec", "pcm_s16le"])
        
        cmd.append(video_path)
        
        return cmd
    
    def _get_audio_info(self, audio_path: str) -> dict:
        """
        Get information about an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        try:
            import subprocess
            
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {}
            
            import json
            info = json.loads(result.stdout)
            format_info = info.get("format", {})
            audio_stream = None
            
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break
            
            return {
                "duration": float(format_info.get("duration", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
                "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
                "codec": audio_stream.get("codec_name", "unknown") if audio_stream else "unknown",
                "file_size": int(format_info.get("size", 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}
    
    def extract_audio_chunks(
        self,
        video_path: str,
        chunk_duration: int = 60,
        format: str = "wav"
    ) -> Generator[ExtractionResult, None, None]:
        """
        Extract audio from video in chunks.
        Useful for processing long videos.
        
        Args:
            video_path: Path to input video
            chunk_duration: Duration of each chunk in seconds
            format: Output audio format
            
        Yields:
            ExtractionResult objects for each chunk
        """
        valid, error = self._validate_video_file(video_path)
        if not valid:
            yield ExtractionResult(success=False, error=error)
            return
        
        # Get video info
        info = self._probe_video(video_path)
        
        if info is None:
            yield ExtractionResult(success=False, error="Could not probe video")
            return
        
        total_duration = info.duration
        current_time = 0
        
        while current_time < total_duration:
            end_time = min(current_time + chunk_duration, total_duration)
            
            output_filename = f"chunk_{current_time}_{end_time}.{format}"
            
            result = self.extract_audio(
                video_path,
                output_filename=output_filename,
                format=format,
                start_time=current_time,
                end_time=end_time
            )
            
            yield result
            
            current_time = end_time
    
    def get_supported_formats(self) -> dict:
        """
        Get supported video formats.
        
        Returns:
            Dictionary mapping formats to file extensions
        """
        return {
            fmt.value: exts for fmt, exts in self.SUPPORTED_FORMATS.items()
        }


def extract_audio_from_upload(
    file_content: bytes,
    filename: str,
    output_dir: str = "uploads/audio",
    sample_rate: int = 16000
) -> ExtractionResult:
    """
    Convenience function to extract audio from uploaded file content.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        output_dir: Output directory
        sample_rate: Output sample rate
        
    Returns:
        ExtractionResult object
    """
    # Save uploaded file temporarily
    temp_dir = Path("uploads/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_video_path = temp_dir / f"temp_{uuid.uuid4()}_{filename}"
    
    try:
        with open(temp_video_path, "wb") as f:
            f.write(file_content)
        
        extractor = VideoToAudioExtractor(
            output_dir=output_dir,
            sample_rate=sample_rate
        )
        
        return extractor.extract_audio(str(temp_video_path))
    
    except Exception as e:
        return ExtractionResult(success=False, error=str(e))
    
    finally:
        # Clean up temp file
        if temp_video_path.exists():
            temp_video_path.unlink()
