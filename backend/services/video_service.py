"""
Video Processing Service
Extracts audio from video files and processes through STT/Translation pipeline.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingResult:
    """Result of video processing."""
    success: bool
    audio_path: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


class VideoProcessor:
    """Process video files and extract audio."""
    
    SUPPORTED_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv']
    
    def __init__(self):
        """Initialize video processor."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are available."""
        try:
            # Try v2.0+ import structure
            from moviepy.video.io.VideoFileClip import VideoFileClip
            self.moviepy = True
            self.VideoFileClip = VideoFileClip
            logger.info("MoviePy library loaded successfully (v2.0+)")
        except ImportError:
            try:
                # Fallback to v1.0 structure
                from moviepy.editor import VideoFileClip
                self.moviepy = True
                self.VideoFileClip = VideoFileClip
                logger.info("MoviePy library loaded successfully (v1.0)")
            except ImportError as e:
                logger.warning(f"MoviePy not installed. Error: {e}")
                self.moviepy = None
        except Exception as e:
            logger.warning(f"Error loading MoviePy: {e}")
            self.moviepy = None
    
    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> VideoProcessingResult:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file (default: temp file)
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            VideoProcessingResult with audio path or error
        """
        if self.moviepy is None:
            return VideoProcessingResult(
                success=False,
                error="MoviePy library not installed"
            )
        
        # Validate video file
        video_path = Path(video_path)
        if not video_path.exists():
            return VideoProcessingResult(
                success=False,
                error=f"Video file not found: {video_path}"
            )
        
        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return VideoProcessingResult(
                success=False,
                error=f"Unsupported format: {video_path.suffix}. Supported: {self.SUPPORTED_FORMATS}"
            )
        
        try:
            # Create output path if not provided
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_path = temp_file.name
                temp_file.close()
            
            logger.info(f"Extracting audio from: {video_path}")
            
            # Load video
            video = self.VideoFileClip(str(video_path))
            
            # Apply time range if specified
            if start_time is not None or end_time is not None:
                start = start_time or 0
                end = end_time or video.duration
                video = video.subclip(start, end)
            
            duration = video.duration
            
            # Extract audio
            audio = video.audio
            if audio is None:
                video.close()
                return VideoProcessingResult(
                    success=False,
                    error="Video has no audio track"
                )
            
            # Write audio to file
            audio.write_audiofile(
                output_path,
                fps=16000,  # 16kHz sample rate (same as our STT pipeline)
                nbytes=2,
                codec='pcm_s16le',
                logger=None  # Suppress moviepy's verbose logging
            )
            
            # Clean up
            video.close()
            
            logger.info(f"Audio extracted successfully: {output_path} (duration: {duration:.2f}s)")
            
            return VideoProcessingResult(
                success=True,
                audio_path=output_path,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}", exc_info=True)
            return VideoProcessingResult(
                success=False,
                error=str(e)
            )
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if self.moviepy is None:
            return {"error": "MoviePy not installed"}
        
        try:
            video = self.VideoFileClip(video_path)
            info = {
                "duration": video.duration,
                "fps": video.fps,
                "size": video.size,
                "has_audio": video.audio is not None
            }
            video.close()
            return info
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {"error": str(e)}


# Singleton instance
_video_processor = None


def get_video_processor() -> VideoProcessor:
    """Get singleton video processor instance."""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
