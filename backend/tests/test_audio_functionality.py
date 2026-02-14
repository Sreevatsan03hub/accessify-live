"""
Unit Tests for Audio Functionality
Tests for the audio service, video-to-audio extraction, and unified pipeline.
"""
import os
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio_service import (
    AudioConfig,
    AudioBuffer,
    normalize_audio,
    apply_noise_reduction,
    convert_audio_format
)
from utils.video_to_audio import (
    VideoToAudioExtractor,
    ExtractionResult,
    VideoInfo,
    AudioCodec,
    VideoFormat
)
from config import get_config, AudioConfig as ConfigAudioConfig


# ==================== AUDIO SERVICE TESTS ====================

class TestAudioConfig:
    """Tests for AudioConfig dataclass."""
    
    def test_default_config(self):
        """Test default audio configuration."""
        config = AudioConfig()
        
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.dtype == "float32"
        assert config.buffer_size == 5
    
    def test_custom_config(self):
        """Test custom audio configuration."""
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=2048,
            dtype="int16"
        )
        
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 2048
        assert config.dtype == "int16"


class TestAudioBuffer:
    """Tests for AudioBuffer class."""
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = AudioBuffer(max_samples=32000)
        
        assert buffer.max_samples == 32000
        assert buffer.data_count == 0
        assert buffer.write_pos == 0
    
    def test_buffer_write(self):
        """Test writing to buffer."""
        buffer = AudioBuffer(max_samples=1000)
        test_data = np.random.randn(100).astype(np.float32)
        
        buffer.write(test_data)
        
        assert buffer.data_count == 100
    
    def test_buffer_read(self):
        """Test reading from buffer."""
        buffer = AudioBuffer(max_samples=1000)
        test_data = np.random.randn(100).astype(np.float32)
        
        buffer.write(test_data)
        read_data = buffer.read(100)
        
        assert len(read_data) == 100
        np.testing.assert_array_almost_equal(read_data, test_data)
    
    def test_buffer_read_less_than_written(self):
        """Test reading less data than written."""
        buffer = AudioBuffer(max_samples=1000)
        test_data = np.random.randn(200).astype(np.float32)
        
        buffer.write(test_data)
        read_data = buffer.read(50)
        
        assert len(read_data) == 50
    
    def test_buffer_read_more_than_available(self):
        """Test reading more data than available."""
        buffer = AudioBuffer(max_samples=1000)
        test_data = np.random.randn(50).astype(np.float32)
        
        buffer.write(test_data)
        read_data = buffer.read(100)
        
        assert len(read_data) == 100  # Padded with zeros
    
    def test_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = AudioBuffer(max_samples=1000)
        test_data = np.random.randn(100).astype(np.float32)
        
        buffer.write(test_data)
        buffer.clear()
        
        assert buffer.data_count == 0
    
    def test_buffer_full_check(self):
        """Test buffer full check."""
        buffer = AudioBuffer(max_samples=100)
        test_data = np.random.randn(50).astype(np.float32)
        
        assert not buffer.is_full()
        
        buffer.write(test_data)
        
        assert not buffer.is_full()
        
        buffer.write(np.random.randn(60).astype(np.float32))
        
        assert buffer.is_full()


class TestAudioProcessing:
    """Tests for audio processing functions."""
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        normalized = normalize_audio(audio, target_level=0.5)
        
        max_val = np.max(np.abs(normalized))
        assert max_val <= 0.5
    
    def test_normalize_empty_audio(self):
        """Test normalizing empty audio."""
        audio = np.array([], dtype=np.float32)
        
        normalized = normalize_audio(audio)
        
        assert len(normalized) == 0
    
    def test_normalize_silence(self):
        """Test normalizing silence."""
        audio = np.zeros(100, dtype=np.float32)
        
        normalized = normalize_audio(audio)
        
        assert len(normalized) == 100
    
    def test_apply_noise_reduction(self):
        """Test noise reduction."""
        audio = np.array([0.001, 0.002, 0.5, 0.002, 0.001], dtype=np.float32)
        
        reduced = apply_noise_reduction(audio, threshold=0.01)
        
        # Low amplitude samples should be zeroed
        assert reduced[0] == 0
        assert reduced[2] != 0  # High amplitude sample
    
    def test_convert_audio_format(self):
        """Test audio format conversion."""
        # Create int16 audio
        audio_int16 = np.array([0, 1000, 2000, 3000], dtype=np.int16)
        
        # Convert to float32
        audio_float = convert_audio_format(audio_int16, "int16", "float32")
        
        assert audio_float.dtype == np.float32
        assert np.max(np.abs(audio_float)) <= 1.0
    
    def test_convert_same_format(self):
        """Test converting to same format."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = convert_audio_format(audio, "float32", "float32")
        
        np.testing.assert_array_almost_equal(result, audio)


# ==================== VIDEO TO AUDIO TESTS ====================

class TestVideoToAudioExtractor:
    """Tests for VideoToAudioExtractor class."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = VideoToAudioExtractor(
            output_dir="/tmp/test_audio",
            sample_rate=16000,
            channels=1
        )
        
        assert extractor.sample_rate == 16000
        assert extractor.channels == 1
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        extractor = VideoToAudioExtractor()
        formats = extractor.get_supported_formats()
        
        assert "mp4" in formats
        assert "avi" in formats
        assert "mkv" in formats
    
    def test_validate_video_file_not_found(self):
        """Test validation of non-existent file."""
        extractor = VideoToAudioExtractor()
        
        valid, error = extractor._validate_video_file("/nonexistent/video.mp4")
        
        assert not valid
        assert "not found" in error.lower()
    
    def test_validate_unsupported_format(self):
        """Test validation of unsupported format."""
        extractor = VideoToAudioExtractor()
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            temp_path = f.name
        
        try:
            valid, error = extractor._validate_video_file(temp_path)
            
            assert not valid
            assert "unsupported" in error.lower()
        finally:
            os.unlink(temp_path)


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""
    
    def test_successful_result(self):
        """Test successful extraction result."""
        result = ExtractionResult(
            success=True,
            audio_path="/path/to/audio.wav",
            duration=10.5,
            sample_rate=16000,
            channels=1
        )
        
        assert result.success
        assert result.audio_path == "/path/to/audio.wav"
        assert result.duration == 10.5
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed extraction result."""
        result = ExtractionResult(
            success=False,
            error="Test error message"
        )
        
        assert not result.success
        assert result.error == "Test error message"


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""
    
    def test_video_info_creation(self):
        """Test video info creation."""
        info = VideoInfo(
            duration=120.5,
            width=1920,
            height=1080,
            fps=30.0,
            audio_codec="aac",
            sample_rate=48000,
            channels=2,
            bitrate=5000,
            file_size=1000000000,
            format="mp4"
        )
        
        assert info.duration == 120.5
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0


# ==================== CONFIG TESTS ====================

class TestConfig:
    """Tests for configuration."""
    
    def test_get_audio_config(self):
        """Test getting audio configuration."""
        config = get_config()
        
        assert config.audio.sample_rate == 16000
        assert config.audio.channels == 1
    
    def test_get_video_config(self):
        """Test getting video configuration."""
        config = get_config()
        
        assert ".mp4" in config.video.supported_formats
        assert ".avi" in config.video.supported_formats
    
    def test_get_translation_config(self):
        """Test getting translation configuration."""
        config = get_config()
        
        assert "hi" in config.translation.supported_languages
        assert "ta" in config.translation.supported_languages
        assert "te" in config.translation.supported_languages
    
    def test_config_singleton(self):
        """Test configuration singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2


# ==================== HELPER FUNCTIONS ====================

def create_test_audio_file(path: str, sample_rate: int = 16000, duration: float = 1.0):
    """Create a test audio WAV file."""
    import wave
    
    num_samples = int(sample_rate * duration)
    audio_data = (np.random.randn(num_samples) * 32767).astype(np.int16)
    
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
