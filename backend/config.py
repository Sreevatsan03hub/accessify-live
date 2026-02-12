"""
Accessify Configuration
Centralized configuration for the Accessify platform.
"""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class Environment(Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000  # 16kHz for speech recognition
    channels: int = 1  # Mono for speech
    chunk_size: int = 1024  # Samples per chunk
    buffer_size: int = 5  # Number of chunks to buffer
    dtype: str = "float32"  # Data type for audio samples
    noise_threshold: float = 0.01  # Silence threshold
    normalization_level: float = 0.5  # Target normalization level
    capture_duration: int = 5  # Default capture duration in seconds


@dataclass
class VideoConfig:
    """Video processing configuration"""
    supported_formats: list = field(default_factory=lambda: [
        ".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv"
    ])
    max_file_size_mb: int = 500  # Maximum upload size
    chunk_duration: int = 60  # Chunk duration for extraction
    output_format: str = "wav"


@dataclass
class StorageConfig:
    """Storage configuration"""
    upload_dir: str = "uploads"
    audio_dir: str = "uploads/audio"
    temp_dir: str = "uploads/temp"
    max_storage_gb: int = 10


@dataclass
class SpeechToTextConfig:
    """Speech-to-text configuration"""
    model: str = "base"  # Whisper model size (tiny, base, small, medium, large)
    language: str = "en"  # Default language
    beam_size: int = 5  # Beam size for decoding
    word_timestamps: bool = True  # Include word-level timestamps
    vad_filter: bool = True  # Filter out non-speech


@dataclass
class TranslationConfig:
    """Translation configuration"""
    supported_languages: list = field(default_factory=lambda: [
        "en", "hi", "ta", "te", "bn", "kn", "ml", "mr", "gu", "pa"
    ])
    default_target: str = "hi"  # Default target language


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 27017
    name: str = "accessify"
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration"""
    name: str = "Accessify"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])
    
    # Component configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    speech_to_text: SpeechToTextConfig = field(default_factory=SpeechToTextConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


# Singleton configuration instance
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Get the application configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        AppConfig instance
    """
    global _config
    
    if _config is None:
        _config = AppConfig()
        
        # Override with environment variables if set
        _config.audio.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", _config.audio.sample_rate))
        _config.audio.channels = int(os.getenv("AUDIO_CHANNELS", _config.audio.channels))
        _config.audio.chunk_size = int(os.getenv("AUDIO_CHUNK_SIZE", _config.audio.chunk_size))
        
        _config.speech_to_text.model = os.getenv("STT_MODEL", _config.speech_to_text.model)
        _config.speech_to_text.language = os.getenv("STT_LANGUAGE", _config.speech_to_text.language)
        
        _config.database.host = os.getenv("DB_HOST", _config.database.host)
        _config.database.port = int(os.getenv("DB_PORT", _config.database.port))
        _config.database.name = os.getenv("DB_NAME", _config.database.name)
        
        _config.debug = os.getenv("DEBUG", str(_config.debug)).lower() == "true"
        
        if os.getenv("ENVIRONMENT"):
            try:
                _config.environment = Environment(os.getenv("ENVIRONMENT"))
            except ValueError:
                pass
    
    return _config


def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file or environment."""
    global _config
    _config = None
    return get_config(config_path)


# Convenience functions
def get_audio_config() -> AudioConfig:
    """Get audio configuration."""
    return get_config().audio


def get_video_config() -> VideoConfig:
    """Get video configuration."""
    return get_config().video


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    return get_config().storage


def get_speech_to_text_config() -> SpeechToTextConfig:
    """Get speech-to-text configuration."""
    return get_config().speech_to_text


def get_translation_config() -> TranslationConfig:
    """Get translation configuration."""
    return get_config().translation


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database
