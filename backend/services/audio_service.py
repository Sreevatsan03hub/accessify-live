"""
Audio Service Module for Live Microphone Capture
Handles real-time audio input from microphone for speech-to-text processing.
"""
import numpy as np
from typing import Optional, Callable, Generator
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    PCM_16BIT = "int16"
    PCM_32BIT = "int32"
    FLOAT_32BIT = "float32"


@dataclass
class AudioConfig:
    """Configuration for audio capture"""
    sample_rate: int = 16000  # 16kHz for speech recognition
    channels: int = 1  # Mono for speech
    chunk_size: int = 1024  # Samples per chunk
    dtype: str = "float32"  # Data type for audio samples
    buffer_size: int = 5  # Number of chunks to buffer


class MicrophoneCapture:
    """
    Real-time microphone audio capture class.
    Captures audio from microphone and provides it as a continuous stream.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize microphone capture.
        
        Args:
            config: AudioConfig object with capture settings
        """
        self.config = config or AudioConfig()
        self._is_capturing = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._device_info = None
        
    def list_devices(self) -> list:
        """
        List all available audio input devices.
        
        Returns:
            List of device dictionaries with name, index, and capabilities
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
            return [
                {
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'default_sample_rate': dev['default_samplerate']
                }
                for i, dev in enumerate(input_devices)
            ]
        except ImportError:
            logger.warning("sounddevice not installed. Install with: pip install sounddevice")
            return []
    
    def get_default_device(self) -> Optional[dict]:
        """
        Get the default audio input device.
        
        Returns:
            Device dictionary or None if no device found
        """
        devices = self.list_devices()
        if devices:
            # Try to find a device with "microphone" in name
            for dev in devices:
                if 'mic' in dev['name'].lower() or 'microphone' in dev['name'].lower():
                    return dev
            return devices[0]  # Return first available device
        return None
    
    def _capture_thread(self, device_index: Optional[int] = None):
        """
        Internal thread for continuous audio capture.
        
        Args:
            device_index: Specific device index to use, or None for default
        """
        try:
            import sounddevice as sd
            
            def audio_callback(indata, frames, time, status):
                """Callback function for each audio chunk."""
                if status:
                    logger.warning(f"Audio capture status: {status}")
                # Convert to proper format and put in queue
                audio_data = indata.copy()
                self._audio_queue.put(audio_data)
            
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                device=device_index,
                callback=audio_callback
            ):
                while self._is_capturing:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    
        except Exception as e:
            logger.error(f"Error in audio capture thread: {e}")
            self._is_capturing = False
    
    def start(self, device_index: Optional[int] = None) -> bool:
        """
        Start audio capture from microphone.
        
        Args:
            device_index: Specific device index, or None for default
            
        Returns:
            True if capture started successfully, False otherwise
        """
        if self._is_capturing:
            logger.warning("Audio capture already running")
            return True
        
        try:
            # Test device availability
            import sounddevice as sd
            
            # Try to get device info
            if device_index is None:
                default_dev = self.get_default_device()
                if default_dev:
                    device_index = default_dev['index']
            
            # Test the device
            sd.check_input_settings(
                device=device_index,
                samplerate=self.config.sample_rate,
                channels=self.config.channels
            )
            
            self._is_capturing = True
            self._thread = threading.Thread(
                target=self._capture_thread,
                args=(device_index,),
                daemon=True
            )
            self._thread.start()
            
            logger.info(f"Microphone capture started with device {device_index}")
            return True
            
        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice")
            return False
        except Exception as e:
            logger.error(f"Failed to start microphone capture: {e}")
            self._is_capturing = False
            return False
    
    def stop(self):
        """Stop audio capture."""
        if not self._is_capturing:
            return
        
        self._is_capturing = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Microphone capture stopped")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Maximum time to wait for a chunk (in seconds)
            
        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def read_chunks(self, duration_seconds: float = 5.0) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks for a specified duration.
        
        Args:
            duration_seconds: How long to capture audio
            
        Yields:
            Audio chunks as numpy arrays
        """
        self.start()
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds and self._is_capturing:
            chunk = self.get_audio_chunk(timeout=0.5)
            if chunk is not None:
                yield chunk
        
        self.stop()
    
    def capture_segment(self, duration_seconds: float = 5.0) -> Optional[np.ndarray]:
        """
        Capture a complete audio segment.
        
        Args:
            duration_seconds: Duration of audio to capture
            
        Returns:
            Complete audio segment as numpy array, or None on error
        """
        chunks = list(self.read_chunks(duration_seconds))
        
        if chunks:
            return np.concatenate(chunks, axis=0)
        return None
    
    def is_active(self) -> bool:
        """Check if audio capture is currently active."""
        return self._is_capturing
    
    def get_config(self) -> AudioConfig:
        """Get current audio configuration."""
        return self.config
    
    def get_audio_info(self) -> dict:
        """Get information about the current audio capture."""
        return {
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'chunk_size': self.config.chunk_size,
            'dtype': self.config.dtype,
            'is_active': self._is_capturing,
            'queue_size': self._audio_queue.qsize() if self._audio_queue else 0
        }


class AudioBuffer:
    """
    Circular buffer for managing audio chunks.
    Useful for real-time processing with overlap or lookahead.
    """
    
    def __init__(self, max_samples: int = 16000):  # 1 second at 16kHz
        """
        Initialize audio buffer.
        
        Args:
            max_samples: Maximum number of samples to buffer
        """
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.write_pos = 0
        self.max_samples = max_samples
        self.data_count = 0
    
    def write(self, data: np.ndarray):
        """
        Write audio data to buffer.
        
        Args:
            data: Audio data to write
        """
        available = self.max_samples - self.data_count
        
        if len(data) >= available:
            # Buffer full, keep only the last 'available' samples
            # This overwrites the oldest data
            self.buffer = data[-available:].copy()
            self.write_pos = len(data[-available:])
            self.data_count = self.max_samples
        else:
            # Add to buffer
            end_pos = (self.write_pos + len(data)) % self.max_samples
            
            if end_pos > self.write_pos:
                self.buffer[self.write_pos:end_pos] = data
            else:
                # Wrapped around
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:end_pos] = data[first_part:]
            
            self.write_pos = end_pos
            self.data_count = min(self.data_count + len(data), self.max_samples)
    
    def read(self, num_samples: int) -> np.ndarray:
        """
        Read audio data from buffer.
        
        Args:
            num_samples: Number of samples to read
            
        Returns:
            Audio data or zeros if not enough data
        """
        if num_samples > self.data_count:
            return np.zeros(num_samples, dtype=self.buffer.dtype)
        
        # Read from before write position
        read_pos = (self.write_pos - self.data_count) % self.max_samples
        
        if read_pos + num_samples <= self.max_samples:
            return self.buffer[read_pos:read_pos + num_samples].copy()
        else:
            # Wrapped around
            first_part = self.buffer[read_pos:].copy()
            remaining = num_samples - len(first_part)
            second_part = self.buffer[:remaining].copy()
            return np.concatenate([first_part, second_part])
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.data_count = 0
    
    def get_available_samples(self) -> int:
        """Get number of samples available in buffer."""
        return self.data_count
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.data_count >= self.max_samples


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.5) -> np.ndarray:
    """
    Normalize audio to target level.
    
    Args:
        audio_data: Input audio data
        target_level: Target peak level (0.0 to 1.0)
        
    Returns:
        Normalized audio data
    """
    if len(audio_data) == 0:
        return audio_data
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val * target_level
    return audio_data


def apply_noise_reduction(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Simple noise reduction by removing very low amplitude signals.
    
    Args:
        audio_data: Input audio data
        threshold: Silence threshold (0.0 to 1.0)
        
    Returns:
        Noise-reduced audio data
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Calculate RMS of entire audio
    rms = np.sqrt(np.mean(audio_data ** 2))
    
    if rms < threshold:
        # Entire audio is below threshold, return silence
        return np.zeros_like(audio_data)
    
    # Set very low amplitude samples to zero
    result = audio_data.copy()
    mask = np.abs(result) < threshold
    result[mask] = 0
    
    return result


def convert_audio_format(
    audio_data: np.ndarray, 
    from_dtype: str, 
    to_dtype: str
) -> np.ndarray:
    """
    Convert audio data between formats.
    
    Args:
        audio_data: Input audio data
        from_dtype: Source format
        to_dtype: Target format
        
    Returns:
        Converted audio data
    """
    type_map = {
        'int16': np.int16,
        'int32': np.int32,
        'float32': np.float32
    }
    
    from_np = type_map.get(from_dtype, np.float32)
    to_np = type_map.get(to_dtype, np.float32)
    
    if from_np == to_np:
        return audio_data
    
    # Convert to float first
    if from_np == np.int16:
        audio_data = audio_data.astype(np.float32) / 32767.0
    elif from_np == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483647.0
    
    # Convert from float
    if to_np == np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    elif to_np == np.int32:
        audio_data = (audio_data * 2147483647).astype(np.int32)
    
    return audio_data
