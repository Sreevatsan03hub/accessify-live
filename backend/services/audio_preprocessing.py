"""
Advanced Audio Preprocessing Module
Provides noise reduction, filtering, and normalization for improved transcription.
"""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def high_pass_filter(audio: np.ndarray, sample_rate: int = 16000, cutoff: float = 80.0) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency rumble.
    
    Args:
        audio: Input audio data (float32, -1 to 1)
        sample_rate: Sample rate in Hz
        cutoff: Cutoff frequency in Hz (default 80Hz removes rumble but keeps speech)
        
    Returns:
        Filtered audio
    """
    try:
        from scipy.signal import butter, filtfilt
        
        # Design 5th-order Butterworth high-pass filter
        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist
        b, a = butter(5, normalized_cutoff, btype='high')
        
        # Apply filter (zero-phase filtering)
        filtered = filtfilt(b, a, audio)
        return filtered.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"High-pass filter failed: {e}. Returning original audio.")
        return audio


def reduce_noise(audio: np.ndarray, sample_rate: int = 16000, noise_duration: float = 0.5) -> np.ndarray:
    """
    Reduce background noise using spectral gating.
    
    Args:
        audio: Input audio data (float32, -1 to 1)
        sample_rate: Sample rate in Hz
        noise_duration: Duration of initial silence to use for noise profile (seconds)
        
    Returns:
        Noise-reduced audio
    """
    try:
        import noisereduce as nr
        
        # Use first noise_duration seconds as noise profile (assumed to be silence/noise)
        noise_sample_count = int(noise_duration * sample_rate)
        
        if len(audio) > noise_sample_count:
            # Use beginning as noise profile
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.8  # Reduce noise by 80%
            )
        else:
            # Audio too short, skip noise reduction
            reduced = audio
            
        return reduced.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}. Returning original audio.")
        return audio


def normalize_rms(audio: np.ndarray, target_rms: float = 0.05, max_gain: float = 100.0) -> np.ndarray:
    """
    Normalize audio based on RMS (Root Mean Square) level.
    This is more robust than peak normalization for speech.
    
    Args:
        audio: Input audio data (float32, -1 to 1)
        target_rms: Target RMS level (default 0.05 = -26dBFS, good for Whisper)
        max_gain: Maximum gain multiplier to prevent noise amplification
        
    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < 1e-6:
        # Audio is essentially silence
        logger.warning("Audio RMS too low (silence). Skipping normalization.")
        return audio
    
    # Calculate required gain
    gain = target_rms / current_rms
    
    # Cap the gain to avoid noise explosions
    if gain > max_gain:
        logger.info(f"Capping gain at {max_gain}x (requested: {gain:.1f}x)")
        gain = max_gain
    
    # Apply gain
    normalized = audio * gain
    
    # Hard clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    
    logger.info(f"RMS normalization: {current_rms:.6f} -> {np.sqrt(np.mean(normalized**2)):.6f} (gain: {gain:.1f}x)")
    
    return normalized.astype(np.float32)


def preprocess_for_transcription(
    audio: np.ndarray,
    sample_rate: int = 16000,
    apply_highpass: bool = True,
    apply_noise_reduction: bool = True,
    apply_normalization: bool = True,
    target_rms: float = 0.05,
    max_gain: float = 100.0
) -> np.ndarray:
    """
    Full preprocessing pipeline for speech-to-text.
    
    Pipeline:
    1. High-pass filter (remove rumble)
    2. Noise reduction (spectral gating)
    3. RMS normalization (boost to optimal level)
    
    Args:
        audio: Input audio data (float32, -1 to 1)
        sample_rate: Sample rate in Hz
        apply_highpass: Enable high-pass filter
        apply_noise_reduction: Enable noise reduction
        apply_normalization: Enable RMS normalization
        target_rms: Target RMS level for normalization
        max_gain: Maximum gain multiplier
        
    Returns:
        Preprocessed audio ready for transcription
    """
    processed = audio.copy()
    
    # Step 1: High-pass filter
    if apply_highpass:
        processed = high_pass_filter(processed, sample_rate)
    
    # Step 2: Noise reduction
    if apply_noise_reduction:
        processed = reduce_noise(processed, sample_rate)
    
    # Step 3: RMS normalization
    if apply_normalization:
        processed = normalize_rms(processed, target_rms, max_gain)
    
    return processed
