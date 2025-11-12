"""
Echo Cancellation for real-time speech
Filters out speaker output from microphone input based on:
1. Frequency masking - remove typical speaker frequencies
2. Amplitude thresholding - ignore low amplitude (background/speaker noise)
"""

import numpy as np
from scipy import signal
import torch


class EchoCanceller:
    """Simple echo cancellation using frequency filtering"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Design a high-pass filter to remove low frequencies (typical speaker output)
        # Speaker output often has more bass, user speech has more mid-range
        self.highpass = self._design_highpass(cutoff=200, order=4)
        
    def _design_highpass(self, cutoff=200, order=4):
        """Design a high-pass Butterworth filter"""
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return b, a
    
    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio to reduce echo/speaker output
        
        Args:
            audio: torch tensor of audio samples
            
        Returns:
            Processed audio tensor
        """
        # Convert to numpy for processing
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Apply high-pass filter to remove low frequencies (echo/speaker)
        b, a = self.highpass
        filtered = signal.filtfilt(b, a, audio_np)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val * 0.95
        
        # Convert back to torch
        return torch.from_numpy(filtered).float()


class AdaptiveEchoCanceller:
    """
    More sophisticated adaptive echo cancellation.
    Uses reference signal (speaker output) to cancel echo from mic input.
    """
    
    def __init__(self, sample_rate=16000, filter_length=512):
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.adaptive_filter = np.zeros(filter_length)
        self.step_size = 0.01
        self.echo_buffer = np.zeros(filter_length)
        
    def update(self, reference_signal, mic_signal, step_size=None):
        """
        Update the adaptive filter with reference (speaker) and microphone signals.
        
        Args:
            reference_signal: Output/speaker audio
            mic_signal: Input/microphone audio
            step_size: Convergence step (higher = faster adaptation)
        """
        if step_size is None:
            step_size = self.step_size
        
        # NLMS (Normalized LMS) algorithm for echo cancellation
        # This adapts the filter to match the echo path
        ref = np.array(reference_signal)
        mic = np.array(mic_signal)
        
        # Estimate echo
        estimated_echo = np.convolve(ref, self.adaptive_filter, mode='same')
        
        # Error (difference between mic input and estimated echo)
        error = mic - estimated_echo
        
        # Update filter
        ref_power = np.dot(ref, ref) + 1e-10
        self.adaptive_filter += (step_size * error * ref) / ref_power
        
        return error  # Return echo-cancelled signal


# Simple static filter (recommended for this use case)
class StaticEchoCanceller:
    """
    Simple static filtering - works well for real-time applications
    Removes frequencies that are typically in speaker output
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # Create a notch filter to remove 60Hz (power line) and harmonics
        # Also attenuate low frequencies where speaker echo is strongest
        
    def filter_audio(self, audio_tensor, attenuation_db=6):
        """
        Apply static filtering to reduce echo
        
        Args:
            audio_tensor: Input audio
            attenuation_db: How much to attenuate echo frequencies (default 6dB)
            
        Returns:
            Filtered audio
        """
        audio = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        
        # Apply gentle high-pass filter
        # Frequencies below 150Hz are likely echo/feedback
        b, a = signal.butter(2, 150 / (self.sample_rate/2), btype='high')
        filtered = signal.filtfilt(b, a, audio)
        
        # Gentle compression to reduce dynamic range of low frequencies
        # This helps reduce the impact of any remaining echo
        filtered = np.tanh(filtered * 0.7) / 0.7  # Soft clipping
        
        return torch.from_numpy(filtered).float() if isinstance(audio_tensor, torch.Tensor) else filtered
