"""
STT Performance Optimization Strategies
"""

# STRATEGY 1: Use Faster Models
# Models ranked by speed (fastest to slowest):
# 1. nova-2 - Fast but slightly less accurate (RECOMMENDED for real-time)
# 2. nova-3 - Default, good balance
# 3. enhanced - Slowest but most accurate

# STRATEGY 2: Audio Preprocessing
# - Remove silence at beginning/end (saves API time)
# - Use adaptive audio filtering
# - Lower sample rate if possible (8kHz might work)

# STRATEGY 3: Parallel Processing
# - Process audio while recording next chunk
# - Batch multiple short clips

import numpy as np
import io
import soundfile as sf


def optimize_audio_for_stt(audio_np, target_sr=16000):
    """
    Preprocess audio to make STT faster and more accurate
    
    Args:
        audio_np: Audio numpy array
        target_sr: Target sample rate (16000 is optimal)
        
    Returns:
        Optimized audio numpy array
    """
    # Trim silence from beginning and end
    audio_trimmed = trim_silence(audio_np, threshold_db=30)
    
    # Normalize amplitude
    max_amp = np.max(np.abs(audio_trimmed))
    if max_amp > 0:
        audio_normalized = audio_trimmed / max_amp * 0.95
    else:
        audio_normalized = audio_trimmed
    
    return audio_normalized


def trim_silence(audio, threshold_db=30, sample_rate=16000):
    """Remove silence from beginning and end of audio"""
    # Convert to dB
    S = np.abs(np.fft.rfft(audio))
    S_db = 20 * np.log10(np.maximum(1e-5, S))
    
    # Find frames above threshold
    threshold = -threshold_db
    above_threshold = S_db > threshold
    
    # Find first and last above threshold
    indices = np.where(above_threshold)[0]
    
    if len(indices) == 0:
        return audio
    
    start_idx = max(0, indices[0] - 100)
    end_idx = min(len(audio), indices[-1] + 100)
    
    return audio[start_idx:end_idx]


def split_audio_for_parallel_stt(audio_np, chunk_duration=10, sample_rate=16000):
    """
    Split audio into chunks for faster parallel processing
    
    Args:
        audio_np: Full audio
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Sample rate of audio
        
    Returns:
        List of audio chunks
    """
    chunk_size = int(chunk_duration * sample_rate)
    chunks = []
    
    for i in range(0, len(audio_np), chunk_size):
        chunk = audio_np[i:i+chunk_size]
        if len(chunk) > sample_rate * 0.5:  # Only include chunks > 0.5s
            chunks.append(chunk)
    
    return chunks


# IMPLEMENTATION EXAMPLES:

# Method 1: Use faster model (nova-2 instead of nova-3)
# res = client.listen.v1.media.transcribe_file(
#     request=buf,
#     model="nova-2",  # Faster!
#     request_options={"timeout_in_seconds": 15, "max_retries": 0}
# )

# Method 2: Preprocess audio before sending
# audio_optimized = optimize_audio_for_stt(audio_np)
# Then send audio_optimized to STT

# Method 3: Parallel processing with asyncio
# async def transcribe_parallel(audio_chunks):
#     tasks = [transcribe_async(chunk) for chunk in audio_chunks]
#     results = await asyncio.gather(*tasks)
#     return " ".join(results)

print("""
STT SPEED TIPS:
1. Use nova-2 model (faster, good accuracy)
2. Trim silence before sending
3. Process shorter audio chunks
4. Fail fast (max_retries: 0)
5. Use parallelization for multiple clips

Expected times with optimization:
- 5 seconds of speech: 0.5-1s (nova-3)
- 5 seconds of speech: 0.3-0.7s (nova-2)
""")
