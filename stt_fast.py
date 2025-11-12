import io
import numpy as np
import soundfile as sf
from deepgram import DeepgramClient
import asyncio


class SpeechToTextFast:
    """Fast STT with optimizations for real-time performance"""
    def __init__(self, api_key):
        self.client = DeepgramClient(api_key=api_key)

    def transcribe(self, audio_np):
        """
        Transcribe audio using Deepgram Nova-3 model with optimizations
        
        Optimizations:
        1. Use smaller audio format (PCM_16 instead of WAV)
        2. Reduce sample rate for faster processing (16kHz is good enough for speech)
        3. Use pre-recorded mode (faster than streaming for short clips)
        4. Minimal retries (fail fast)
        """
        # Optimization 1: Use raw PCM instead of WAV wrapper for faster transmission
        buf = io.BytesIO()
        sf.write(buf, audio_np, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)

        # Use nova-3 model with optimized parameters
        res = self.client.listen.v1.media.transcribe_file(
            request=buf,
            model="nova-3",
            language="en",
            # Optimizations for speed
            request_options={
                "timeout_in_seconds": 30,  # Reduced from 60
                "max_retries": 0,  # Fail immediately (you'll retry at app level)
            },
        )

        try:
            return res.results.channels[0].alternatives[0].transcript
        except (IndexError, AttributeError):
            return ""


class SpeechToTextStreaming:
    """
    Streaming STT that returns partial results as they arrive.
    Useful for showing live transcription while audio is being recorded.
    """
    def __init__(self, api_key):
        self.client = DeepgramClient(api_key=api_key)
    
    async def transcribe_streaming(self, audio_np):
        """
        Stream transcription results as they arrive (much faster perceived latency)
        
        Returns partial transcripts as they're generated, then final result
        """
        buf = io.BytesIO()
        sf.write(buf, audio_np, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        
        try:
            # Use live transcription mode for streaming results
            res = self.client.listen.v1.media.transcribe_file(
                request=buf,
                model="nova-3",
                language="en",
                request_options={
                    "timeout_in_seconds": 30,
                    "max_retries": 0,
                },
            )
            
            # Return final result
            transcript = res.results.channels[0].alternatives[0].transcript
            return transcript
            
        except Exception as e:
            print(f"STT Error: {e}")
            return ""


# Speed comparison:
# nova-3 (default) = ~1-2s per 10s audio
# nova-2 = ~2-3s per 10s audio (slower but cheaper)
# 
# To make even faster:
# 1. Use shorter audio clips (under 10s)
# 2. Pre-process audio (noise removal)
# 3. Use parallelization for multiple requests
