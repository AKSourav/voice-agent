import numpy as np
import whisper
from logger_client import logger


class SpeechToTextUltraFast:
    def __init__(self, api_key=None):
        try:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model("tiny")
            logger.info("Whisper model loaded successfully")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.initialized = False
            raise

    def transcribe(self, audio_np):
        if not self.initialized:
            return ""
        
        if not self._is_valid_audio(audio_np):
            return ""
        
        try:
            audio_trimmed = self._trim_silence(audio_np)
            
            result = self.model.transcribe(
                audio_trimmed,
                language="en",
                fp16=False,
                verbose=False,
            )
            
            transcript = result.get("text", "").strip()
            
            if transcript:
                logger.info(f"Transcribed: {transcript}")
            
            return transcript
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _is_valid_audio(self, audio_np):
        if audio_np is None or len(audio_np) == 0:
            return False
        return len(audio_np) >= 8000

    def _trim_silence(self, audio, threshold=0.01):
        if len(audio) == 0:
            return audio
        
        energy = np.abs(audio)
        above_threshold = energy > threshold
        
        indices = np.where(above_threshold)[0]
        if len(indices) == 0:
            return audio
        
        margin_samples = int(16000 * 0.2)
        start = max(0, indices[0] - margin_samples)
        end = min(len(audio), indices[-1] + margin_samples)
        
        return audio[start:end]
