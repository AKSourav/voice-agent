import io
import numpy as np
import soundfile as sf
from deepgram import DeepgramClient


class SpeechToTextUltraFast:
    def __init__(self, api_key):
        self.client = DeepgramClient(api_key=api_key)

    def transcribe(self, audio_np):
        audio_trimmed = self._trim_silence(audio_np)
        
        buf = io.BytesIO()
        sf.write(buf, audio_trimmed, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)

        try:
            res = self.client.listen.v1.media.transcribe_file(
                request=buf,
                model="nova-2",
                language="en",
                request_options={
                    "timeout_in_seconds": 15,
                    "max_retries": 0,
                },
            )

            transcript = res.results.channels[0].alternatives[0].transcript
            return transcript.strip() if transcript else ""
        except Exception:
            return ""

    def _trim_silence(self, audio, threshold=0.02):
        energy = np.abs(audio)
        above_threshold = energy > threshold
        
        indices = np.where(above_threshold)[0]
        if len(indices) == 0:
            return audio
        
        start = max(0, indices[0] - 3200)
        end = min(len(audio), indices[-1] + 3200)
        
        return audio[start:end]
