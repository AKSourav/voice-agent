import io
import numpy as np
import soundfile as sf
from deepgram import DeepgramClient


class SpeechToText:
    def __init__(self, api_key):
        self.client = DeepgramClient()

    def transcribe(self, audio_np):
        buf = io.BytesIO()
        sf.write(buf, audio_np, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)

        res = self.client.listen.v1.media.transcribe_file(
            request=buf,
            model="nova-3",
            request_options={"timeout_in_seconds": 60, "max_retries": 3},
        )

        return res.results.channels[0].alternatives[0].transcript
