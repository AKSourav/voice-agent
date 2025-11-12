import asyncio
import numpy as np
import sounddevice as sd
from elevenlabs import ElevenLabs
from io_devices import speaker_device

class TextToSpeech:
    """
    ElevenLabs TTS with built-in streaming (low-latency) playback using the official client.
    """

    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.device = speaker_device
        # audio setup
        self.sample_rate = 22050
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=speaker_device)

        # control flags
        self._stop_event = asyncio.Event()
        self._cancel_event = asyncio.Event()

    # ====== Control Methods ======
    def interrupt(self):
        """Immediately stop and cancel playback."""
        print("Interrupt called")
        self._cancel_event.set()
        self._stop_event.set()
        self.stop_audio()

    def pause(self):
        """Pause playback."""
        print("⏸Pause called")
        self._stop_event.set()

    def resume(self):
        """Resume playback."""
        print("Resume called")
        self._stop_event.clear()

    def stop_audio(self):
        """Completely stop current playback."""
        self.stream.stop()

    # ====== Main Async Streaming ======
    async def speak_stream(self, text: str):
        if not text or not text.strip():
            return

        self._stop_event.clear()
        self._cancel_event.clear()

        # Start audio stream
        self.stream.start()

        print("Starting ElevenLabs TTS stream...")

        # Get async loop reference
        loop = asyncio.get_event_loop()

        # Start ElevenLabs TTS streaming generator
        stream = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",  # raw PCM data for immediate playback
        )

        try:
            for chunk in stream:
                if self._cancel_event.is_set():
                    print("Cancel event detected, stopping playback.")
                    break

                # Wait if paused
                while self._stop_event.is_set() and not self._cancel_event.is_set():
                    print(f"{self._stop_event.is_set()=}")
                    await asyncio.sleep(0.3)

                audio_data = np.frombuffer(chunk, dtype=np.int16)
                self.stream.write(audio_data)

        finally:
            self.stream.stop()
            print("Stream closed cleanly")
