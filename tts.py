import asyncio
import pygame
from io import BytesIO
from elevenlabs import ElevenLabs

class TextToSpeech:
    def __init__(self, api_key, voice_id="pNInz6obpgDQGcFmaJgB"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id

        # init pygame mixer once
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        self.channel = pygame.mixer.Channel(0)

        # control flags
        self._stop_event = asyncio.Event()
        self._cancel_event = asyncio.Event()

    # ======== Control Methods ========

    def interrupt(self):
        print("interrupt called")
        self._cancel_event.set()
        self._stop_event.set()
        self.stop_audio()

    def pause(self):
        print("pause called")
        self._stop_event.set()
        self.pause_audio()

    def resume(self):
        print("resume called")
        self._stop_event.clear()
        self.resume_audio()

    # ======== Audio Control ========

    def play_audio(self, data: bytes):
        """Play an MP3 buffer using pygame."""
        from pygame import mixer
        from tempfile import NamedTemporaryFile

        # Save mp3 chunk temporarily (pygame needs a file-like object)
        with NamedTemporaryFile(delete=True, suffix=".mp3") as f:
            f.write(data)
            f.flush()
            sound = mixer.Sound(f.name)
            self.channel.play(sound)

            # Block until this small chunk finishes or canceled
            while self.channel.get_busy() and not self._cancel_event.is_set():
                if self._stop_event.is_set():
                    mixer.pause()
                    while self._stop_event.is_set() and not self._cancel_event.is_set():
                        pygame.time.wait(10)
                    mixer.unpause()
                pygame.time.wait(10)

    def pause_audio(self):
        pygame.mixer.pause()

    def resume_audio(self):
        pygame.mixer.unpause()

    def stop_audio(self):
        pygame.mixer.stop()

    # ======== Async Stream Method ========

    async def speak_stream(self, text: str):
        if not text or not text.strip():
            return

        # reset flags
        self._stop_event.clear()
        self._cancel_event.clear()

        loop = asyncio.get_running_loop()

        # get ElevenLabs stream in executor (blocking)
        stream_res = await loop.run_in_executor(
            None,
            lambda: self.client.text_to_speech.stream(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_22050_32",
            ),
        )

        batch = b""
        limit = 80000

        for chunk in stream_res:
            if self._cancel_event.is_set():
                return

            # Wait if paused
            while self._stop_event.is_set() and not self._cancel_event.is_set():
                await asyncio.sleep(0.01)

            batch += chunk

            if len(batch) > limit:
                await loop.run_in_executor(None, self.play_audio, batch)
                batch = b""

        if batch and not self._cancel_event.is_set():
            await loop.run_in_executor(None, self.play_audio, batch)
