import asyncio
from asyncio.subprocess import DEVNULL
import os
import numpy as np
import pygame
from elevenlabs import ElevenLabs
from io_devices import speaker_device_name


class TextToSpeech:
    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.backend = os.getenv("TTS_BACKEND", "elevenlabs").lower()
        self.client = ElevenLabs(api_key=api_key) if self.backend == "elevenlabs" else None
        self.voice_id = voice_id
        self.sample_rate = 22050
        self.output_device = speaker_device_name

        if self.backend == "elevenlabs":
            if pygame.mixer.get_init():
                pygame.mixer.quit()

            try:
                pygame.mixer.init(
                    frequency=self.sample_rate,
                    size=-16,
                    channels=1,
                    buffer=4096,
                    devicename=self.output_device,
                )
            except pygame.error:
                pygame.mixer.init(
                    frequency=self.sample_rate,
                    size=-16,
                    channels=1,
                    buffer=4096,
                )
        
        self.current_channel = None
        self.is_playing = False

    async def speak_stream(self, text: str):
        if not text or not text.strip():
            return

        if self.backend == "macos":
            voice = os.getenv("MACOS_TTS_VOICE", "Samantha")
            rate = os.getenv("MACOS_TTS_RATE", "220")
            try:
                self.is_playing = True
                proc = await asyncio.create_subprocess_exec(
                    "say",
                    "-v", voice,
                    "-r", str(rate),
                    text,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )
                await proc.wait()
                self.is_playing = False
            except Exception:
                self.is_playing = False
            return

        stream = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",
        )

        audio_buffer = bytearray()

        for chunk in stream:
            audio_buffer.extend(chunk)

        if not audio_buffer:
            return

        audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
        sound = pygame.sndarray.make_sound(audio_array)
        self.current_channel = sound.play()
        self.is_playing = True

        try:
            while self.current_channel and self.current_channel.get_busy():
                await asyncio.sleep(0.02)
        finally:
            if self.current_channel:
                try:
                    self.current_channel.stop()
                except Exception:
                    pass
                self.current_channel = None
            self.is_playing = False
