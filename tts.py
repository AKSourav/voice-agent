import asyncio
import numpy as np
import pygame
from elevenlabs import ElevenLabs
from io_devices import speaker_device_name


class TextToSpeech:
    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.sample_rate = 22050
        self.output_device = speaker_device_name

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
        
        self._stop_event = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()
        self.current_channel = None
        self.is_playing = False
        self.is_paused = False

    def interrupt(self):
        self._cancel_event.set()
        self._interrupt_event.set()
        self.stop_audio()

    def user_speech_detected(self):
        self._interrupt_event.set()
        self.interrupt()

    def pause(self):
        if self.current_channel and self.current_channel.get_busy():
            self.current_channel.pause()
            self.is_paused = True

    def resume(self):
        if self.current_channel and self.is_paused:
            self.current_channel.unpause()
            self.is_paused = False

    def stop_audio(self):
        try:
            if self.current_channel:
                self.current_channel.stop()
                self.current_channel = None
            self.is_playing = False
            self.is_paused = False
        except Exception:
            pass

    async def speak_stream(self, text: str):
        if not text or not text.strip():
            return

        self._stop_event.clear()
        self._cancel_event.clear()
        self._interrupt_event.clear()

        stream = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",
        )

        try:
            audio_chunks = []
            min_chunks = 2
            started = False
            
            for chunk in stream:
                if self._cancel_event.is_set() or self._interrupt_event.is_set():
                    return
                
                audio_chunks.append(chunk)
                
                if not started and len(audio_chunks) >= min_chunks:
                    audio_data = b"".join(audio_chunks)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
                    self.is_playing = True
                    self.is_paused = False
                    started = True
                    audio_chunks = []
            
            if audio_chunks:
                audio_data = b"".join(audio_chunks)
                if started:
                    while self.current_channel and self.current_channel.get_busy():
                        if self._cancel_event.is_set() or self._interrupt_event.is_set():
                            self.stop_audio()
                            return
                        await asyncio.sleep(0.05)
                else:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
                    self.is_playing = True
                    self.is_paused = False
                    started = True
                
                if started and audio_data:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
            
            if started:
                while self.current_channel and self.current_channel.get_busy():
                    if self._cancel_event.is_set() or self._interrupt_event.is_set():
                        self.stop_audio()
                        return
                    
                    if self._stop_event.is_set() and not self.is_paused:
                        self.pause()
                    elif not self._stop_event.is_set() and self.is_paused:
                        self.resume()
                    
                    await asyncio.sleep(0.02)
                
                self.is_playing = False

        except Exception:
            self.is_playing = False
        finally:
            self.stop_audio()
