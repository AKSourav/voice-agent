import asyncio
import numpy as np
import pygame
import io
from elevenlabs import ElevenLabs


class TextToSpeech:
    """
    ElevenLabs TTS with pygame mixer playback (more stable on macOS than sounddevice).
    """

    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.sample_rate = 22050
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=4096)
        
        # control flags
        self._stop_event = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()  # Signal user speech detected
        self.current_channel = None
        self.is_playing = False
        self.is_paused = False

    # ====== Control Methods ======
    def interrupt(self):
        """Immediately stop and cancel playback."""
        print("Interrupt called")
        self._cancel_event.set()
        self._interrupt_event.set()
        self.stop_audio()

    def user_speech_detected(self):
        """Called when user speech is detected during AI playback."""
        print("🛑 User speech detected, interrupting AI...")
        self._interrupt_event.set()
        self.interrupt()

    def pause(self):
        """Pause playback."""
        print("⏸Pause called")
        if self.current_channel and self.current_channel.get_busy():
            self.current_channel.pause()
            self.is_paused = True
            print("  → Paused successfully")

    def resume(self):
        """Resume playback."""
        print("Resume called")
        if self.current_channel and self.is_paused:
            self.current_channel.unpause()
            self.is_paused = False
            print("  → Resumed successfully")
        elif not self.current_channel:
            print("  → No active channel to resume")
        elif not self.is_paused:
            print("  → Not paused, nothing to resume")

    def stop_audio(self):
        """Completely stop current playback."""
        try:
            if self.current_channel:
                self.current_channel.stop()
                self.current_channel = None
            self.is_playing = False
            self.is_paused = False
        except Exception as e:
            print(f"Error stopping audio: {e}")

    # ====== Main Async Streaming ======
    async def speak_stream(self, text: str):
        if not text or not text.strip():
            print("Empty text, skipping TTS")
            return

        self._stop_event.clear()
        self._cancel_event.clear()
        self._interrupt_event.clear()

        print("Starting ElevenLabs TTS stream...")

        # Start ElevenLabs TTS streaming generator
        stream = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",  # raw PCM data for immediate playback
        )

        try:
            # Buffer audio chunks and start playback early
            audio_chunks = []
            min_chunks = 5  # Start after collecting a few chunks (~500ms)
            started = False
            
            for chunk in stream:
                if self._cancel_event.is_set() or self._interrupt_event.is_set():
                    print("Playback interrupted/cancelled during download.")
                    return
                
                audio_chunks.append(chunk)
                
                # Start playback as soon as we have enough chunks (early start = low latency)
                if not started and len(audio_chunks) >= min_chunks:
                    audio_data = b"".join(audio_chunks)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
                    self.is_playing = True
                    self.is_paused = False
                    started = True
                    print(f"🎵 Started playback with {len(audio_data)} bytes ({len(audio_chunks)} chunks)")
                    audio_chunks = []  # Clear for next batch
            
            # If we have remaining audio after stream ends, play it
            if audio_chunks:
                audio_data = b"".join(audio_chunks)
                if started:
                    # Wait for current playback, then play remaining
                    while self.current_channel and self.current_channel.get_busy():
                        if self._cancel_event.is_set() or self._interrupt_event.is_set():
                            self.stop_audio()
                            return
                        await asyncio.sleep(0.05)
                else:
                    # Never started - very short text, play directly
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
                    self.is_playing = True
                    self.is_paused = False
                    started = True
                    print(f"🎵 Final playback with {len(audio_data)} bytes")
                
                # Play remaining audio
                if started and audio_data:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(audio_array)
                    self.current_channel = sound.play()
            
            # Wait for final playback to finish
            if started:
                while self.current_channel and self.current_channel.get_busy():
                    # Check for interrupt/cancel
                    if self._cancel_event.is_set() or self._interrupt_event.is_set():
                        print("Playback interrupted during playback.")
                        self.stop_audio()
                        return
                    
                    # Check pause flag
                    if self._stop_event.is_set() and not self.is_paused:
                        print("Pause event detected, pausing...")
                        self.pause()
                    elif not self._stop_event.is_set() and self.is_paused:
                        print("Pause event cleared, resuming...")
                        self.resume()
                    
                    await asyncio.sleep(0.05)
                
                self.is_playing = False
                print("✓ Playback finished normally")

        except Exception as e:
            print(f"Stream error: {e}")
            import traceback
            traceback.print_exc()
            self.is_playing = False
        finally:
            self.stop_audio()
            print("Stream closed cleanly")
