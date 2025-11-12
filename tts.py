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
        self.stream = None
        self.stream_active = False

        # control flags
        self._stop_event = asyncio.Event()
        self._cancel_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()  # Signal user speech detected

    # ====== Control Methods ======
    def interrupt(self):
        """Immediately stop and cancel playback."""
        print("Interrupt called")
        self._cancel_event.set()
        self._stop_event.set()
        self.stop_audio()

    def user_speech_detected(self):
        """Called when user speech is detected during AI playback."""
        print("🛑 User speech detected, interrupting AI...")
        self._interrupt_event.set()
        self.interrupt()

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
        self.stream_active = False  # Signal to stop writing
        if self.stream:
            try:
                self.stream.stop()
            except:
                pass
            try:
                self.stream.close()
            except:
                pass
            self.stream = None

    # ====== Main Async Streaming ======
    async def speak_stream(self, text: str):
        if not text or not text.strip():
            return

        self._stop_event.clear()
        self._cancel_event.clear()
        self._interrupt_event.clear()

        # Create fresh stream for this speech
        try:
            self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=self.device)
            self.stream.start()
            self.stream_active = True
        except Exception as e:
            print(f"Error starting stream: {e}")
            return

        print("Starting ElevenLabs TTS stream...")

        # Start ElevenLabs TTS streaming generator
        stream = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_22050",  # raw PCM data for immediate playback
        )

        try:
            for chunk in stream:
                # Check interrupt flags BEFORE trying to write
                if self._cancel_event.is_set() or self._interrupt_event.is_set():
                    print("Playback interrupted/cancelled.")
                    break

                # If stream was stopped externally, exit
                if not self.stream_active:
                    break

                # Wait if paused
                while self._stop_event.is_set() and not self._cancel_event.is_set() and not self._interrupt_event.is_set():
                    await asyncio.sleep(0.1)

                # Check again after pause in case interrupted
                if self._cancel_event.is_set() or self._interrupt_event.is_set() or not self.stream_active:
                    break

                # Only write if stream is still active
                if self.stream_active and self.stream:
                    try:
                        audio_data = np.frombuffer(chunk, dtype=np.int16)
                        self.stream.write(audio_data)
                    except Exception as e:
                        print(f"Error writing audio: {e}")
                        self.stream_active = False
                        break

        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            self.stop_audio()
            print("Stream closed cleanly")
