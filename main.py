from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import numpy as np
import torch
from logger_client import logger
from vad import VoiceActivityDetector
from audio_capture import AudioCapture
from stt_ultraf import SpeechToTextUltraFast as SpeechToText
from llm import LLM

# Try pygame version first (more stable on macOS), fall back to sounddevice
try:
    from tts_pygame import TextToSpeech
    print("Using pygame mixer for TTS (more stable on macOS)")
except Exception as e:
    print(f"Pygame TTS not available: {e}, falling back to sounddevice")
    from tts import TextToSpeech


class SimpleAgent:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.queue = asyncio.Queue()

        # Use higher VAD threshold to reduce false positives from noise
        # 0.5 = default (too sensitive), 0.7 = better for noisy environments
        self.vad = VoiceActivityDetector(sample_rate, threshold=0.7)
        self.tts = TextToSpeech(os.environ["ELEVEN_API_KEY"])
        self.audio = AudioCapture(sample_rate, self.vad, self.queue, self.tts)
        self.stt = SpeechToText(os.environ["DEEPGRAM_API_KEY"])
        self.llm = LLM(os.environ["GOOGLE_API_KEY"])
        
        self.loop = asyncio.get_event_loop()
        self.current_task = None

    async def run(self):
        asyncio.create_task(self.audio.start())

        while True:
            buf = await self.queue.get()
            print(f"Received audio buffer: {len(buf)} chunks")

            # cancel previous processing
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()

            # start new processing without blocking
            self.current_task = asyncio.create_task(self.handle_audio(buf))

    async def handle_audio(self, buf):
        logger.info("Transcribing...")
        if not buf:
            return

        audio = np.concatenate(buf).astype(np.float32)

        # offload STT to executor
        text = await self.loop.run_in_executor(None, self.stt.transcribe, audio)
        print(f"Transcribed: {text}")
        if not text:
            return
        logger.info(f"User: {text}")
        await self.handle_response(text)
    async def handle_response(self, text):
        logger.info("AI Thinking...")

        # Clear VAD buffer to prevent old audio from triggering interrupt
        self.vad.buffer = torch.zeros(0)
        
        # Get LLM response first (run in executor to not block event loop)
        reply = await self.loop.run_in_executor(None, self.llm.ask, text)
        if not reply:
            return

        logger.info(f"AI: {reply}")

        try:
            # Mark that AI is about to speak
            self.audio.ai_is_speaking = True
            
            # Pause mic during ENTIRE TTS playback to prevent speaker feedback
            # We'll handle interrupts differently - reset confidence on each chunk
            self.audio.is_paused = True
            
            # Stream TTS - starts playing as audio arrives
            await self.tts.speak_stream(reply)
        finally:
            # Mark that AI is done speaking
            self.audio.ai_is_speaking = False
            # Ensure mic is resumed
            self.audio.is_paused = False


async def main():
    agent = SimpleAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
