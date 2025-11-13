from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import numpy as np
import torch
from logger_client import logger
from vad import VoiceActivityDetector
from audio_capture import AudioCapture
from stt import SpeechToTextUltraFast as SpeechToText
from llm import LLM
from tts import TextToSpeech


class Agent:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.queue = asyncio.Queue()
        self.vad = VoiceActivityDetector(sample_rate, threshold=0.7)
        self.tts = TextToSpeech(os.environ["ELEVEN_API_KEY"])
        self.audio = AudioCapture(sample_rate, self.vad, self.queue, self.tts)
        
        # Use local Whisper STT (ultra-fast, no API needed)
        try:
            self.stt = SpeechToText()
            logger.info("Using Whisper STT")
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")
            raise
        
        self.llm = LLM(os.environ["GOOGLE_API_KEY"])
        self.loop = asyncio.get_event_loop()
        self.current_task = None

    async def run(self):
        asyncio.create_task(self.audio.start())
        while True:
            buf = await self.queue.get()
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
            self.current_task = asyncio.create_task(self.handle_audio(buf))

    async def handle_audio(self, buf):
        logger.info("Transcribing...")
        if not buf:
            return

        try:
            audio = np.concatenate(buf).astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
            
            text = await self.loop.run_in_executor(None, self.stt.transcribe, audio)
            
            if not text or not text.strip():
                return
            
            logger.info(f"User: {text}")
            await self.handle_response(text)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    async def handle_response(self, text):
        logger.info("AI Thinking...")
        self.vad.buffer = torch.zeros(0)
        
        reply = await self.loop.run_in_executor(None, self.llm.ask, text)
        if not reply:
            return

        logger.info(f"AI: {reply}")

        try:
            self.audio.ai_is_speaking = True
            self.audio.is_paused = True
            await self.tts.speak_stream(reply)
        finally:
            self.audio.ai_is_speaking = False
            self.audio.is_paused = False


async def main():
    agent = Agent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
