from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import numpy as np
from logger_client import logger
from vad import VoiceActivityDetector
from audio_capture import AudioCapture
from stt import SpeechToText
from llm import LLM
from tts import TextToSpeech


class SimpleAgent:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.queue = asyncio.Queue()

        self.vad = VoiceActivityDetector(sample_rate)
        self.audio = AudioCapture(sample_rate, self.vad, self.queue)
        self.stt = SpeechToText(os.environ["DEEPGRAM_API_KEY"])
        self.llm = LLM(os.environ["GOOGLE_API_KEY"])
        self.tts = TextToSpeech(os.environ["ELEVEN_API_KEY"])
        
        self.loop = asyncio.get_event_loop()
        self.current_task = None

    async def run(self):
        asyncio.create_task(self.audio.start())

        while True:
            buf = await self.queue.get()
            print(f"Anupam{len(buf)=}")
            # stop any current speech playback
            self.tts.pause()

            # cancel previous processing
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()

            # start new processing without blocking
            self.current_task = asyncio.create_task(self.handle_audio(buf))

    async def handle_audio(self, buf):
        logger.info("Trancribing...")
        if not buf:
            return

        audio = np.concatenate(buf).astype(np.float32)

        # offload STT to executor
        text = await self.loop.run_in_executor(None, self.stt.transcribe, audio)
        print(f"{text=}")
        if not text:
            self.tts.resume()
            return
        logger.info(f"User: {text}")
        await self.handle_response(text)
    async def handle_response(self, text):
        logger.info("AI Thinking...")

        # offload LLM call
        reply = await self.loop.run_in_executor(None, self.llm.ask, text)
        if not reply:
            return

        logger.info(f"AI: {reply}")

        # pause microphone while speaking
        # self.audio.is_paused = True

        # offload TTS streaming so event loop isn't blocked
        # await self.loop.run_in_executor(None, self.tts.speak_stream, reply)
        await self.tts.speak_stream(reply)

        self.audio.is_paused = False


async def main():
    agent = SimpleAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
