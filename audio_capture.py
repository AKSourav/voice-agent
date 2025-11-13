import asyncio
import numpy as np
import torch
import time
import sounddevice as sd
from logger_client import logger
from tts import TextToSpeech
from io_devices import mic_device


class AudioCapture:
    def __init__(self, sample_rate, vad, queue, tts: TextToSpeech):
        self.sample_rate = sample_rate
        self.vad = vad
        self.queue = queue
        self.audio_buffer = []
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.loop = asyncio.get_event_loop()
        self.tts = tts
        self.device = mic_device
        self.ai_is_speaking = False

    def notify_listening(self):
        logger.info("Listening...")

    async def start(self):
        device = self.device

        def callback(indata, frames, t, status):
            if self.ai_is_speaking:
                self.audio_buffer = []
                return
            
            samples = torch.tensor(indata.flatten(), dtype=torch.float32)
            self.vad.add_audio(samples)

            for prob, chunk_data in self.vad.get_speech_chunks():
                speech_detected = prob > self.vad.threshold
                
                if speech_detected:
                    if not self.is_speaking:
                        self.notify_listening()
                    self.is_speaking = True
                    self.audio_buffer.append(chunk_data.numpy())
                    self.last_speech_time = time.time()
                else:
                    if self.is_speaking and time.time() - self.last_speech_time > 0.3:
                        self.is_speaking = False
                        buf = self.audio_buffer.copy()
                        self.audio_buffer = []
                        asyncio.run_coroutine_threadsafe(self.queue.put(buf), self.loop)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=self.vad.chunk_size,
            device=device
        )

        with stream:
            while True:
                await asyncio.sleep(0.03)