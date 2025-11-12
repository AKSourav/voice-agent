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
        self.is_paused = False
        self.last_speech_time = time.time()
        self.loop = asyncio.get_event_loop()
        self.tts = tts
        self.device = mic_device
        self.ai_is_speaking = False  # Track if AI is currently speaking
        self.speech_confirmed = False  # Only interrupt if we confirm sustained speech

    def notify_listening(self):
        logger.info("Listening...")

    async def start(self):
        chunk = self.vad.chunk_size
        device = self.device
        speech_confidence = 0  # Track consecutive speech detections
        paused_speech_buffer = []  # Buffer speech detected while paused
        sustained_pause_confidence = 0  # Track sustained detections during pause

        def callback(indata, frames, t, status):
            nonlocal speech_confidence, paused_speech_buffer, sustained_pause_confidence
            
            samples = torch.tensor(indata.flatten(), dtype=torch.float32)
            self.vad.add_audio(samples)

            for prob, chunk_data in self.vad.get_speech_chunks():
                # Higher threshold for interrupt detection (0.7) vs normal speech (0.5)
                speech_detected = prob > self.vad.threshold
                
                # If paused (AI speaking), buffer detected speech for later analysis
                if self.is_paused and speech_detected:
                    paused_speech_buffer.append((prob, chunk_data))
                    
                    # Track sustained high-confidence speech
                    if prob > 0.90:
                        sustained_pause_confidence += 1
                    else:
                        sustained_pause_confidence = 0
                    
                    # Only interrupt if we get sustained high-confidence speech (3+ chunks)
                    # This prevents interruption from single artifacts/clicks
                    if sustained_pause_confidence >= 3:  
                        logger.info(f"⚠️  Sustained strong speech detected during pause (confidence: {prob:.3f})")
                        self.tts.user_speech_detected()
                        self.audio_buffer = []
                        self.is_speaking = False
                        paused_speech_buffer = []
                        speech_confidence = 0
                        sustained_pause_confidence = 0
                        return
                else:
                    sustained_pause_confidence = 0
                
                # Reset confidence when mic is unpaused
                if not self.is_paused and paused_speech_buffer:
                    paused_speech_buffer = []
                    speech_confidence = 0
                
                if speech_detected:
                    speech_confidence += 1
                else:
                    speech_confidence = 0
                
                # Only process user speech if not paused (avoid feedback from speaker)
                if speech_detected:
                    if not self.is_paused:
                        if self.is_speaking == False: 
                            self.notify_listening()
                        self.is_speaking = True
                        self.audio_buffer.append(chunk_data.numpy())
                        self.last_speech_time = time.time()
                else:
                    if self.is_speaking and time.time() - self.last_speech_time > 0.7:
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
