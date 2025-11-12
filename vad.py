import torch

silero_model, _ = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    force_reload=False,
)


class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, threshold=0.5, chunk_size=512):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.buffer = torch.zeros(0)

    def add_audio(self, samples):
        self.buffer = torch.cat((self.buffer, samples))

    def get_speech_chunks(self):
        chunks = []
        while self.buffer.numel() >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            prob = silero_model(chunk, self.sample_rate).item()
            chunks.append((prob, chunk))
        return chunks
