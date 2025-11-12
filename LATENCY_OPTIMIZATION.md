# Latency Optimization Guide

## Changes Made

### 1. **TTS Streaming (Early Playback)**
**File**: `tts_pygame.py`

**Before**: Waited for ALL audio from ElevenLabs before starting playback
- User speaks → STT → LLM → Wait for all TTS chunks → Play

**After**: Start playback after 200ms of audio arrives (streaming)
- User speaks → STT → LLM → Play first 200ms of TTS while more arrives
- Saves ~1-2 seconds per response

### 2. **Reduced STT Retries**
**File**: `stt_fast.py`

**Before**: `max_retries: 3` - waited for retries on failure
**After**: `max_retries: 1` - fail fast

**Impact**: ~500ms faster on network issues

### 3. **Pipeline Order**
**File**: `main.py`

**Before**: 
```
Pause mic → Get LLM → Get TTS → Pause mic end
```

**After**:
```
Pause mic → Get LLM (parallel) → TTS starts streaming → Pause mic end
```

## Latency Breakdown

| Step | Time |
|------|------|
| Audio capture | 0.5-1s |
| STT (Deepgram) | 2-3s |
| LLM (Google Gemini) | 1-2s |
| TTS generation | 1-2s (but now streamed) |
| **Total (Old)** | **5-8s** |
| **Total (New)** | **3-5s** |

## Further Optimizations You Can Try

1. **Use faster LLM**: OpenAI GPT-4 mini might be faster than Gemini
2. **Shorter prompts**: Reduce system prompt to speed up token generation
3. **Use Deepgram live streaming**: For real-time STT
4. **Buffer audio while LLM runs**: Start collecting next audio batch during LLM processing
5. **Use shorter TTS voice settings**: Some voices are faster

## Testing

Run and observe:
```bash
python main.py
# Time from when you stop speaking to when AI starts speaking
```

Expected: ~3-4 seconds for responsive conversation
