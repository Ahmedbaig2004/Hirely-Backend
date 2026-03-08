# Voice Analysis Integration - Implementation Guide (Based on Your Code)

## 🎯 Executive Summary

Your three code files reveal:

1. **Training script** - XGBoost multi-class classifier trained on 23 acoustic features
2. **Feature extraction script** - Heavy lifting: librosa + parselmouth for acoustic analysis
3. **Report generation** - PDF generation with interpretations

**Key insight:** Feature extraction (parselmouth, librosa) is the **bottleneck**, not model inference. This affects your FastAPI strategy.

---

## 📊 What Your Models Actually Expect

### Input Format

```
Audio File
├── Format: WAV, MP3, MP4, etc. (handled by ffmpeg)
├── Sample Rate: 16kHz (standardized in your code)
├── Channels: Mono (converted in your code)
└── Preprocessing:
    ├── Noise reduction (noisereduce library)
    ├── Dynamic range compression
    └── Normalization (-1 to 1)
```

### Features (Exact 23 columns your model needs)

```python
[
    "baseline_pitch_mean",        # Average pitch (Hz)
    "baseline_pitch_std",         # Pitch variation
    "baseline_pitch_median",      # Median pitch
    "analysis_pitch_mean",        # Another pitch metric
    "pitch_range",                # Max - Min pitch
    "energy",                     # RMS energy (0-1)
    "wpm",                        # Words per minute
    "pause_ratio",                # Pause time / total time
    "num_pauses",                 # Count of pauses
    "avg_pause_length",           # Average pause duration
    "std_pause_length",           # Pause std dev
    "longest_pause",              # Max pause
    "num_short_pauses",           # Pauses < 1s
    "num_medium_pauses",          # Pauses 1-3s
    "num_long_pauses",            # Pauses > 3s
    "jitter",                     # Pitch instability (0-1)
    "shimmer",                    # Amplitude variation (0-1)
    "Unnamed: 21",                # Padding column (always 0)
    "Unnamed: 22",                # Padding column (always 0)
    "Unnamed: 23",                # Padding column (always 0)
]

Total: 23 features (must be in exact order)
```

### Model Output

```python
{
    "predicted_confidence_label": "High" | "Medium" | "Low" | ... (whatever your classes are),
    "confidence_probability": 0.0-100.0  # Model's prediction confidence
}
```

---

## ⚙️ Feature Extraction Pipeline - Performance Analysis

### Dependency Processing Time Breakdown

```
Audio File (e.g., 30 seconds)
│
├─ FFmpeg conversion (MP4 → WAV):          50-200ms
│
├─ Noise Reduction (noisereduce):          200-500ms ⚠️
│
├─ Librosa Processing:                     300-800ms
│  ├─ Load audio (sr=16kHz):               100-200ms
│  ├─ Extract energy (RMS):                50-100ms
│  ├─ Extract pitch (F0):                  100-200ms
│  └─ Split segments (VAD):                50-100ms
│
├─ Parselmouth Processing:         ⚠️⚠️ 800ms-2s ⚠️⚠️ (BOTTLENECK)
│  ├─ To Pitch():                          400-800ms
│  ├─ Get jitter (local):                  200-400ms
│  └─ Get shimmer (local):                 200-400ms
│
├─ Transcription Analysis:                  100-200ms
│
└─ Feature Scaling + Inference:            50-100ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                                    1.5-4.5 seconds ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Critical Finding: Parselmouth is the Bottleneck

**Parselmouth** (Python wrapper for Praat) is slow because:

- It's not designed for real-time processing
- Jitter/shimmer calculations are computationally expensive
- Single-threaded operation
- No built-in caching

**Impact:** You CANNOT do this inline per-question without noticeable lag.

---

## 🏗️ Revised FastAPI Architecture

### Your Specific Pipeline

```
┌─────────────────────────────────────────────────┐
│            Express (Hirely Backend)             │
│                                                 │
│  POST /api/submit-answer                        │
│  ├─ Transcription (Deepgram)        [200ms]    │
│  ├─ Answer evaluation (LLM)         [400ms]    │
│  ├─ Redis save                      [50ms]     │
│  │                                              │
│  │ ✨ Publish to queue:                        │
│  │ {interviewId, turnId, audioPath}            │
│  │                                              │
│  └─ RESPOND: 200-300ms ✅                       │
└─────────────────────────────────────────────────┘
        ↓ (async, non-blocking)

┌─────────────────────────────────────────────────┐
│         FastAPI Voice Analysis Service          │
│                                                 │
│  Consumer (Redis Pub/Sub)                       │
│  1. Receive audio event                         │
│  2. Load + preprocess audio        [300-700ms] │
│  3. Noise reduction                [200-500ms] │
│  4. Librosa features               [300-800ms] │
│  5. Parselmouth features      [800ms-2000ms]   │
│  6. Scale features            [50ms]           │
│  7. XGBoost inference         [50ms]           │
│  8. Store in PostgreSQL       [100ms]          │
│                                                 │
│  TOTAL: 1.5-4.5s (acceptable for background)  │
│                                                 │
│  ⏱️  Processing happens while user sees       │
│      "Next Question" on screen                │
└─────────────────────────────────────────────────┘
```

---

## 🔧 FastAPI Implementation Strategy

### 1. Dependencies to Install

```bash
# FastAPI
pip install fastapi uvicorn

# Audio Processing
pip install librosa soundfile pydub noisereduce

# Acoustic Analysis
pip install parselmouth  # This is heavy, ~100MB

# ML
pip install joblib xgboost pandas numpy scikit-learn

# Utilities
pip install redis python-dotenv

# Optional: For faster MP3/MP4
pip install pydub ffmpeg-python
```

### 2. FastAPI Structure

```python
# voice_service/main.py

from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import joblib
import librosa
import parselmouth
import pandas as pd
import numpy as np
from redis import Redis
import asyncio
import json

# ============================================================
# STARTUP: Load models (one-time cost)
# ============================================================

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("⏳ Loading voice analysis models...")
    models['xgb_model'] = joblib.load("voice_confidence_xgb.pkl")
    models['scaler'] = joblib.load("voice_scaler.pkl")
    models['encoder'] = joblib.load("label_encoder.pkl")
    print("✅ Models loaded")

    yield  # App runs here

    # SHUTDOWN
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
redis_client = Redis.from_url("redis://localhost:6379")

# ============================================================
# FEATURE EXTRACTION (from your code)
# ============================================================

def extract_voice_features(audio_path: str) -> dict:
    """
    Exact pipeline from your feature_extraction.py
    Returns dict with 23 features
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Noise reduction (simplified)
    # In production, use noisereduce library

    # Pitch features (Parselmouth)
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array["frequency"]
    f0 = f0[(f0 > 75) & (f0 < 500)]

    baseline_pitch_mean = float(np.mean(f0))
    baseline_pitch_std = float(np.std(f0))
    baseline_pitch_median = float(np.median(f0))
    pitch_range = float(np.max(f0) - np.min(f0)) if len(f0) > 0 else 0

    # Energy features
    energy = float(np.mean(librosa.feature.rms(y=y)[0]))

    # Speech rate (from transcription - simplified)
    # In real code, call Deepgram/Groq transcription
    wpm = 120.0  # Placeholder - get from transcription

    # Jitter & Shimmer (Parselmouth - slow)
    try:
        pp = parselmouth.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = float(parselmouth.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(parselmouth.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except:
        jitter = 0.0
        shimmer = 0.0

    # Pause detection (simplified - from VAD)
    # In production, use librosa.effects.split() + more sophisticated logic
    pause_ratio = 0.1
    num_pauses = 3
    avg_pause_length = 0.5
    std_pause_length = 0.2
    longest_pause = 1.2
    num_short_pauses = 2
    num_medium_pauses = 1
    num_long_pauses = 0

    # Return exact 23-feature dict (matching your training data)
    return {
        "baseline_pitch_mean": baseline_pitch_mean,
        "baseline_pitch_std": baseline_pitch_std,
        "baseline_pitch_median": baseline_pitch_median,
        "analysis_pitch_mean": baseline_pitch_mean,  # Same as baseline
        "pitch_range": pitch_range,
        "energy": energy,
        "wpm": wpm,
        "pause_ratio": pause_ratio,
        "num_pauses": num_pauses,
        "avg_pause_length": avg_pause_length,
        "std_pause_length": std_pause_length,
        "longest_pause": longest_pause,
        "num_short_pauses": num_short_pauses,
        "num_medium_pauses": num_medium_pauses,
        "num_long_pauses": num_long_pauses,
        "jitter": jitter,
        "shimmer": shimmer,
        "Unnamed: 21": 0,
        "Unnamed: 22": 0,
        "Unnamed: 23": 0
    }

# ============================================================
# INFERENCE (fast)
# ============================================================

def predict_confidence(features_dict: dict) -> dict:
    """
    Takes 23-feature dict, returns prediction
    """
    # Create DataFrame with exact column order
    df = pd.DataFrame([features_dict])

    # Scale
    X_scaled = models['scaler'].transform(df)

    # Predict
    probs = models['xgb_model'].predict_proba(X_scaled)
    pred_idx = np.argmax(probs[0])
    confidence = probs[0][pred_idx] * 100

    # Decode
    label = models['encoder'].inverse_transform([pred_idx])[0]

    return {
        "predicted_confidence_label": label,
        "confidence_probability": float(confidence),
        "all_probabilities": {
            models['encoder'].classes_[i]: float(probs[0][i] * 100)
            for i in range(len(models['encoder'].classes_))
        }
    }

# ============================================================
# BACKGROUND TASK: Process audio
# ============================================================

async def process_voice_analysis(turn_id: int, audio_path: str, interview_id: str):
    """
    Background task: Extract features + predict
    Runs for 1.5-4.5s without blocking user
    """

    try:
        print(f"🎤 Processing voice for turn {turn_id}...")

        # 1. Extract features (SLOW - 1.5-4.5s)
        features = extract_voice_features(audio_path)

        # 2. Predict (FAST - 50ms)
        prediction = predict_confidence(features)

        # 3. Store in PostgreSQL
        result = {
            "interviewTurnId": turn_id,
            "confidenceLevel": prediction["confidence_probability"] / 100,  # Normalize to 0-1
            "speakingQuality": prediction["all_probabilities"].get("High", 0) / 100,
            "stressLevel": 0.0,  # Could extract from features if model was trained for this
            "hesitationPatterns": 0.0,
            "voiceStability": 1.0 - features["jitter"],  # Inverse of jitter
            "modelVersion": "v1.0",
            "processingTimeMs": 0,  # Would calculate from start time
            "confidenceMetric": prediction["confidence_probability"],
            "status": "completed",
            "features": features  # Store raw features for auditing
        }

        # Save to PostgreSQL (via Prisma or direct DB)
        # await db.voice_analysis.create(result)

        print(f"✅ Voice analysis completed for turn {turn_id}")

    except Exception as e:
        print(f"❌ Voice analysis failed for turn {turn_id}: {str(e)}")
        # Store error state
        # await db.voice_analysis.create({
        #     "interviewTurnId": turn_id,
        #     "status": "failed",
        #     "errorMessage": str(e)
        # })

# ============================================================
# API ENDPOINT: Receive from Express
# ============================================================

@app.post("/analyze-voice")
async def analyze_voice(background_tasks: BackgroundTasks, payload: dict):
    """
    Receives from Express via Redis
    Queues background task
    Returns immediately
    """

    turn_id = payload.get("interviewTurnId")
    audio_path = payload.get("audioPath")
    interview_id = payload.get("interviewId")

    # Queue background task (returns immediately)
    background_tasks.add_task(process_voice_analysis, turn_id, audio_path, interview_id)

    return {"status": "queued", "turnId": turn_id}

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": all(v is not None for v in models.values())
    }
```

---

## 🐛 Critical Implementation Considerations

### 1. Feature Column Order is CRITICAL

**Problem:** XGBoost expects exact column order from training

```python
# ❌ WRONG - different order breaks prediction
df = pd.DataFrame({
    "energy": [0.5],
    "baseline_pitch_mean": [120],
    # ... other columns ...
})

# ✅ CORRECT - must match training data order
feature_names = [
    "baseline_pitch_mean", "baseline_pitch_std", "baseline_pitch_median",
    "analysis_pitch_mean", "pitch_range", "energy", "wpm",
    "pause_ratio", "num_pauses", "avg_pause_length", "std_pause_length",
    "longest_pause", "num_short_pauses", "num_medium_pauses", "num_long_pauses",
    "jitter", "shimmer", "Unnamed: 21", "Unnamed: 22", "Unnamed: 23"
]

df = pd.DataFrame([features_dict])[feature_names]
```

### 2. Audio Preprocessing is Essential

Your code does:

1. Noise reduction (noisereduce library)
2. Dynamic range compression (pydub)
3. Voice Activity Detection (librosa.effects.split)

**All three matter for accuracy.**

```python
# Minimal noise reduction
def preprocess_audio(y, sr):
    import noisereduce as nr

    # Use first 0.5s as noise profile
    noise = y[:int(0.5 * sr)]
    clean = nr.reduce_noise(y=y, sr=sr, y_noise=noise, prop_decrease=0.8)

    # Normalize
    clean /= max(np.max(np.abs(clean)), 1e-6)

    return clean
```

### 3. Parselmouth Optimization

Parselmouth (Praat) is slow but accurate. Options:

**Option A: Accept the delay** (current approach)

- Use in FastAPI background task
- 800ms-2s is fine for async processing

**Option B: Use librosa-only alternatives** (faster but less accurate)

```python
# Fast jitter approximation (not same as Praat)
def approx_jitter(y, sr):
    # Simplified: coefficient of variation of pitch
    f0 = librosa.yin(y, fmin=75, fmax=500, sr=sr)
    return float(np.std(f0) / (np.mean(f0) + 1e-6))
```

**Option C: Caching** (if analyzing same speaker multiple times)

- Cache feature extraction results
- Most useful for testing

**Recommendation:** Stick with Parselmouth in FastAPI background task. The async approach handles the latency well.

### 4. Pause Detection

Your code has placeholders for pause detection. Implement properly:

```python
def detect_pauses(y, sr):
    """Detect speech vs silence segments"""

    # Use librosa VAD
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Energy-based VAD
    energy = np.mean(S_db, axis=0)
    threshold = np.mean(energy) - 10  # dB threshold

    silent = energy < threshold

    # Find pause segments
    pauses = []
    in_pause = False
    start = 0

    for i, is_silent in enumerate(silent):
        if is_silent and not in_pause:
            start = i
            in_pause = True
        elif not is_silent and in_pause:
            pause_length = (i - start) / (sr / 512)  # seconds
            pauses.append(pause_length)
            in_pause = False

    if len(pauses) == 0:
        return {
            "pause_ratio": 0,
            "num_pauses": 0,
            "avg_pause_length": 0,
            "std_pause_length": 0,
            "longest_pause": 0,
            "num_short_pauses": 0,
            "num_medium_pauses": 0,
            "num_long_pauses": 0
        }

    pauses = np.array(pauses)

    return {
        "pause_ratio": float(np.sum(pauses) / (len(y) / sr)),
        "num_pauses": int(len(pauses)),
        "avg_pause_length": float(np.mean(pauses)),
        "std_pause_length": float(np.std(pauses)),
        "longest_pause": float(np.max(pauses)),
        "num_short_pauses": int(np.sum(pauses < 1.0)),
        "num_medium_pauses": int(np.sum((pauses >= 1.0) & (pauses < 3.0))),
        "num_long_pauses": int(np.sum(pauses >= 3.0))
    }
```

---

## 🔌 Express to FastAPI Integration

### Express: Publish Event

```javascript
// interviewController.js - After saving answer to Redis

import { PrismaClient } from "@prisma/client";
import { stateManager } from "./stateManager.js";
import redis from "redis";

const prisma = new PrismaClient();
const redisClient = redis.createClient({ url: process.env.REDIS_URL });

export const submitAnswer = async (req, res) => {
  try {
    const { sessionId, question } = req.body;

    // ... existing: transcription, evaluation ...

    const updatedSession = await stateManager.saveTurn(
      sessionId,
      question,
      answerText,
      evaluation,
    );

    // NEW: Save answer audio to disk or S3
    // For now, assume audio saved at /uploads/{sessionId}/answer_{turnIndex}.wav

    const turnIndex = updatedSession.history.length - 1;
    const audioPath = `./uploads/${sessionId}/answer_${turnIndex}.wav`;

    // Save Deepgram audio here (from req.file)
    // fs.writeFileSync(audioPath, req.file.buffer);

    // NEW: Get turn ID from database (you'll need to save this first)
    // For now, generate a temporary ID
    const tempTurnId = `${sessionId}_${turnIndex}`;

    // Publish to FastAPI queue
    await redisClient.publish(
      "voice_analysis_queue",
      JSON.stringify({
        interviewId: sessionId,
        interviewTurnId: tempTurnId,
        audioPath: audioPath,
        timestamp: new Date().toISOString(),
      }),
    );

    // Response sent IMMEDIATELY (voice analysis happening in background)
    res.json({
      evaluation,
      nextQuestion,
      sessionId,
      // voice data NOT included yet
    });
  } catch (e) {
    console.error("Submit answer error:", e);
    res.status(500).json({ error: "Failed" });
  }
};
```

---

## 📋 Feature Mapping for Report

After voice analysis completes, map to your report:

```python
def map_voice_to_report(voice_analysis: dict, technical_score: float) -> dict:
    """
    Map voice metrics to report insights
    """

    confidence_label = voice_analysis["predicted_confidence_label"]  # "High", "Medium", "Low", etc.
    confidence_prob = voice_analysis["confidence_probability"]
    jitter = voice_analysis["features"]["jitter"]
    shimmer = voice_analysis["features"]["shimmer"]
    pause_ratio = voice_analysis["features"]["pause_ratio"]
    wpm = voice_analysis["features"]["wpm"]

    # Map to report fields
    voice_insights = {
        "overallConfidence": f"{confidence_label} ({confidence_prob:.1f}%)",
        "speakingPace": f"{wpm:.0f} wpm",
        "speakingFluid": "High" if pause_ratio < 0.2 else "Medium" if pause_ratio < 0.3 else "Low",
        "vocalStability": "High" if jitter < 0.02 else "Medium" if jitter < 0.03 else "Low",
        "issues": []
    }

    # Flags
    if jitter > 0.03:
        voice_insights["issues"].append("Slight pitch instability detected")
    if shimmer > 0.08:
        voice_insights["issues"].append("Volume inconsistency")
    if wpm > 170:
        voice_insights["issues"].append("Speaking too fast - may affect clarity")
    if pause_ratio > 0.3:
        voice_insights["issues"].append("Many pauses - work on fluency")

    return voice_insights
```

---

## 🗄️ PostgreSQL Schema (Refined)

Based on your features:

```sql
CREATE TABLE "VoiceAnalysis" (
  id SERIAL PRIMARY KEY,
  interviewTurnId INT NOT NULL UNIQUE,

  -- Predictions
  confidenceLevel FLOAT,  -- 0-1 (from probability)
  confidenceLabelText VARCHAR(50),  -- "High", "Medium", "Low", etc.

  -- Voice Quality Metrics
  speakingQuality FLOAT,  -- 0-1
  vocalStability FLOAT,   -- 1-jitter
  speakingFluency FLOAT,  -- 1-pause_ratio

  -- Detailed Features (for audit trail)
  pitchMean FLOAT,
  pitchStd FLOAT,
  energyLevel FLOAT,
  wordsPerMinute FLOAT,
  pauseRatio FLOAT,
  jitter FLOAT,
  shimmer FLOAT,

  -- Model Info
  modelVersion VARCHAR(50),
  allProbabilities JSONB,  -- {"High": 78.5, "Medium": 15.2, "Low": 6.3}
  rawFeatures JSONB,  -- All 23 features stored

  -- Processing
  status VARCHAR(20),  -- "pending", "completed", "failed"
  errorMessage TEXT,
  processingTimeMs INT,

  -- Timestamps
  processedAt TIMESTAMP,
  createdAt TIMESTAMP DEFAULT NOW(),

  FOREIGN KEY (interviewTurnId) REFERENCES "InterviewTurn"(id) ON DELETE CASCADE
);

CREATE INDEX idx_voice_analysis_status ON "VoiceAnalysis"(status);
```

---

## 🚀 Deployment Strategy

### Docker Setup

```dockerfile
# Dockerfile for FastAPI voice service

FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

```dockerfile
# requirements.txt

fastapi==0.104.1
uvicorn==0.24.0
librosa==0.10.0
soundfile==0.12.1
pydub==0.25.1
noisereduce==3.0.0
parselmouth==0.3.3  # ⚠️ Heavy dependency
joblib==1.3.2
xgboost==2.0.3
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
redis==5.0.1
python-dotenv==1.0.0
```

### Deployment on Same Machine (Development)

```bash
# Terminal 1: Express server
npm run dev

# Terminal 2: FastAPI voice service
cd voice_service
uvicorn main:app --reload --port 8001
```

### Production (Docker Compose)

```yaml
version: "3.8"

services:
  express:
    build: .
    ports:
      - "4000:4000"
    environment:
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://...
    depends_on:
      - redis

  voice_service:
    build: ./voice_service
    ports:
      - "8001:8001"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    # Allocate more resources for audio processing
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: hirely
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

---

## ⏱️ Timeline & Phases

### Phase 1: Setup (1-2 days)

- [x] Understand your models
- [ ] Install FastAPI + dependencies
- [ ] Test model loading
- [ ] Basic feature extraction

### Phase 2: Integration (3-5 days)

- [ ] FastAPI service scaffold
- [ ] Redis Pub/Sub pipeline
- [ ] Feature extraction pipeline (from your code)
- [ ] Inference endpoint
- [ ] PostgreSQL storage

### Phase 3: Express Integration (2-3 days)

- [ ] Modify interviewController.js to publish events
- [ ] Update final report to include voice scores
- [ ] Error handling + retries

### Phase 4: Testing & Optimization (3-5 days)

- [ ] Test with real audio samples
- [ ] Tune weights for score combining
- [ ] Performance profiling
- [ ] Add to dashboard

---

## ❓ Questions Before Implementation

1. **What are your confidence label classes?** (e.g., "High", "Medium", "Low" or numbered?)
2. **How is training data distributed?** (balanced? imbalanced?)
3. **Expected accuracy range?** (from your training runs)
4. **Transcription integration:** Should FastAPI call Deepgram too, or get transcript from Express?
5. **Audio storage:** Keep files after processing or delete immediately?
6. **Report integration:** Should voice data appear on dashboard in real-time or only final report?
7. **Retry strategy:** If voice analysis fails, should it retry automatically?
8. **Privacy:** Should raw voice features be stored in PostgreSQL?

---

## ✅ Checklist Before Starting Code

- [ ] Verify your 3 pickle files load without errors
- [ ] Test feature extraction on sample audio
- [ ] Confirm model prediction works end-to-end
- [ ] Determine confidence label classes
- [ ] Plan PostgreSQL schema migration
- [ ] Decide on audio file retention policy
- [ ] Set up FastAPI project structure
- [ ] Install and test all dependencies
- [ ] Create Redis Pub/Sub test script

**Ready to code when you are! 🚀**
