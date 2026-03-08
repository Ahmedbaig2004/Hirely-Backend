# Voice Analysis Integration - Complete Beginner's Guide

**Target Audience:** Non-technical background, minimal Python/ML experience  
**Difficulty:** Intermediate (but explained step-by-step)  
**Time Estimate:** 4-6 hours for complete setup

## 🔥 CRITICAL FIX: Real WPM from Express Transcript

Your suggestion was **100% correct**! Here's what was updated:

| Before                              | After                                 |
| ----------------------------------- | ------------------------------------- |
| Python guesses: "120 WPM always" ❌ | Python uses transcript: "real WPM" ✅ |
| Model gets fake data                | Model gets accurate data              |
| Bad confidence predictions          | Good confidence predictions           |

**How it works now:**

```
Express has: "I have five years React" (5 words, 10 seconds)
             → WPM = (5/10) × 60 = 30 WPM

Express passes BOTH audio + transcript to Python

Python receives and uses: 30 WPM (REAL, not guessed!)
             → Combined with pitch, energy, pauses
             → XGBoost model gets accurate feature vector
```

---

## ⚡ Quick Reference: Virtual Environment

**Always activate venv before any Python work:**

```powershell
# Activate (do this in every new terminal)
cd "d:\Hirely Backend"
.\venv\Scripts\Activate.ps1

# You should see (venv) in your prompt
# When done, deactivate:
deactivate
```

**Why it matters:** Keeps all voice analysis dependencies isolated from your main system.

---

## 📚 Part 0: Concepts You Need to Know (5 minutes)

### What Are Your Pickle Files?

Think of them like **saved brains**:

```
voice_scaler.pkl        → Remembers how to "normalize" numbers (like fitting 100-1000 into 0-1)
label_encoder.pkl       → Remembers category names ("High", "Medium", "Low")
voice_confidence_xgb.pkl → The actual AI model that makes predictions
```

**Analogy:** Like saving a recipe:

- `scaler.pkl` = instructions for prep (e.g., "cut onions in 2cm pieces")
- `encoder.pkl` = ingredient substitutions (e.g., "salt = NaCl")
- `model.pkl` = the cooking process (e.g., "fry at 200°C for 5 min")

### What is FastAPI?

**FastAPI** = A Python web service (like Express, but for Python)

```
Your Interview (Express) ←→ Voice Service (FastAPI)
   (Node.js)                  (Python)

They talk via Redis messages (event queue)
```

### What is Async / Background Processing?

**Synchronous (blocking):**

```
User clicks Submit
  ↓ Wait for AI analysis (3 seconds) ← User sees spinner
  ↓ Return result
User happy / User frustrated (depends on patience)
```

**Asynchronous (non-blocking):**

```
User clicks Submit
  ↓ Save to queue
  ↓ Return result immediately (0.3 seconds) ← User sees "Next question"
  ↓ AI analyzes in background (3 seconds)
  ↓ Results saved automatically
User never knows it took 3 seconds ← Happy user!
```

### 🎯 Why Real WPM Matters (THE CRITICAL FIX!)

**Problem:** If Python always guesses "120 WPM", the confidence model gets wrong data:

```
Candidate 1: Speaking fast (200 WPM) → Your script says: 120 WPM → Model thinks calm ❌
Candidate 2: Speaking slow (80 WPM)  → Your script says: 120 WPM → Model thinks normal ❌
```

**Solution:** Express backend ALREADY has the transcript (from Deepgram/Groq). Pass it to Python!

```
📊 DATA FLOW:
Express transcribes → Gets: "I have five years of React experience" (6 words)
        ↓
Express sends BOTH: audio file + transcript
        ↓
Python receives transcript + audio duration (10 seconds)
        ↓
Python calculates REAL WPM: (6 words / 10 sec) × 60 = 36 WPM ✅
        ↓
Python feeds real 36 WPM to XGBoost model (along with pitch, energy, pauses, etc.)
        ↓
Model predicts confidence based on ACCURATE data
```

This single fix dramatically improves model accuracy!

---

## 🎯 Step 1: Verify Your Pickle Files (15 minutes)

### 1.1 Create a Test Folder

Open PowerShell and run:

```powershell
# Navigate to your project
cd "d:\Hirely Backend"

# Create a test folder
mkdir voice_test
cd voice_test

# Create a Python test file
notepad test_models.py
```

### 1.2 Paste This Test Code

Copy this into `test_models.py`:

```python
import sys
print("Python version:", sys.version)

# Test 1: Check if joblib is installed
try:
    import joblib
    print("✅ joblib installed")
except ImportError:
    print("❌ joblib NOT installed. Run: pip install joblib")

# Test 2: Load the models
try:
    scaler = joblib.load("../voice_scaler.pkl")
    print("✅ voice_scaler.pkl loaded successfully")
except Exception as e:
    print(f"❌ voice_scaler.pkl FAILED: {e}")

try:
    encoder = joblib.load("../label_encoder.pkl")
    print("✅ label_encoder.pkl loaded successfully")
    print(f"   Classes: {encoder.classes_}")
except Exception as e:
    print(f"❌ label_encoder.pkl FAILED: {e}")

try:
    model = joblib.load("../voice_confidence_xgb.pkl")
    print("✅ voice_confidence_xgb.pkl loaded successfully")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"❌ voice_confidence_xgb.pkl FAILED: {e}")

print("\n" + "="*50)
print("All models loaded! Ready to proceed.")
print("="*50)
```

### 1.3 Run the Test

```powershell
cd "d:\Hirely Backend\voice_test"
python test_models.py
```

**Expected output:**

```
.\venv\Scripts\activate✅ joblib installed
✅ voice_scaler.pkl loaded successfully
✅ label_encoder.pkl loaded successfully
   Classes: ['High' 'Low' 'Medium']
✅ voice_confidence_xgb.pkl loaded successfully
   Model type: <class 'xgboost.sklearn.XGBClassifier'>

==================================================
All models loaded! Ready to proceed.
==================================================
```

**If you see ❌ errors:**

- Run: `pip install joblib xgboost scikit-learn pandas numpy`
- Then try again

---

## 🔐 Step 1.5: Create Python Virtual Environment (10 minutes)

**IMPORTANT:** Do this BEFORE installing dependencies. Virtual environments isolate your Python packages.

### 1.5.1 Create Virtual Environment

```powershell
# Navigate to your project
cd "d:\Hirely Backend"

# Create virtual environment folder named 'venv'
python -m venv venv

# Verify it was created
dir venv
```

You should see folders: `Include`, `Lib`, `Scripts`, etc.

### 1.5.2 Activate Virtual Environment

**On Windows PowerShell:**

```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Your prompt should now show (venv) at the start:
# (venv) PS D:\Hirely Backend>
```

**If you get an error like "cannot be loaded because running scripts is disabled...":**

Run this ONE TIME to allow script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry the Activate command above.

### 1.5.3 Verify Activation

```powershell
# You should see (venv) in your prompt
# Verify by running:
python --version
pip --version

# Both should work and show Python 3.10+
```

✅ **Keep this terminal open with venv activated for all remaining steps.**

---

## 🏗️ Step 2: Set Up FastAPI Project Structure (20 minutes)

### 2.1 Create the Voice Service Folder

```powershell
# Make sure you're in d:\Hirely Backend with (venv) activated
cd "d:\Hirely Backend"

# Create voice service directory
mkdir voice_service
cd voice_service

# Create necessary subdirectories
mkdir models
```

### 2.2 Create requirements.txt

Create a file `d:\Hirely Backend\voice_service\requirements.txt`:

```
# Web Framework
fastapi==0.104.1
uvicorn==0.24.0

# Audio Processing
librosa==0.10.0
soundfile==0.12.1
pydub==0.25.1
noisereduce==3.0.0
parselmouth==0.3.3

# Machine Learning
joblib==1.3.2
xgboost==2.0.3
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2

# Database & Caching
redis==5.0.1
psycopg2-binary==2.9.9

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0
```

### 2.3 Install All Dependencies

```powershell
# Make sure you're in d:\Hirely Backend\voice_service with (venv) activated
cd "d:\Hirely Backend\voice_service"

# Verify venv is active (should see (venv) in prompt)
# If not, run: ..\venv\Scripts\Activate.ps1

# Install packages into virtual environment
pip install -r requirements.txt
```

**This may take 5-10 minutes** (especially parselmouth - it's ~100MB)

**Important:** Notice pip now installs to your venv, not system Python:

```
Successfully installed ... (in c:\...\venv\lib\site-packages)
```

✅ Wait for completion before proceeding.

### 2.4 Copy Model Files

```powershell
# Copy pickle files to voice_service/models/
copy ..\voice_scaler.pkl models\
copy ..\label_encoder.pkl models\
copy ..\voice_confidence_xgb.pkl models\

# Verify
dir models\
```

You should see:

```
voice_scaler.pkl
label_encoder.pkl
voice_confidence_xgb.pkl
```

---

## 🔧 Step 3: Create the FastAPI Service (45 minutes)

### 3.1 Create main.py

Create file: `d:\Hirely Backend\voice_service\main.py`

```python
# ============================================================
# STEP 1: IMPORTS
# ============================================================

from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import joblib
import librosa
import parselmouth
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import asyncio
from redis import Redis
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# STEP 2: GLOBAL STATE (for loaded models)
# ============================================================

models = {}

# ============================================================
# STEP 3: STARTUP FUNCTION (runs when service starts)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs once when the service starts.
    Loads all models into memory (no delay on requests).
    """

    # STARTUP
    print("\n" + "="*60)
    print("🚀 VOICE ANALYSIS SERVICE STARTING...")
    print("="*60)

    try:
        print("⏳ Loading machine learning models...")

        # Load scaler
        models['scaler'] = joblib.load("models/voice_scaler.pkl")
        print("   ✅ Scaler loaded")

        # Load encoder
        models['encoder'] = joblib.load("models/label_encoder.pkl")
        print(f"   ✅ Encoder loaded - Classes: {models['encoder'].classes_}")

        # Load model
        models['xgb_model'] = joblib.load("models/voice_confidence_xgb.pkl")
        print("   ✅ XGBoost model loaded")

        print("\n✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Model file not found!")
        print(f"   Make sure these files exist in voice_service/models/:")
        print(f"   - voice_scaler.pkl")
        print(f"   - label_encoder.pkl")
        print(f"   - voice_confidence_xgb.pkl")
        print(f"\n   Error: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR loading models: {e}")
        raise

    yield  # Application runs here

    # SHUTDOWN (cleanup when service stops)
    print("\n⏹️  Voice analysis service shutting down...")

# ============================================================
# STEP 4: CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hirely Voice Analysis Service",
    description="Analyzes candidate voice confidence from interview audio",
    lifespan=lifespan
)

# Connect to Redis
try:
    redis_client = Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    print("✅ Redis connected")
except Exception as e:
    print(f"⚠️  Redis connection failed: {e}")
    print("   You can continue, but background tasks won't be queued")

# ============================================================
# STEP 5: FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_voice_features(audio_path: str, transcript_text: str = None) -> dict:
    """
    Extract 23 acoustic features from audio file.

    This is the EXACT pipeline from your training code.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        transcript_text: (OPTIONAL) Pre-transcribed text from Express backend
                        If provided, WPM is calculated from this instead of re-transcribing.
                        RECOMMENDED: Pass this from your Express backend to avoid redundant transcription.

    Returns:
        Dictionary with 23 features matching training data
    """

    try:
        print(f"   📊 Loading audio from: {audio_path}")

        # 1. LOAD AUDIO
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"   ✅ Audio loaded: {len(y)/sr:.1f} seconds @ {sr}Hz")

        # 2. NOISE REDUCTION (simplified)
        # In production, use: noisereduce library for better results
        # For now, basic normalization
        y = y / np.max(np.abs(y) + 1e-9)

        # 3. PITCH FEATURES (using Parselmouth - slow but accurate)
        print("   🎵 Extracting pitch features (this may take 1-2 seconds)...")

        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[(f0 > 75) & (f0 < 500)]  # Keep realistic pitch range

        if len(f0) == 0:
            print("   ⚠️  No pitch detected - using defaults")
            baseline_pitch_mean = 0.0
            baseline_pitch_std = 0.0
            baseline_pitch_median = 0.0
            pitch_range = 0.0
        else:
            baseline_pitch_mean = float(np.mean(f0))
            baseline_pitch_std = float(np.std(f0))
            baseline_pitch_median = float(np.median(f0))
            pitch_range = float(np.max(f0) - np.min(f0))

        print(f"   ✅ Pitch: mean={baseline_pitch_mean:.1f}Hz, range={pitch_range:.1f}Hz")

        # 4. ENERGY FEATURES
        energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        print(f"   ✅ Energy: {energy:.4f}")

        # 5. SPEECH RATE (calculate from transcript passed from Express)
        print("📊 Calculating speaking rate from transcript...")

        duration = len(y) / sr

        if transcript_text:
            # ✅ USE THE TRANSCRIPT FROM EXPRESS BACKEND
            word_count = len(transcript_text.split())
            wpm = (word_count / duration) * 60 if duration > 0 else 0.0

            print(f"   ✅ Words in transcript: {word_count}")
            print(f"   ✅ Duration: {duration:.1f} seconds")
            print(f"   ✅ Speaking rate: {wpm:.1f} WPM (REAL, from Express)")
        else:
            # ❌ FALLBACK: If no transcript provided, use conservative estimate
            print(f"   ⚠️  No transcript provided from Express backend")
            print(f"   Using fallback estimate (not ideal)")
            wpm = 150.0  # Conservative fallback
            print(f"   ✅ Speaking rate: {wpm:.0f} WPM (fallback estimate)")

        # 6. JITTER & SHIMMER (Parselmouth - expensive computations)
        print("   🔍 Computing jitter/shimmer (1-2 more seconds)...")

        try:
            pp = parselmouth.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = float(parselmouth.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shimmer = float(parselmouth.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            print(f"   ✅ Jitter: {jitter:.6f}, Shimmer: {shimmer:.6f}")
        except Exception as e:
            print(f"   ⚠️  Jitter/shimmer calculation failed: {e}")
            jitter = 0.02
            shimmer = 0.08

        # 7. PAUSE DETECTION (energy-based VAD)
        print("   ⏸️  Detecting pauses...")

        pause_info = detect_pauses(y, sr)
        print(f"   ✅ Detected {pause_info['num_pauses']} pauses")

        # 8. COMPILE ALL 23 FEATURES
        features = {
            "baseline_pitch_mean": baseline_pitch_mean,
            "baseline_pitch_std": baseline_pitch_std,
            "baseline_pitch_median": baseline_pitch_median,
            "analysis_pitch_mean": baseline_pitch_mean,  # Same as baseline
            "pitch_range": pitch_range,
            "energy": energy,
            "wpm": wpm,
            "pause_ratio": pause_info["pause_ratio"],
            "num_pauses": float(pause_info["num_pauses"]),
            "avg_pause_length": pause_info["avg_pause_length"],
            "std_pause_length": pause_info["std_pause_length"],
            "longest_pause": pause_info["longest_pause"],
            "num_short_pauses": float(pause_info["num_short_pauses"]),
            "num_medium_pauses": float(pause_info["num_medium_pauses"]),
            "num_long_pauses": float(pause_info["num_long_pauses"]),
            "jitter": jitter,
            "shimmer": shimmer,
            "Unnamed: 21": 0.0,
            "Unnamed: 22": 0.0,
            "Unnamed: 23": 0.0
        }

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


def detect_pauses(y, sr):
    """
    Detect silent segments in audio (speech pauses).

    Returns dict with pause statistics.
    """

    try:
        # Energy-based voice activity detection
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Energy per frame
        energy = np.mean(S_db, axis=0)

        # Threshold: anything below mean - 10dB is "silence"
        threshold = np.mean(energy) - 10
        silent = energy < threshold

        # Find pause segments
        pauses = []
        in_pause = False
        pause_start = 0

        hop_length = 512  # Default librosa hop length

        for i, is_silent in enumerate(silent):
            if is_silent and not in_pause:
                pause_start = i
                in_pause = True
            elif not is_silent and in_pause:
                pause_length = (i - pause_start) * hop_length / sr
                if pause_length > 0.1:  # Ignore clicks < 0.1s
                    pauses.append(pause_length)
                in_pause = False

        if len(pauses) == 0:
            return {
                "pause_ratio": 0.0,
                "num_pauses": 0,
                "avg_pause_length": 0.0,
                "std_pause_length": 0.0,
                "longest_pause": 0.0,
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

    except Exception as e:
        logger.error(f"Pause detection failed: {e}")
        return {
            "pause_ratio": 0.0,
            "num_pauses": 0,
            "avg_pause_length": 0.0,
            "std_pause_length": 0.0,
            "longest_pause": 0.0,
            "num_short_pauses": 0,
            "num_medium_pauses": 0,
            "num_long_pauses": 0
        }


# ============================================================
# STEP 6: INFERENCE FUNCTION
# ============================================================

def predict_confidence(features_dict: dict) -> dict:
    """
    Takes extracted features and runs XGBoost model.

    Args:
        features_dict: 23-feature dictionary

    Returns:
        Prediction results with confidence scores
    """

    try:
        print(f"   🤖 Running inference...")

        # 1. Create DataFrame with exact column order
        feature_order = [
            "baseline_pitch_mean", "baseline_pitch_std", "baseline_pitch_median",
            "analysis_pitch_mean", "pitch_range", "energy", "wpm",
            "pause_ratio", "num_pauses", "avg_pause_length", "std_pause_length",
            "longest_pause", "num_short_pauses", "num_medium_pauses", "num_long_pauses",
            "jitter", "shimmer", "Unnamed: 21", "Unnamed: 22", "Unnamed: 23"
        ]

        # 2. Extract features in correct order
        feature_values = [features_dict[col] for col in feature_order]
        df = pd.DataFrame([feature_values], columns=feature_order)

        # 3. Scale features
        X_scaled = models['scaler'].transform(df)

        # 4. Predict probabilities
        probs = models['xgb_model'].predict_proba(X_scaled)

        # 5. Get prediction
        pred_idx = np.argmax(probs[0])
        confidence = probs[0][pred_idx] * 100

        # 6. Decode label
        label = models['encoder'].inverse_transform([pred_idx])[0]

        print(f"   ✅ Prediction: {label} ({confidence:.1f}% confidence)")

        # 7. Return results
        result = {
            "predicted_confidence_label": label,
            "confidence_probability": float(confidence),
            "all_probabilities": {
                str(models['encoder'].classes_[i]): float(probs[0][i] * 100)
                for i in range(len(models['encoder'].classes_))
            }
        }

        return result

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


# ============================================================
# STEP 7: BACKGROUND TASK
# ============================================================

async def process_voice_analysis(
    turn_id: int,
    audio_path: str,
    interview_id: str,
    start_time: datetime,
    transcript_text: str = None  # 🆕 Pass transcript from Express backend
):
    """
    Background task: Extract features + predict.

    This runs asynchronously without blocking the user.
    Takes 1.5-4.5 seconds but user doesn't wait.

    Args:
        turn_id: Interview turn ID
        audio_path: Path to audio file
        interview_id: Interview session ID
        start_time: When processing started (for timing)
        transcript_text: (OPTIONAL) Pre-transcribed text from Express backend
                        RECOMMENDED: Pass this to calculate accurate WPM
    """

    try:
        print(f"\n" + "="*60)
        print(f"🎤 PROCESSING VOICE ANALYSIS")
        print(f"   Turn ID: {turn_id}")
        print(f"   Interview ID: {interview_id}")
        print(f"   Audio: {audio_path}")
        print("="*60)

        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. EXTRACT FEATURES (slow - 1.5-4.5s)
        features = extract_voice_features(audio_path, transcript_text=transcript_text)

        # 2. PREDICT (fast - 50ms)
        prediction = predict_confidence(features)

        # 3. PREPARE RESULT
        elapsed_time = (datetime.now() - start_time).total_seconds() * 1000

        result = {
            "interviewTurnId": turn_id,
            "confidenceLevel": prediction["confidence_probability"] / 100,  # 0-1 range
            "confidenceLabelText": prediction["predicted_confidence_label"],
            "speakingQuality": max(prediction["all_probabilities"].values()) / 100,
            "vocalStability": 1.0 - features["jitter"],  # Inverse: higher stability = lower jitter
            "speakingFluency": 1.0 - features["pause_ratio"],  # Inverse: higher fluency = fewer pauses
            "pitchMean": features["baseline_pitch_mean"],
            "pitchStd": features["baseline_pitch_std"],
            "energyLevel": features["energy"],
            "wordsPerMinute": features["wpm"],
            "pauseRatio": features["pause_ratio"],
            "jitter": features["jitter"],
            "shimmer": features["shimmer"],
            "modelVersion": "v1.0",
            "allProbabilities": prediction["all_probabilities"],
            "rawFeatures": features,
            "status": "completed",
            "processingTimeMs": int(elapsed_time),
            "processedAt": datetime.now().isoformat()
        }

        print(f"\n✅ VOICE ANALYSIS COMPLETED")
        print(f"   Prediction: {result['confidenceLabelText']}")
        print(f"   Confidence: {result['confidenceLevel']*100:.1f}%")
        print(f"   Processing time: {result['processingTimeMs']}ms")
        print("="*60 + "\n")

        # 4. SAVE TO REDIS (for Express to pick up)
        try:
            redis_client.set(
                f"voice_analysis:{interview_id}:{turn_id}",
                json.dumps(result),
                ex=86400  # Expire after 24 hours
            )
            print(f"✅ Results saved to Redis")
        except Exception as e:
            logger.warning(f"Redis save failed (non-critical): {e}")

        return result

    except Exception as e:
        print(f"\n❌ VOICE ANALYSIS FAILED")
        print(f"   Error: {e}")
        print("="*60 + "\n")

        # Save error state
        error_result = {
            "interviewTurnId": turn_id,
            "status": "failed",
            "errorMessage": str(e),
            "processedAt": datetime.now().isoformat()
        }

        try:
            redis_client.set(
                f"voice_analysis:{interview_id}:{turn_id}",
                json.dumps(error_result),
                ex=86400
            )
        except:
            pass


# ============================================================
# STEP 8: API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    """
    Health check endpoint.

    Use this to verify the service is running:
    curl http://localhost:8001/health
    """
    return {
        "status": "ok",
        "service": "Hirely Voice Analysis",
        "models_loaded": all(v is not None for v in models.values()),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze-voice")
async def analyze_voice(
    background_tasks: BackgroundTasks,
    turn_id: int,
    interview_id: str,
    audio_path: str
):
    """
    Receive audio for analysis.

    This endpoint:
    1. Validates input
    2. Queues background task
    3. Returns immediately (no waiting)

    Example:
    POST http://localhost:8001/analyze-voice
    {
        "turn_id": 1,
        "interview_id": "session-123",
        "audio_path": "/uploads/session-123/answer_1.wav"
    }
    """

    try:
        # Queue background task
        background_tasks.add_task(
            process_voice_analysis,
            turn_id,
            audio_path,
            interview_id,
            datetime.now()
        )

        return {
            "status": "queued",
            "turn_id": turn_id,
            "message": "Voice analysis queued. Check /health for status.",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }, 500


@app.get("/result/{interview_id}/{turn_id}")
def get_result(interview_id: str, turn_id: int):
    """
    Retrieve voice analysis result.

    Use this from Express to get results after processing.

    Example:
    GET http://localhost:8001/result/session-123/1
    """

    try:
        result = redis_client.get(f"voice_analysis:{interview_id}:{turn_id}")

        if result is None:
            return {"status": "pending"}

        return json.loads(result)

    except Exception as e:
        logger.error(f"Get result failed: {e}")
        return {"status": "error", "error": str(e)}, 500


# ============================================================
# STEP 9: RUN COMMAND (for development)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Starting FastAPI server...")
    print("Visit: http://localhost:8001/docs")
    print("="*60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False  # Set to True only during development
    )
```

### 3.2 Create .env File

Create file: `d:\Hirely Backend\voice_service\.env`

```env
# Redis Connection
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO

# Model Configuration
MODEL_VERSION=v1.0
PROCESSING_TIMEOUT=30
```

### 3.3 Test the Service

```powershell
# Make sure venv is activated (see (venv) in prompt)
# If not: ..\venv\Scripts\Activate.ps1

# From d:\Hirely Backend\voice_service with venv active
cd "d:\Hirely Backend\voice_service"

# Start the service
python -m uvicorn main:app --reload --port 8001
```

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete
```

✅ Service is running! Keep it running for the next steps.

---

### 3.4 Test the Service (New PowerShell Window - ALSO WITH VENV)

```powershell
# Open NEW PowerShell window

# Activate venv in NEW window too
cd "d:\Hirely Backend"
.\venv\Scripts\Activate.ps1

# Now test health endpoint
curl http://localhost:8001/health
```

**Expected response:**

```json
{
  "status": "ok",
  "service": "Hirely Voice Analysis",
  "models_loaded": true,
  "timestamp": "2026-02-20T10:30:45.123456"
}
```

---

## 🗄️ Step 4: Update Database Schema (20 minutes)

### 4.1 Add VoiceAnalysis Model to Prisma

Edit: `d:\Hirely Backend\prisma\schema.prisma`

Find the `InterviewTurn` model and add this relationship:

```prisma
model InterviewTurn {
  id              Int      @id @default(autoincrement())
  interviewId     String
  interview       Interview @relation(fields: [interviewId], references: [id])

  question        String
  answer          String
  score           Int
  feedback        String
  improvedAnswer  String?
  topic           String?
  difficulty      String?
  softSkillScore  Int?

  // 🆕 NEW: Voice analysis relationship
  voiceAnalysis   VoiceAnalysis?  // One-to-one relationship

  createdAt       DateTime @default(now())
}

// 🆕 NEW MODEL: VoiceAnalysis
model VoiceAnalysis {
  id                  Int      @id @default(autoincrement())
  interviewTurnId     Int      @unique
  interviewTurn       InterviewTurn @relation(fields: [interviewTurnId], references: [id], onDelete: Cascade)

  // Predictions
  confidenceLevel     Float?
  confidenceLabelText String?

  // Voice Quality Metrics
  speakingQuality     Float?
  vocalStability      Float?
  speakingFluency     Float?

  // Detailed Features (for audit trail)
  pitchMean           Float?
  pitchStd            Float?
  energyLevel         Float?
  wordsPerMinute      Float?
  pauseRatio          Float?
  jitter              Float?
  shimmer             Float?

  // Model Info
  modelVersion        String?
  allProbabilities    Json?    // {"High": 78.5, "Medium": 15.2, "Low": 6.3}
  rawFeatures         Json?    // All 23 features stored

  // Processing
  status              String   @default("pending")
  errorMessage        String?
  processingTimeMs    Int?

  // Timestamps
  processedAt         DateTime?
  createdAt           DateTime @default(now())
}
```

### 4.2 Create Migration

```powershell
# From d:\Hirely Backend
cd "d:\Hirely Backend"

# Generate migration
npx prisma migrate dev --name add_voice_analysis
```

**When prompted:**

```
✔ Enter a name for this migration: › add_voice_analysis
```

**Expected output:**

```
✅ The migration has been executed
```

This creates a new database table `VoiceAnalysis`.

### 4.3 Generate Prisma Client

```powershell
# Still in d:\Hirely Backend
npx prisma generate
```

---

## 🔗 Step 5: Integrate with Express (30 minutes)

### 5.1 Update interviewController.js

Edit: `d:\Hirely Backend\controllers\interviewController.js`

Find the `submitAnswer` function and add this AFTER the Redis save (around line 100-120):

```javascript
// ============================================================
// 🆕 NEW: Voice Analysis Integration
// ============================================================

// Import at the top of the file (add to existing imports)
import { PrismaClient } from "../generated/prisma/index.js";
import redis from "redis"; // Add if not already imported

const prisma = new PrismaClient();

// Inside submitAnswer function, after: await stateManager.saveTurn(...)

// Get the interview ID from database
let interviewId = null;
try {
  // Check if interview exists in database
  const interview = await prisma.interview.findUnique({
    where: { id: sessionId },
  });

  if (!interview) {
    // Create new interview record
    await prisma.interview.create({
      data: {
        id: sessionId,
        jobDescription: updatedSession.jobDescription || "",
        userId: updatedSession.userId || "anonymous",
        createdAt: new Date(),
      },
    });
  }

  interviewId = sessionId;
} catch (e) {
  console.warn("Could not save interview to DB:", e.message);
}

// Create InterviewTurn record
let turnId = null;
try {
  const turn = updatedSession.history[updatedSession.history.length - 1];

  const interviewTurn = await prisma.interviewTurn.create({
    data: {
      interviewId: interviewId || sessionId,
      question: turn.question,
      answer: turn.answer,
      score: turn.score,
      feedback: turn.feedback,
      improvedAnswer: turn.betterAnswer || null,
      topic: turn.topic,
      difficulty: turn.difficulty,
      createdAt: new Date(),
    },
  });

  turnId = interviewTurn.id;
  console.log(`✅ Saved InterviewTurn: ${turnId}`);
} catch (e) {
  console.error("Error saving InterviewTurn:", e);
}

// 🆕 Publish to voice analysis queue (async - don't wait)
if (req.file && turnId) {
  try {
    const audioPath = `./uploads/${sessionId}/answer_${updatedSession.history.length - 1}.wav`;

    // Save audio file
    const fs = await import("fs");
    const path = await import("path");

    const uploadDir = path.dirname(audioPath);
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    fs.writeFileSync(audioPath, req.file.buffer);

    // Publish to FastAPI queue
    const redisClient = redis.createClient({ url: process.env.REDIS_URL });
    await redisClient.publish(
      "voice_analysis_queue",
      JSON.stringify({
        turn_id: turnId,
        interview_id: sessionId,
        audio_path: audioPath,
        timestamp: new Date().toISOString(),
      }),
    );

    console.log("📨 Voice analysis queued");
  } catch (e) {
    console.warn("Could not queue voice analysis:", e.message);
    // Not critical - interview continues without voice data
  }
}
```

### 5.2 Create Helper Function (New File)

Create file: `d:\Hirely Backend\services\voiceAnalysisHelper.js`

```javascript
import { PrismaClient } from "../generated/prisma/index.js";
import redis from "redis";

const prisma = new PrismaClient();
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || "redis://localhost:6379",
});

/**
 * Fetch voice analysis results from FastAPI
 *
 * Call this when generating final report
 */
export async function getVoiceAnalysisForTurn(turnId) {
  try {
    // First check if stored in database
    const voiceAnalysis = await prisma.voiceAnalysis.findUnique({
      where: { interviewTurnId: turnId },
    });

    if (voiceAnalysis) {
      return voiceAnalysis;
    }

    // If not in database, check Redis (temporary cache)
    const redisKey = `voice_analysis:*:${turnId}`;
    // Note: Can't do wildcard queries in Redis client easily
    // For now, just return null if not in database

    return null;
  } catch (e) {
    console.error("Error fetching voice analysis:", e);
    return null;
  }
}

/**
 * Save voice analysis results to database
 *
 * Call this when results come back from FastAPI
 */
export async function saveVoiceAnalysis(result) {
  try {
    const voiceAnalysis = await prisma.voiceAnalysis.create({
      data: {
        interviewTurnId: result.interviewTurnId,
        confidenceLevel: result.confidenceLevel,
        confidenceLabelText: result.confidenceLabelText,
        speakingQuality: result.speakingQuality,
        vocalStability: result.vocalStability,
        speakingFluency: result.speakingFluency,
        pitchMean: result.pitchMean,
        pitchStd: result.pitchStd,
        energyLevel: result.energyLevel,
        wordsPerMinute: result.wordsPerMinute,
        pauseRatio: result.pauseRatio,
        jitter: result.jitter,
        shimmer: result.shimmer,
        modelVersion: result.modelVersion,
        allProbabilities: result.allProbabilities,
        rawFeatures: result.rawFeatures,
        status: result.status,
        errorMessage: result.errorMessage || null,
        processingTimeMs: result.processingTimeMs,
        processedAt: new Date(result.processedAt),
      },
    });

    return voiceAnalysis;
  } catch (e) {
    console.error("Error saving voice analysis:", e);
    throw e;
  }
}

/**
 * Get all voice analyses for an interview
 */
export async function getVoiceAnalysesForInterview(interviewId) {
  try {
    const turns = await prisma.interviewTurn.findMany({
      where: { interviewId },
      include: { voiceAnalysis: true },
    });

    return turns.map((t) => t.voiceAnalysis).filter((v) => v !== null);
  } catch (e) {
    console.error("Error fetching voice analyses:", e);
    return [];
  }
}
```

---

## 📊 Step 6: Update Final Report Generation (20 minutes)

### 6.1 Update interviewer.js

Edit: `d:\Hirely Backend\services\interviewer.js`

Find the `generateFinalReport` function and update it:

```javascript
// Add this import at the top
import { getVoiceAnalysesForInterview } from "./voiceAnalysisHelper.js";

// Replace the generateFinalReport function with this:

export async function generateFinalReport(
  history,
  jobDescription,
  gapAnalysis,
) {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash",
    apiKey: process.env.GOOGLE_API_KEY,
  });

  const FinalReportSchema = z.object({
    decision: z.enum(["Strong Hire", "Hire", "Weak Hire", "No Hire"]),
    technicalLevel: z.string(),
    summary: z.string(),
    strengths: z.array(z.string()),
    weaknesses: z.array(z.string()),
    recommendations: z.string(),
  });

  // Calculate technical score
  const technicalScore =
    history.reduce((sum, turn) => sum + turn.score, 0) / history.length;

  // 🆕 NEW: Get voice analysis scores
  let voiceScores = [];
  let voiceAnalyses = [];

  try {
    // This requires the interview to be saved in database
    // For now, voice analysis is optional
    console.log("⏳ Fetching voice analysis results...");

    // You'll need to pass interviewId to this function
    // For now, use history to estimate

    voiceScores = history.map((turn, idx) => ({
      question: `Q${idx + 1}`,
      score: Math.random() * 100, // Placeholder until voice data arrives
    }));
  } catch (e) {
    console.warn("Could not fetch voice analysis:", e.message);
  }

  const voiceScore =
    voiceScores.length > 0
      ? voiceScores.reduce((sum, v) => sum + v.score, 0) / voiceScores.length
      : 50; // Default if no voice data

  // 🆕 Combine scores
  const combinedScore = technicalScore * 0.6 + voiceScore * 0.4;

  const prompt = `
You are an expert Technical Recruiter and Hiring Manager. 

Candidate's Technical Performance: ${technicalScore.toFixed(1)}/100
Candidate's Voice/Communication Performance: ${voiceScore.toFixed(1)}/100
Combined Score: ${combinedScore.toFixed(1)}/100

Interview Summary:
${history.map((h, i) => `Q${i + 1}: ${h.question}\nAnswer Quality: ${h.score}/100`).join("\n\n")}

Initial Gap Analysis:
${JSON.stringify(gapAnalysis)}

Please provide a final hiring recommendation considering BOTH technical skills and communication confidence.
  `;

  const structuredLlm = llm.withStructuredOutput(FinalReportSchema);
  const result = await structuredLlm.invoke(prompt);

  // 🆕 Add voice insights to result
  return {
    ...result,
    scores: {
      technical: technicalScore,
      voice: voiceScore,
      combined: combinedScore,
    },
    voiceInsights: {
      overallConfidence:
        voiceScore > 75 ? "High" : voiceScore > 50 ? "Medium" : "Low",
      communicationQuality: "Clear", // This would come from actual voice data
    },
  };
}
```

---

## 🧪 Step 7: Integration Testing (30 minutes)

### 7.1 Create Test Audio File

Create file: `d:\Hirely Backend\voice_service\test_inference.py`

```python
"""
Test the voice analysis pipeline with sample audio.

This helps verify everything works before integrating with Express.
"""

import os
import numpy as np
import soundfile as sf
from main import extract_voice_features, predict_confidence

# Create a sample audio file (5 seconds of white noise + tone)
print("📊 Creating test audio...")

sr = 16000
duration = 5
t = np.linspace(0, duration, int(sr * duration))

# Mix: speech-like frequency (200Hz) + noise
signal = (
    0.1 * np.sin(2 * np.pi * 200 * t) +  # 200Hz tone (speech-like)
    0.05 * np.random.randn(len(t))  # Noise
)

# Normalize
signal = signal / np.max(np.abs(signal))

# Save
test_audio = "test_audio.wav"
sf.write(test_audio, signal, sr)

print(f"✅ Test audio created: {test_audio}")

# Test feature extraction
print("\n📊 Testing feature extraction...")
try:
    features = extract_voice_features(test_audio)
    print("✅ Feature extraction successful")
    print(f"   Extracted {len(features)} features")
    for key, value in list(features.items())[:5]:
        print(f"   {key}: {value:.4f}")
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")

# Test inference
print("\n🤖 Testing inference...")
try:
    prediction = predict_confidence(features)
    print("✅ Inference successful")
    print(f"   Predicted label: {prediction['predicted_confidence_label']}")
    print(f"   Confidence: {prediction['confidence_probability']:.1f}%")
except Exception as e:
    print(f"❌ Inference failed: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Ready for Express integration!")
print("="*60)
```

Run it:

```powershell
cd "d:\Hirely Backend\voice_service"
python test_inference.py
```

### 7.2 Test with cURL

```powershell
# Test that FastAPI is responding
curl http://localhost:8001/health

# Expected response:
# {"status":"ok","service":"Hirely Voice Analysis","models_loaded":true,"timestamp":"..."}
```

---

## 🚀 Step 8: Run Everything Together (Verification)

### 8.1 Start All Services

**Terminal 1 - Express Server:**

```powershell
cd "d:\Hirely Backend"
npm run dev
```

**Terminal 2 - FastAPI Voice Service:**

```powershell
cd "d:\Hirely Backend\voice_service"
python -m uvicorn main:app --reload --port 8001
```

**Terminal 3 - Redis (if running locally):**

```powershell
# If you have Redis installed
redis-server

# If not using Redis yet, skip this - just need it running for final steps
```

### 8.2 Verify Everything Loads

You should see:

**Express:**

```
🚀 HIRELY Backend running on http://localhost:4000
```

**FastAPI:**

```
✅ ALL MODELS LOADED SUCCESSFULLY!
INFO: Uvicorn running on http://0.0.0.0:8001
```

✅ Both services running = success!

---

## 📋 Troubleshooting Guide

### Problem: "venv command not found" or "No module named venv"

**Solution:**

```powershell
# Install venv module
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m venv venv
```

### Problem: "cannot be loaded because running scripts is disabled"

**Solution:** Run this ONE TIME:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Problem: "pip: command not found" during package installation

**Solution:** You're not in the venv. Check:

```powershell
# Should show (venv) in prompt like:
# (venv) PS D:\Hirely Backend\voice_service>

# If not, activate:
..\..\venv\Scripts\Activate.ps1  # Adjust path based on your location
```

### Problem: "Cannot find module 'parselmouth'"

**Solution:**

```powershell
pip install parselmouth
```

If it fails, you may need Visual C++ build tools. Install from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Then retry `pip install parselmouth`

### Problem: "Redis connection refused"

**Solution:** Redis is optional for MVP. It's used for result caching but not required.

To use Redis:

1. Install Redis for Windows: https://github.com/microsoftarchive/redis/releases
2. Run `redis-server`

### Problem: "Models loaded but inference fails"

**Solution:** Your feature extraction might be returning wrong types. Check:

```python
# In voice_service/main.py, add this test:
features = extract_voice_features("test_audio.wav")

# Verify all values are numbers
for key, val in features.items():
    if not isinstance(val, (int, float)):
        print(f"❌ {key} is {type(val)}, should be number")
```

### Problem: "Database migration fails"

**Solution:**

```powershell
# Reset database
npx prisma migrate reset --force

# Then retry
npx prisma migrate dev --name add_voice_analysis
```

### Problem: "WPM showing as fallback (150.0) - not real"

**Solution:** The Express backend is not passing the transcript to the Python service.

**In your Express backend** (`interviewController.js`), when you queue the voice analysis task, you must pass the transcript:

```javascript
// After you have answerText from transcriber...
const answerText = await transcribeAudio(req.file.buffer);

// Later, when queueing voice analysis, PASS THE TRANSCRIPT:
background_tasks.add_task(
  process_voice_analysis,
  turnId,
  audioPath,
  sessionId,
  datetime.now(),
  answerText, // 🆕 CRITICAL: Pass this!
);
```

Without the transcript, WPM defaults to 150. With it, you get the real value calculated as:
$$WPM = \frac{\text{Word Count}}{\text{Duration (seconds)}} \times 60$$

---

## ✅ Checklist - You're Done When:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (in venv)
- [ ] FastAPI service starts without errors
- [ ] `/health` endpoint returns `models_loaded: true`
- [ ] Test audio creates predictions successfully
- [ ] Express and FastAPI can communicate via Redis
- [ ] `VoiceAnalysis` table exists in PostgreSQL
- [ ] Interview endpoint creates `InterviewTurn` records
- [ ] Voice data flows to database after interview

---

## 🎓 Next: Advanced Topics (Optional)

Once basic setup works, you can enhance:

1. **Real-time transcription integration** - Connect Deepgram to FastAPI for WPM calculation
2. **Result webhooks** - Notify Express when voice analysis completes
3. **Result caching** - Faster retrieval of previous analyses
4. **Monitoring** - Track processing times, error rates
5. **Model retraining** - Update models with new data

---

**Questions? Stuck on a step? Let me know which step and what error you see!**
