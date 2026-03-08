from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
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
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
import warnings
warnings.filterwarnings("ignore")

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
    Extract 23 acoustic features using Script 1's dual audio pipeline.

    PIPELINE:
    1. Load audio (raw)
    2. Extract noise profile (first 0.5 seconds)
    3. Create CLEAN version (80% noise reduction) → for voice activity detection
    4. Create RAW version (60% noise reduction) → for feature analysis
    5. Extract features from raw version
    6. Calculate accurate WPM from Express transcript

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        transcript_text: Pre-transcribed text from Express backend
                        REQUIRED for accurate WPM calculation

    Returns:
        Dictionary with 23 features matching training data
    """

    try:
        print(f"   📊 Loading audio from: {audio_path}")

        # ============================================================
        # STEP 1: LOAD AUDIO
        # ============================================================
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"   ✅ Audio loaded: {len(y)/sr:.1f} seconds @ {sr}Hz")

        # ============================================================
        # STEP 2: DUAL AUDIO PIPELINE (Noise Reduction)
        # ============================================================
        print("   🔊 Creating dual audio pipeline...")

        # Extract noise profile from first 0.5 seconds
        noise = y[:int(0.5 * sr)]

        # VERSION 1: CLEAN (80% noise reduction) - for VAD
        print("      🔇 Reducing noise (80% removal for clarity)...")
        clean = nr.reduce_noise(y=y, sr=sr, y_noise=noise, prop_decrease=0.8)
        clean /= max(np.max(np.abs(clean)), 1e-6)  # Normalize

        # VERSION 2: RAW (60% noise reduction) - preserves voice qualities
        print("      🔊 Gentle noise reduction (60% removal to preserve features)...")
        raw = nr.reduce_noise(y=y, sr=sr, y_noise=noise, prop_decrease=0.6)
        raw /= max(np.max(np.abs(raw)), 1e-6)  # Normalize

        print("   ✅ Dual pipeline complete")

        # ============================================================
        # STEP 3: VAD + COMPRESSION (on clean audio)
        # ============================================================
        print("   ✂️  Applying Voice Activity Detection (VAD)...")

        segments = librosa.effects.split(clean, top_db=30)
        speech = np.concatenate([clean[s:e] for s, e in segments]) if segments.any() else clean

        # Compress dynamic range for clarity
        audio_segment = AudioSegment(
            np.int16(speech * 32767).tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        audio_segment = compress_dynamic_range(audio_segment)
        print(f"   ✅ VAD complete: extracted {len(segments)} speech segments")

        # ============================================================
        # STEP 4: EXTRACT FEATURES (from RAW audio)
        # ============================================================
        print("   🎵 Extracting acoustic features...")

        # Load raw audio for Parselmouth (pitch analysis)
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

        print(f"   ✅ Pitch: mean={baseline_pitch_mean:.1f}Hz, std={baseline_pitch_std:.1f}Hz, range={pitch_range:.1f}Hz")

        # ENERGY (from raw audio)
        energy = float(np.mean(librosa.feature.rms(y=raw)[0]))
        print(f"   ✅ Energy: {energy:.4f}")

        # ============================================================
        # STEP 5: SPEECH RATE (WPM from Express transcript)
        # ============================================================
        print("   📊 Calculating speaking rate from Express transcript...")

        duration = len(raw) / sr

        if transcript_text and len(transcript_text.strip()) > 0:
            # ✅ USE REAL TRANSCRIPT FROM EXPRESS
            word_count = len(transcript_text.split())
            wpm = (word_count / duration) * 60 if duration > 0 else 0.0
            print(f"   ✅ Words: {word_count} | Duration: {duration:.1f}s | WPM: {wpm:.1f}")
        else:
            # Fallback only if no transcript
            wpm = 150.0
            print(f"   ⚠️  No transcript from Express. Using fallback: {wpm:.0f} WPM")

        # ============================================================
        # STEP 6: JITTER & SHIMMER
        # ============================================================
        print("   🔍 Computing voice stability metrics (jitter/shimmer)...")

        try:
            pp = parselmouth.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = float(parselmouth.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shimmer = float(parselmouth.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            print(f"   ✅ Jitter: {jitter:.6f} | Shimmer: {shimmer:.6f}")
        except Exception as e:
            print(f"   ⚠️  Jitter/shimmer failed (will use defaults): {e}")
            jitter = 0.02
            shimmer = 0.08

        # ============================================================
        # STEP 7: PAUSE DETECTION
        # ============================================================
        print("   ⏸️  Detecting pauses in speech...")

        pause_info = detect_pauses(raw, sr)
        print(f"   ✅ Detected {pause_info['num_pauses']} pauses | Pause ratio: {pause_info['pause_ratio']:.2%}")

        # ============================================================
        # STEP 8: COMPILE ALL 23 FEATURES
        # ============================================================
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

        print(f"   ✅ All 23 features extracted successfully")
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


# ============================================================
# STEP 8: API ENDPOINTS (Corrected)
# ============================================================

class AnalysisRequest(BaseModel):
    turn_id: int
    interview_id: str
    audio_path: str
    transcript: str = None  # Optional field

@app.post("/analyze-voice")
async def analyze_voice(
    request_data: AnalysisRequest,          # <--- FastAPI now looks for JSON body
    background_tasks: BackgroundTasks
):
    """
    Receive audio and transcript for analysis via JSON body.
    """
    try:
        # Access data using request_data.field_name
        print(f"📥 Received analysis request for Turn {request_data.turn_id}")

        # Queue background task and PASS the transcript
        background_tasks.add_task(
            process_voice_analysis,
            request_data.turn_id,
            request_data.audio_path,
            request_data.interview_id,
            datetime.now(),
            request_data.transcript
        )

        return {
            "status": "queued",
            "turn_id": request_data.turn_id,
            "message": "Voice analysis queued successfully.",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {"status": "error", "error": str(e)}, 500

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
    print("🚀 Starting FastAPI server...")
    print("Visit: http://localhost:8001/docs")
    print("="*60 + "\n")

    # Change "main:app" (string) to app (object) to fix the loop error
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001
    )