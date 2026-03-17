from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import asyncio
from redis import Redis
import logging
from pydub import AudioSegment
import tempfile
from pathlib import Path
import opensmile
import shap
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE (loaded models & openSMILE)
# ============================================================
models = {}
smile = None  # will initialize in lifespan

# ============================================================
# STARTUP / SHUTDOWN (lifespan)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global smile
    print("\n" + "="*60)
    print("🚀 VOICE ANALYSIS SERVICE STARTING...")
    print("="*60)
    try:
        print("⏳ Loading models and openSMILE...")
        # Load your XGBoost model & top features
        models['final_model'] = joblib.load("xgboost_final_engineered_45features.pkl")
        models['top_features'] = joblib.load("top_45_features_engineered.pkl")
        print(f"✅ Model loaded - using {len(models['top_features'])} features")

        # Initialize openSMILE (done once at startup)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("✅ openSMILE initialized")

        print("\n✅ SERVICE READY!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        raise

    yield  # App runs here

    # Shutdown
    print("\n⏹️ Voice analysis service shutting down...")

# ============================================================
# CREATE FASTAPI APP
# ============================================================
app = FastAPI(
    title="Hirely Voice Analysis Service",
    description="Analyzes candidate voice confidence from interview audio",
    lifespan=lifespan
)

# Redis connection
try:
    redis_client = Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True
    )
    redis_client.ping()
    print("✅ Redis connected")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    print("Continuing without Redis (results won't be queued)")

# ============================================================
# FEATURE EXTRACTION (your current pipeline)
# ============================================================
def extract_single_file(filepath: str):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    logger.info(f"Processing audio: {filepath.name}")

    try:
        feats = smile.process_file(str(filepath))
    except Exception as e:
        logger.warning(f"Direct processing failed: {e}. Converting to WAV...")
        tmp_path = None
        try:
            tmp_path = tempfile.mktemp(suffix=".wav")
            audio = AudioSegment.from_file(filepath, format=filepath.suffix.lstrip("."))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_path, format="wav")
            feats = smile.process_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete temp file: {cleanup_err}")

    df = pd.DataFrame([feats.iloc[0].to_dict()])
    df["file"] = filepath.name
    return df

def add_engineered_features(df):
    logger.info("Adding engineered vocal features...")
    
    if all(col in df for col in ['jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean']):
        df['vocal_instability'] = df['jitterLocal_sma3nz_amean'] * df['shimmerLocaldB_sma3nz_amean']

    if all(col in df for col in ['loudness_sma3_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm']):
        df['vocal_projection'] = df['loudness_sma3_amean'] / (df['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] + 0.05)

    if all(col in df for col in ['F1frequency_sma3nz_amean', 'F2frequency_sma3nz_amean']):
        df['resonance_ratio'] = df['F2frequency_sma3nz_amean'] / (df['F1frequency_sma3nz_amean'] + 1e-4)

    if all(col in df for col in ['HNRdBACF_sma3nz_amean', 'alphaRatioV_sma3nz_amean']):
        df['voice_quality'] = df['HNRdBACF_sma3nz_amean'] - df['alphaRatioV_sma3nz_amean']

    if all(col in df for col in ['loudness_sma3_pctlrange0-2', 'loudnessPeaksPerSec']):
        df['loudness_dynamics_power'] = df['loudness_sma3_pctlrange0-2'] * df['loudnessPeaksPerSec']

    if all(col in df for col in ['VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec']):
        df['voiced_flow'] = df['VoicedSegmentsPerSec'] * df['MeanVoicedSegmentLengthSec']

    if all(col in df for col in ['slopeV0-500_sma3nz_amean', 'slopeV500-1500_sma3nz_amean']):
        df['brightness_contrast'] = df['slopeV500-1500_sma3nz_amean'] - df['slopeV0-500_sma3nz_amean']

    return df

# ============================================================
# PREDICTION FUNCTION (adapted to your current model)
# ============================================================
def predict_confidence(audio_path: str, transcript_text: str = None):
    try:
        logger.info(f"Starting voice analysis for audio: {audio_path}")

        # 1. Extract eGeMAPS features
        features_df = extract_single_file(audio_path)
        
        # 2. Add engineered features (your current logic)
        features_df = add_engineered_features(features_df)

        # 3. Select only the top features used in training
        X_new = features_df[models['top_features']]

        # Fill any missing columns with 0 (safety)
        for col in models['top_features']:
            if col not in X_new.columns:
                X_new[col] = 0.0

        # 4. Predict (your XGBoost model)
        confidence_score = models['final_model'].predict(X_new)[0]

        # 5. Prepare result (adapted to your format)
        result = {
            "confidence_score": float(confidence_score),  # 0–1 range
            "status": "completed",
            "processed_at": datetime.now().isoformat(),
            "processing_time_ms": 2500,  # placeholder - can measure if needed
            "raw_features": features_df.iloc[0].to_dict()  # optional
        }

        logger.info(f"Prediction successful: confidence = {confidence_score:.4f}")
        return result

    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "processed_at": datetime.now().isoformat()
        }

# ============================================================
# BACKGROUND TASK
# ============================================================
async def process_voice_analysis(
    turn_id: int,
    audio_path: str,
    interview_id: str,
    start_time: datetime,
    transcript_text: str = None
):
    try:
        result = predict_confidence(audio_path, transcript_text)
        
        # Save to Redis
        try:
            redis_client.set(
                f"voice_analysis:{interview_id}:{turn_id}",
                json.dumps(result),
                ex=86400
            )
            logger.info(f"Results saved to Redis for turn {turn_id}")
        except Exception as redis_err:
            logger.warning(f"Redis save failed: {redis_err}")

    except Exception as e:
        logger.error(f"Background task failed: {e}")

# ============================================================
# API ENDPOINTS (unchanged from your group member's version)
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Hirely Voice Analysis",
        "models_loaded": bool(models.get('final_model')),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze-voice")
async def analyze_voice(
    background_tasks: BackgroundTasks,
    turn_id: int,
    interview_id: str,
    audio_path: str,
    transcript: str = None  # from Express
):
    try:
        background_tasks.add_task(
            process_voice_analysis,
            turn_id,
            audio_path,
            interview_id,
            datetime.now(),
            transcript
        )
        return {
            "status": "queued",
            "turn_id": turn_id,
            "message": "Voice analysis queued.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {"status": "error", "error": str(e)}, 500

@app.get("/result/{interview_id}/{turn_id}")
def get_result(interview_id: str, turn_id: int):
    try:
        result = redis_client.get(f"voice_analysis:{interview_id}:{turn_id}")
        if result is None:
            return {"status": "pending"}
        return json.loads(result)
    except Exception as e:
        logger.error(f"Get result failed: {e}")
        return {"status": "error", "error": str(e)}, 500

# ============================================================
# RUN (development)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting FastAPI Voice Analysis Server...")
    print("Visit: http://localhost:8001/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)i