from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from upstash_redis import Redis
import logging
import shap
import opensmile
from pydub import AudioSegment
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================
models = {}

# ============================================================
# FEATURE KNOWLEDGE BASE
# Human-readable labels and coaching tips for every top-45 feature.
# SHAP identifies WHICH features drove the prediction; these dicts
# translate that into language candidates can actually act on.
# ============================================================
FEATURE_LABELS = {
    "loudnessPeaksPerSec":                        "Voice Energy Dynamics",
    "loudness_sma3_percentile20.0":               "Baseline Volume Level",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm":      "Upper Resonance Variability",
    "loudness_dynamics_power":                    "Dynamic Expression Power",
    "vocal_projection":                           "Vocal Projection Strength",
    "F3amplitudeLogRelF0_sma3nz_amean":           "Upper Vocal Resonance",
    "voiced_flow":                                "Speech Continuity",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm":      "Mid Resonance Variation",
    "F2bandwidth_sma3nz_stddevNorm":              "Articulation Precision",
    "F2amplitudeLogRelF0_sma3nz_amean":           "Mid Vocal Resonance",
    "loudness_sma3_meanRisingSlope":              "Vocal Attack (Energy Rise)",
    "loudness_sma3_stddevNorm":                   "Volume Variation",
    "mfcc4V_sma3nz_amean":                        "Voice Timbre Quality",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0": "Median Pitch",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0": "Peak Pitch Level",
    "F3bandwidth_sma3nz_amean":                   "Upper Formant Clarity",
    "mfcc2_sma3_amean":                           "Voice Spectral Shape",
    "mfcc2V_sma3nz_amean":                        "Voiced Spectral Quality",
    "slopeUV500-1500_sma3nz_amean":               "High-Frequency Presence",
    "mfcc1_sma3_stddevNorm":                      "Voice Quality Variation",
    "hammarbergIndexV_sma3nz_stddevNorm":         "Voice Quality Consistency",
    "alphaRatioUV_sma3nz_amean":                  "Spectral Energy Balance",
    "mfcc3_sma3_stddevNorm":                      "Mid-Frequency Tonal Variation",
    "StddevUnvoicedSegmentLength":                "Pause Pattern Regularity",
    "mfcc4_sma3_amean":                           "Upper Spectral Quality",
    "loudness_sma3_pctlrange0-2":                 "Loudness Dynamic Range",
    "F0semitoneFrom27.5Hz_sma3nz_amean":          "Average Pitch",
    "vocal_instability":                          "Voice Steadiness",
    "HNRdBACF_sma3nz_amean":                      "Voice Clarity",
    "spectralFlux_sma3_amean":                    "Vocal Expressiveness",
    "shimmerLocaldB_sma3nz_stddevNorm":           "Volume Stability Pattern",
    "mfcc1V_sma3nz_stddevNorm":                   "Core Vocal Quality Variation",
    "F1amplitudeLogRelF0_sma3nz_amean":           "Vowel Resonance Strength",
    "mfcc4V_sma3nz_stddevNorm":                   "Tonal Quality Variation",
    "mfcc4_sma3_stddevNorm":                      "Spectral Tonal Variation",
    "logRelF0-H1-H2_sma3nz_amean":               "Voice Breathiness / Tension",
    "spectralFluxV_sma3nz_stddevNorm":            "Expressive Variation",
    "equivalentSoundLevel_dBp":                   "Overall Sound Level",
    "F2frequency_sma3nz_stddevNorm":              "Vowel Quality Variation",
    "loudness_sma3_meanFallingSlope":             "Vocal Release",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":     "Pitch Variation",
    "slopeV500-1500_sma3nz_amean":                "Voice Brightness",
    "F1frequency_sma3nz_stddevNorm":              "Vowel Articulation Variation",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0": "Low-End Pitch",
    "shimmerLocaldB_sma3nz_amean":                "Volume Consistency",
}

# Positive = this feature pushed the score UP (good); negative = pushed DOWN (area to work on)
FEATURE_TIPS = {
    "loudnessPeaksPerSec": {
        "positive": "Your speech had strong, well-distributed energy peaks — emphasizing key points with volume variation signals confidence and keeps the interviewer engaged.",
        "negative": "Your speech lacked distinct energy peaks. Try consciously stressing key words and conclusions with a slight volume boost to sound more decisive and engaged.",
    },
    "loudness_sma3_percentile20.0": {
        "positive": "Your baseline volume was strong, staying audible even in quieter moments — this prevents the listener from straining to hear you.",
        "negative": "Your quieter moments dropped very soft. Maintain an audible baseline volume even when pausing to think — dropping too quiet signals uncertainty.",
    },
    "vocal_projection": {
        "positive": "Strong vocal projection — your voice carried confidently, making you sound present and authoritative throughout your answer.",
        "negative": "Your vocal projection was weaker than ideal. Try speaking as if addressing someone at the far end of a conference table — project from the diaphragm, not the throat.",
    },
    "voiced_flow": {
        "positive": "Excellent speech continuity — you spoke in smooth, connected segments without awkward fragmentation, signaling genuine fluency and solid preparation.",
        "negative": "Your speech had fragmented voiced segments with gaps between phrases. Using a simple structure (\"first... then... finally...\") before answering helps maintain a continuous, confident delivery.",
    },
    "vocal_instability": {
        "positive": "Your voice was steady with minimal micro-tremor — low pitch and volume fluctuations project composure and control under pressure.",
        "negative": "Vocal micro-tremor was detected (minute combined pitch and volume fluctuations). Taking a slow diaphragmatic breath before answering physically calms the nervous system and steadies the voice.",
    },
    "loudness_dynamics_power": {
        "positive": "Strong dynamic expression — your loudness range combined with energy peaks created an engaging, varied delivery that holds attention.",
        "negative": "Limited dynamic expression detected. Your loudness range was narrow with few energy peaks — deliberately vary your energy: louder for key points, softer for context.",
    },
    "HNRdBACF_sma3nz_amean": {
        "positive": "High voice clarity — your harmonics-to-noise ratio was strong, meaning your voice sounded clean and resonant rather than breathy or rough.",
        "negative": "Reduced voice clarity detected — a noisier voice signal sounds breathy or strained, which listeners associate with anxiety. Staying well-hydrated and doing light vocal warm-ups before interviews helps significantly.",
    },
    "spectralFlux_sma3_amean": {
        "positive": "Good vocal expressiveness — your voice changed dynamically across your response, making your delivery feel alive and invested rather than recited.",
        "negative": "Limited vocal expressiveness detected — a spectrally flat voice sounds monotone. Vary your pace, pitch contour, and emphasis on different parts of your answer.",
    },
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": {
        "positive": "Natural pitch variation — you used good intonation patterns that made your delivery sound conversational and engaged rather than rehearsed.",
        "negative": "Limited pitch variation detected — a flatter pitch contour sounds disengaged. Let your pitch naturally rise when introducing a point and fall conclusively at the end.",
    },
    "F0semitoneFrom27.5Hz_sma3nz_amean": {
        "positive": "Your average pitch was in a confident, natural conversational range that resonates well.",
        "negative": "Your average pitch was outside an optimal range — very high pitches can signal stress, very low pitches can reduce clarity. Recording practice sessions and listening back helps you calibrate.",
    },
    "StddevUnvoicedSegmentLength": {
        "positive": "Consistent pause lengths — your pauses followed a regular pattern, signaling structured thinking and deliberate pacing rather than stumbling.",
        "negative": "Irregular pause lengths detected — mixing very short and very long silences creates an uneven rhythm. Aim for purposeful pauses of 1-2 seconds max when transitioning between ideas.",
    },
    "shimmerLocaldB_sma3nz_amean": {
        "positive": "Consistent volume — your word-to-word loudness variation was low, projecting a controlled and steady delivery.",
        "negative": "Volume inconsistency detected — your loudness fluctuated between words more than ideal. Focus on maintaining steady breath support throughout each sentence.",
    },
    "loudness_sma3_pctlrange0-2": {
        "positive": "Good loudness range — you used a healthy spread of volume levels, giving your speech natural dynamics without sounding erratic.",
        "negative": "Very narrow loudness range — speaking with minimal volume variation flattens your delivery. Consciously vary your energy level across the answer for a more engaging effect.",
    },
    "logRelF0-H1-H2_sma3nz_amean": {
        "positive": "Balanced voice tension — your voice showed a healthy balance between breath and phonation, sounding natural and confident rather than strained or breathy.",
        "negative": "Voice tension imbalance detected — this can indicate breathiness (anxiety, insufficient closure) or pressed phonation (over-tension). Focus on relaxed, supported phonation with even breath flow.",
    },
    "equivalentSoundLevel_dBp": {
        "positive": "Good overall sound level — you spoke at a clearly audible volume throughout, making it easy to follow your answers.",
        "negative": "Your overall speaking volume was below optimal. Speaking slightly louder (as if the person is 2 meters away) projects more confidence and authority.",
    },
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Good upper resonance variation — variability in your upper harmonics adds richness and liveliness to your voice.",
        "negative": "Flat upper resonance — limited variation in your upper harmonics makes the voice sound less resonant. Opening up the back of the throat slightly while speaking can help.",
    },
    "F2amplitudeLogRelF0_sma3nz_amean": {
        "positive": "Strong mid vocal resonance — your second formant energy was well-projected, contributing to a clear and intelligible voice.",
        "negative": "Weak mid vocal resonance — low second formant amplitude makes your voice sound thinner. Working on open-mouth articulation helps strengthen this.",
    },
    "F2bandwidth_sma3nz_stddevNorm": {
        "positive": "Good articulation precision — consistent formant bandwidth variation indicates clear, well-formed vowels across your speech.",
        "negative": "Inconsistent articulation precision detected — variable formant bandwidths indicate unclear vowel formation. Slow down slightly and open your mouth more when articulating.",
    },
    "spectralFluxV_sma3nz_stddevNorm": {
        "positive": "Good expressive variation in voiced segments — your voice changed dynamically where it mattered most.",
        "negative": "Limited expressive variation in voiced speech — your voice stayed too uniform across words. Let your voice shape each idea differently.",
    },
    "loudness_sma3_meanRisingSlope": {
        "positive": "Strong vocal attack — your energy rises quickly when starting phrases, which sounds assertive and purposeful.",
        "negative": "Weak vocal attack — your energy was slow to build at the start of phrases. Beginning phrases with slightly more energy makes you sound more confident.",
    },
    "loudness_sma3_stddevNorm": {
        "positive": "Natural volume variation — your loudness varied in a healthy way that sounds expressive rather than flat.",
        "negative": "Unusual volume variation pattern — either too flat or too erratic. Aim for natural speech rhythms where key words have slightly more energy.",
    },
}


def get_generic_tip(label: str, is_positive: bool) -> str:
    """Fallback tip for features not in FEATURE_TIPS."""
    if is_positive:
        return f"Your {label.lower()} contributed positively to your confidence score — this is a vocal strength to maintain."
    return f"Your {label.lower()} reduced your confidence score. Targeted vocal practice focusing on this dimension can improve future performance."


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("VOICE ANALYSIS SERVICE STARTING...")
    print("="*60)
    try:
        print("Loading models and initializing tools...")

        models['final_model'] = joblib.load("models/xgboost_final_engineered_45features.pkl")
        models['top_features'] = joblib.load("models/top_45_features_engineered.pkl")
        print(f"   XGBoost model loaded — {len(models['top_features'])} features")

        models['smile'] = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("   openSMILE eGeMAPSv02 initialized")

        models['shap_explainer'] = shap.TreeExplainer(models['final_model'])
        print("   SHAP TreeExplainer ready")

        print("\nSERVICE READY!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        raise

    yield

    print("\nVoice analysis service shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Hirely Voice Analysis Service",
    description="Analyzes candidate voice confidence from interview audio",
    lifespan=lifespan
)

# Upstash Redis (REST API — not standard redis library)
try:
    redis_client = Redis(
        url=os.getenv("REDIS_URL"),
        token=os.getenv("REDIS_TOKEN"),
    )
    redis_client.ping()
    print("Upstash Redis connected")
except Exception as e:
    print(f"Upstash Redis connection failed: {e}")
    redis_client = None


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_opensmile_features(filepath: str) -> pd.DataFrame:
    """
    Extract eGeMAPSv02 functionals using openSMILE.
    Falls back to a WAV-converted temp file if direct processing fails
    (handles WebM/Opus audio from the browser recorder).
    """
    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    logger.info(f"Extracting openSMILE features: {fpath.name}")

    try:
        feats = models['smile'].process_file(str(fpath))
    except Exception as e:
        logger.warning(f"Direct openSMILE processing failed ({e}). Converting to WAV...")
        tmp_path = None
        try:
            tmp_path = tempfile.mktemp(suffix=".wav")
            audio = AudioSegment.from_file(str(fpath))
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_path, format="wav")
            feats = models['smile'].process_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return pd.DataFrame([feats.iloc[0].to_dict()])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7 engineered features on top of the raw eGeMAPS functionals.
    Only 4 of these (vocal_instability, vocal_projection, voiced_flow,
    loudness_dynamics_power) appear in the top-45 model features, but
    all 7 are computed here to exactly match training.
    """
    cols = df.columns.tolist()

    if all(c in cols for c in ['jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean']):
        df['vocal_instability'] = (
            df['jitterLocal_sma3nz_amean'] * df['shimmerLocaldB_sma3nz_amean']
        )

    if all(c in cols for c in ['loudness_sma3_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm']):
        df['vocal_projection'] = (
            df['loudness_sma3_amean'] / (df['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] + 0.05)
        )

    if all(c in cols for c in ['F1frequency_sma3nz_amean', 'F2frequency_sma3nz_amean']):
        df['resonance_ratio'] = (
            df['F2frequency_sma3nz_amean'] / (df['F1frequency_sma3nz_amean'] + 1e-4)
        )

    if all(c in cols for c in ['HNRdBACF_sma3nz_amean', 'alphaRatioV_sma3nz_amean']):
        df['voice_quality'] = df['HNRdBACF_sma3nz_amean'] - df['alphaRatioV_sma3nz_amean']

    if all(c in cols for c in ['loudness_sma3_pctlrange0-2', 'loudnessPeaksPerSec']):
        df['loudness_dynamics_power'] = (
            df['loudness_sma3_pctlrange0-2'] * df['loudnessPeaksPerSec']
        )

    if all(c in cols for c in ['VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec']):
        df['voiced_flow'] = (
            df['VoicedSegmentsPerSec'] * df['MeanVoicedSegmentLengthSec']
        )

    if all(c in cols for c in ['slopeV0-500_sma3nz_amean', 'slopeV500-1500_sma3nz_amean']):
        df['brightness_contrast'] = (
            df['slopeV500-1500_sma3nz_amean'] - df['slopeV0-500_sma3nz_amean']
        )

    return df


def compute_wpm(audio_path: str, transcript_text: str | None) -> float:
    """Compute words-per-minute from transcript and audio duration (display metric only)."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000.0
        if transcript_text and transcript_text.strip() and duration_sec > 0:
            word_count = len(transcript_text.strip().split())
            return round((word_count / duration_sec) * 60, 1)
        return 0.0
    except Exception:
        return 0.0


# ============================================================
# SHAP EXPLANATION ENGINE
# ============================================================
def generate_shap_explanations(X: pd.DataFrame, predicted_score: float) -> dict:
    """
    Compute per-prediction SHAP values using TreeExplainer and convert
    the top 5 contributors into actionable coaching advice.

    Each contributor shows:
      - which feature mattered most for THIS prediction
      - whether it helped or hurt the score
      - a concrete coaching tip the candidate can act on
    """
    try:
        explainer = models['shap_explainer']
        shap_values = explainer.shap_values(X)   # shape: (1, 45) for regression
        # expected_value is ndarray([scalar]) for XGBRegressor — extract the float
        ev = explainer.expected_value
        base_value = float(ev[0]) if hasattr(ev, '__len__') else float(ev)

        contributors = []
        for i, feat_name in enumerate(models['top_features']):
            sv = float(shap_values[0][i])
            fval = float(X.iloc[0, i])
            label = FEATURE_LABELS.get(feat_name, feat_name)
            direction = "positive" if sv > 0 else "negative"
            tip = FEATURE_TIPS.get(feat_name, {}).get(
                direction, get_generic_tip(label, sv > 0)
            )
            contributors.append({
                "feature": feat_name,
                "label": label,
                "value": round(fval, 4),
                "shap_value": round(sv, 4),
                "direction": "increased" if sv > 0 else "decreased",
                "impact_magnitude": round(abs(sv), 4),
                "explanation": tip,
            })

        contributors.sort(key=lambda x: x["impact_magnitude"], reverse=True)

        return {
            "base_value": round(base_value, 4),
            "predicted_value": round(predicted_score, 4),
            "top_contributors": contributors[:5],
            "all_shap_values": {
                feat_name: round(float(shap_values[0][i]), 4)
                for i, feat_name in enumerate(models['top_features'])
            },
        }

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return {
            "base_value": 0.0,
            "predicted_value": predicted_score,
            "top_contributors": [],
            "all_shap_values": {},
        }


# ============================================================
# PREDICTION PIPELINE
# ============================================================
def score_to_label(score: float) -> str:
    """Map 0-1 regression output to a text label (used in Gemini prompts and report)."""
    if score >= 0.75:
        return "Highly Confident"
    if score >= 0.50:
        return "Confident"
    if score >= 0.30:
        return "Moderately Confident"
    return "Needs Improvement"


def predict_confidence(audio_path: str, transcript_text: str | None = None) -> dict:
    """
    Full pipeline:
      1. Extract eGeMAPSv02 features via openSMILE
      2. Add 7 engineered features
      3. Select the model's top 45 features
      4. XGBoost regression → 0-1 confidence score
      5. SHAP explanation of what drove the score
      6. WPM from transcript (display metric)
    """
    logger.info(f"Starting voice analysis: {audio_path}")

    # 1 & 2 — feature extraction
    features_df = extract_opensmile_features(audio_path)
    features_df = add_engineered_features(features_df)

    # 3 — select top 45 in training order, fill any missing with 0
    top_features = models['top_features']
    X = pd.DataFrame(index=[0], columns=top_features, dtype=float)
    for col in top_features:
        X[col] = features_df[col].iloc[0] if col in features_df.columns else 0.0

    # 4 — predict (no scaler needed, XGBoost is scale-invariant)
    raw_score = float(models['final_model'].predict(X)[0])
    confidence_score = float(np.clip(raw_score, 0.0, 1.0))

    # 5 — SHAP explanations
    shap_result = generate_shap_explanations(X, confidence_score)

    # 6 — WPM (display only)
    wpm = compute_wpm(audio_path, transcript_text)

    return {
        "confidence_score": confidence_score,
        "confidence_label": score_to_label(confidence_score),
        "wpm": wpm,
        "shap": shap_result,
        "raw_features": features_df.iloc[0].to_dict(),
    }


# ============================================================
# BACKGROUND TASK
# ============================================================
async def process_voice_analysis(
    turn_id: int,
    audio_path: str,
    interview_id: str,
    start_time: datetime,
    transcript_text: str | None = None,
):
    """
    Runs asynchronously after Express queues it.
    Builds the full result payload that matches the Prisma VoiceAnalysis
    schema and saves it to Redis for Express to pick up.
    """
    try:
        print(f"\n{'='*60}")
        print(f"VOICE ANALYSIS — Turn {turn_id} | Session {interview_id}")
        print(f"{'='*60}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prediction = predict_confidence(audio_path, transcript_text)
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        raw = prediction["raw_features"]
        shap = prediction["shap"]

        # Derive backward-compatible schema fields from eGeMAPS features
        # voiced_flow ≈ fraction of time voiced → pause_ratio ≈ 1 - voiced_flow
        vsps = raw.get("VoicedSegmentsPerSec", 1.0) or 1.0
        mvsl = raw.get("MeanVoicedSegmentLengthSec", 0.3) or 0.3
        voiced_fraction = min(1.0, float(vsps) * float(mvsl))
        pause_ratio_approx = max(0.0, round(1.0 - voiced_fraction, 4))

        result = {
            "interviewTurnId": turn_id,

            # Core model output
            "confidenceLevel": round(prediction["confidence_score"], 4),       # 0-1
            "confidenceLabelText": prediction["confidence_label"],              # for Gemini/report

            # Derived quality metrics (backward-compatible with Prisma schema)
            "speakingQuality": round(prediction["confidence_score"], 4),
            "vocalStability": round(
                max(0.0, 1.0 - float(raw.get("jitterLocal_sma3nz_amean", 0) or 0)), 4
            ),
            "speakingFluency": round(voiced_fraction, 4),
            "pitchMean": round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0) or 0), 4),
            "pitchStd": round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0) or 0), 4),
            "energyLevel": round(float(raw.get("equivalentSoundLevel_dBp", 0) or 0), 4),
            "wordsPerMinute": prediction["wpm"],
            "pauseRatio": pause_ratio_approx,
            "jitter": round(float(raw.get("jitterLocal_sma3nz_amean", 0) or 0), 6),
            "shimmer": round(float(raw.get("shimmerLocaldB_sma3nz_amean", 0) or 0), 6),

            "modelVersion": "v2.0-egemaps-shap",

            # SHAP values stored in allProbabilities for DB (Json field)
            "allProbabilities": shap["all_shap_values"],

            # rawFeatures includes eGeMAPS values + SHAP explanations
            "rawFeatures": {
                **{k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                   for k, v in raw.items()},
                "shapExplanations": shap["top_contributors"],
                "shapBaseValue": shap["base_value"],
                # featureExplanations for existing frontend "Why this prediction" section
                "featureExplanations": [
                    c["explanation"]
                    for c in shap["top_contributors"]
                    if c.get("explanation")
                ],
            },

            "status": "completed",
            "processingTimeMs": elapsed_ms,
            "processedAt": datetime.now().isoformat(),
        }

        print(f"Confidence: {result['confidenceLevel']*100:.1f}% ({result['confidenceLabelText']})")
        print(f"WPM: {result['wordsPerMinute']}")
        print(f"Top SHAP driver: {shap['top_contributors'][0]['label'] if shap['top_contributors'] else 'N/A'}")
        print(f"Processing time: {elapsed_ms}ms")
        print(f"{'='*60}\n")

        if redis_client:
            try:
                redis_client.set(
                    f"voice_analysis:{interview_id}:{turn_id}",
                    json.dumps(result),
                    ex=86400,
                )
                logger.info(f"Saved to Redis: voice_analysis:{interview_id}:{turn_id}")
            except Exception as e:
                logger.warning(f"Redis save failed (non-critical): {e}")

        return result

    except Exception as e:
        print(f"\nVOICE ANALYSIS FAILED — Turn {turn_id}: {e}\n")
        logger.error(f"process_voice_analysis error: {e}")

        error_result = {
            "interviewTurnId": turn_id,
            "status": "failed",
            "errorMessage": str(e),
            "processedAt": datetime.now().isoformat(),
        }

        if redis_client:
            try:
                redis_client.set(
                    f"voice_analysis:{interview_id}:{turn_id}",
                    json.dumps(error_result),
                    ex=86400,
                )
            except Exception:
                pass


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Hirely Voice Analysis v2.0",
        "models_loaded": bool(models.get("final_model")),
        "shap_ready": bool(models.get("shap_explainer")),
        "timestamp": datetime.now().isoformat(),
    }


class AnalysisRequest(BaseModel):
    turn_id: int
    interview_id: str
    audio_path: str
    transcript: str | None = None


@app.post("/analyze-voice")
async def analyze_voice(request_data: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Accepts JSON body from Express backend.
    Queues voice analysis as a background task and returns immediately.
    Express polls /result/{interview_id}/{turn_id} for the result.
    """
    try:
        logger.info(f"Queued voice analysis — Turn {request_data.turn_id}")
        background_tasks.add_task(
            process_voice_analysis,
            request_data.turn_id,
            request_data.audio_path,
            request_data.interview_id,
            datetime.now(),
            request_data.transcript,
        )
        return {
            "status": "queued",
            "turn_id": request_data.turn_id,
            "message": "Voice analysis queued.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {"status": "error", "error": str(e)}, 500


@app.get("/result/{interview_id}/{turn_id}")
def get_result(interview_id: str, turn_id: int):
    """Retrieve voice analysis result from Redis. Returns {status: pending} if not ready."""
    try:
        if not redis_client:
            return {"status": "error", "error": "Redis not connected"}
        result = redis_client.get(f"voice_analysis:{interview_id}:{turn_id}")
        if result is None:
            return {"status": "pending"}
        return json.loads(result) if isinstance(result, str) else result
    except Exception as e:
        logger.error(f"Get result failed: {e}")
        return {"status": "error", "error": str(e)}, 500


# ============================================================
# DEVELOPMENT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Hirely Voice Analysis Service v2.0")
    print("Docs: http://localhost:8001/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
