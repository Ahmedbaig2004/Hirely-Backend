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
    # --- Energy / Volume ---
    "loudnessPeaksPerSec":                        "Passion & Emphasis",
    "loudness_sma3_percentile20.0":               "Quiet-Moment Volume",
    "loudness_dynamics_power":                    "Dynamic Range",
    "vocal_projection":                           "Room-Filling Power",
    "loudness_sma3_meanRisingSlope":              "Starting Strength",
    "loudness_sma3_stddevNorm":                   "Volume Variety",
    "loudness_sma3_pctlrange0-2":                 "Whisper-to-Shout Range",
    "loudness_sma3_meanFallingSlope":             "Sentence Endings",
    "loudness_sma3_stddevRisingSlope":            "Attack Variation",
    # --- Fluency / Steadiness ---
    "voiced_flow":                                "Talking in Flow",
    "vocal_instability":                          "Nervousness Meter",
    "StddevUnvoicedSegmentLength":                "Pause Consistency",
    "shimmerLocaldB_sma3nz_stddevNorm":           "Volume Wobble",
    "MeanVoicedSegmentLengthSec":                 "Phrase Length",
    # --- Clarity / Crispness ---
    "HNRdBACF_sma3nz_amean":                      "Voice Smoothness",
    "alphaRatioUV_sma3nz_amean":                  "Consonant Crispness",
    "slopeUV500-1500_sma3nz_amean":               "Speech Crispness",
    "F2bandwidth_sma3nz_stddevNorm":              "Mouth Movement Consistency",
    "F3bandwidth_sma3nz_stddevNorm":              "Pronunciation Consistency",
    # --- Pitch / Intonation ---
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":     "Pitch Movement",
    # --- Breath & Tension ---
    "logRelF0-H1-H2_sma3nz_stddevNorm":          "Breath Control Variation",
    # --- Expressiveness ---
    "spectralFluxV_sma3nz_stddevNorm":            "Expressive Variety",
    "hammarbergIndexUV_sma3nz_amean":             "Spectral Tilt",
    # --- Resonance ---
    "F3amplitudeLogRelF0_sma3nz_stddevNorm":      "Voice Richness Variety",
    "F3amplitudeLogRelF0_sma3nz_amean":           "Voice Richness",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm":      "Vowel Power Variety",
    "F2amplitudeLogRelF0_sma3nz_amean":           "Vowel Power",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm":      "Resonance Variation",
    "F2frequency_sma3nz_stddevNorm":              "Vowel Clarity Range",
    "slopeV500-1500_sma3nz_amean":                "Voice Brightness",
    "F3frequency_sma3nz_stddevNorm":              "Upper Resonance Movement",
    # --- Voice Quality ---
    "mfcc1_sma3_stddevNorm":                      "Voice Texture Variety",
    "mfcc1V_sma3nz_stddevNorm":                   "Core Voice Variety",
    "mfcc2_sma3_stddevNorm":                      "Voice Character Variety",
    "mfcc2V_sma3nz_stddevNorm":                   "Voice Warmth Variety",
    "mfcc3_sma3_stddevNorm":                      "Tonal Variety",
    "mfcc4_sma3_stddevNorm":                      "Tone Consistency",
    "mfcc4V_sma3nz_stddevNorm":                   "Tone Steadiness",
    "hammarbergIndexV_sma3nz_stddevNorm":         "Voice Quality Steadiness",
    "hammarbergIndexV_sma3nz_amean":              "Voice Fullness",
    "alphaRatioV_sma3nz_amean":                   "Spectral Balance",
    "alphaRatioV_sma3nz_stddevNorm":              "Spectral Balance Variation",
    "slopeV0-500_sma3nz_stddevNorm":              "Low-Frequency Variation",
    "slopeV0-500_sma3nz_amean":                   "Voice Warmth",
    "brightness_contrast":                        "Brightness vs Warmth Balance",
}

# ============================================================
# COACHING TIPS — Problem → Why it matters → The Drill
# Every tip must describe a PHYSICAL ACTION the user can do.
# No acoustic jargon. If a user can't visualize the body part
# they need to move, the feedback is useless.
# ============================================================
FEATURE_TIPS = {
    "loudnessPeaksPerSec": {
        "positive": "You punched the important words — that's how confident speakers highlight key points. Keep doing it.",
        "negative": "Your voice stayed at one flat volume the entire time. Interviewers hear monotone as 'bored' or 'unsure.' The fix: Pick 3 key words in your next answer and say them 20% louder than everything else.",
    },
    "loudness_sma3_percentile20.0": {
        "positive": "Even in your quieter moments, your voice stayed clearly audible — that projects steadiness and control.",
        "negative": "When you paused to think, your volume dropped to a near-whisper. Trailing off signals uncertainty to interviewers. The fix: Take a belly breath before answering, and imagine someone is 3 meters away — never drop below 'conversation volume.'",
    },
    "vocal_projection": {
        "positive": "Your voice filled the room — you sounded present and authoritative without shouting.",
        "negative": "Your voice sounded thin, like you were talking to yourself. Weak projection reads as low confidence. The fix: Sit up straight, breathe from your belly, and speak as if addressing someone across a conference table.",
    },
    "voiced_flow": {
        "positive": "You spoke in smooth, connected sentences — no choppy fragments. That signals preparation and genuine fluency.",
        "negative": "Your sentences kept breaking into short, choppy bursts — it sounds like you're making it up on the spot. The fix: Before answering, mentally outline 'First... then... finally...' and speak each chunk as one continuous breath.",
    },
    "vocal_instability": {
        "positive": "Your voice was rock-steady — no shaking or wavering. That projects composure under pressure.",
        "negative": "Your voice had a slight shake or wobble — vocal tremor is the #1 thing interviewers subconsciously associate with nervousness. The fix: Before speaking, exhale slowly for 4 seconds through pursed lips. This physically calms the vocal cords.",
    },
    "loudness_dynamics_power": {
        "positive": "Great dynamic range — you varied between quieter context and louder key points, keeping the listener engaged.",
        "negative": "Your volume barely changed from start to finish — a flat delivery puts listeners on autopilot. The fix: Practice saying 'The MOST important thing is...' — whisper 'the most important thing is' then hit the key word with full volume.",
    },
    "HNRdBACF_sma3nz_amean": {
        "positive": "Your voice sounded clean and smooth — no breathiness or roughness. Easy and pleasant to listen to.",
        "negative": "Your voice sounded breathy or rough, like you were running out of air. A strained voice is harder to follow. The fix: Drink water beforehand, hum for 10 seconds to warm up your vocal cords, and support each sentence with a full breath.",
    },
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": {
        "positive": "Natural pitch movement — your voice rose and fell in conversational patterns that sound engaging and authentic.",
        "negative": "Your pitch barely moved — every sentence sounded the same. Flat pitch = monotone = boring. The fix: Let your pitch rise when introducing a new point, and drop it firmly when concluding. Practice: 'What I learned was... (voice up) that testing matters. (voice down, firm).'",
    },
    "StddevUnvoicedSegmentLength": {
        "positive": "Your pauses were consistent and purposeful — it sounded like structured thinking, not stumbling.",
        "negative": "Your pauses were all over the place — some too short, some awkwardly long. Erratic pauses make you sound unsure. The fix: Use deliberate 1-second pauses between ideas. Count 'one-Mississippi' silently, then continue.",
    },
    "shimmerLocaldB_sma3nz_stddevNorm": {
        "positive": "Steady volume control — your voice maintained consistent power across words, sounding composed.",
        "negative": "Your volume wobbled unpredictably between words — this is perceived as nervousness. The fix: Practice sustaining 'ahhh' at one volume for 5 seconds, then count 'one... two... three...' keeping the same volume on each number.",
    },
    "loudness_sma3_pctlrange0-2": {
        "positive": "Good whisper-to-shout range — you used a healthy spread of volume that kept your delivery dynamic.",
        "negative": "You spoke at one volume the entire time with almost no range — no dynamic range = monotone delivery. The fix: Practice reading a paragraph where you deliberately start quiet, build to medium, and finish strong.",
    },
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Good voice richness variety — your voice had depth and liveliness that keeps listeners engaged.",
        "negative": "Your voice sounded flat and thin — lacking richness. The fix: Open the back of your throat slightly, like the start of a yawn, while speaking. This adds natural warmth and depth to your voice.",
    },
    "F2amplitudeLogRelF0_sma3nz_amean": {
        "positive": "Strong vowel projection — your vowels were well-formed and resonant, making you clearly intelligible.",
        "negative": "Your vowels sounded weak and swallowed — words blurred together. The fix: Over-pronounce vowels when practicing: 'I wOrkEd On A prOjEct...' Exaggerate first, then dial it back 50% for the real interview.",
    },
    "F2bandwidth_sma3nz_stddevNorm": {
        "positive": "Precise mouth movements — your articulation was consistent, producing clear and well-formed sounds.",
        "negative": "Your articulation was inconsistent — some words clear, others mumbled. Sloppy articulation sounds unprepared. The fix: Practice tongue twisters for 30 seconds before the interview — 'Red lorry, yellow lorry' or 'Unique New York.' This wakes up your mouth muscles.",
    },
    "alphaRatioUV_sma3nz_amean": {
        "positive": "Crisp consonants — your 'T's, 'S's, and 'K's were sharp and clear, making every word distinct.",
        "negative": "Your consonants were soft and mushy — words blurred together. The fix: Exaggerate your 'T' and 'K' sounds when practicing. Tap the tip of your tongue hard against the roof of your mouth for each 'T.'",
    },
    "slopeUV500-1500_sma3nz_amean": {
        "positive": "Clear and crisp speech — your words had good bite and definition to them.",
        "negative": "Your speech lacked crispness — it sounded dull and muffled, especially over video calls. The fix: Smile slightly while speaking — this naturally raises the front of your tongue and adds brightness and clarity to your voice.",
    },
    "spectralFluxV_sma3nz_stddevNorm": {
        "positive": "Good expressive variety — your voice shaped each idea differently, keeping the listener engaged.",
        "negative": "Your voice stayed too uniform — every word sounded the same. The fix: Pick one word per sentence to emphasize differently — louder, slower, or higher pitch. Just one word changes the whole feel.",
    },
    "loudness_sma3_meanRisingSlope": {
        "positive": "Strong sentence starts — you hit each phrase with energy right from the first word. Sounds decisive.",
        "negative": "You faded into sentences — the first few words were barely audible. Weak openings sound hesitant. The fix: Start every sentence with a small burst of energy. Don't idle into your point — accelerate into it.",
    },
    "loudness_sma3_stddevNorm": {
        "positive": "Natural volume variation — your loudness rose and fell in ways that felt organic and engaging.",
        "negative": "Your volume pattern was either dead-flat or erratic — neither sounds natural. The fix: Record yourself telling a friend a story. Notice how your volume naturally moves. Mirror that same energy in interview answers.",
    },
}


def get_generic_tip(label: str, is_positive: bool) -> str:
    """Fallback tip for features not in FEATURE_TIPS."""
    if is_positive:
        return f"Your {label.lower()} was a strength — keep doing what you're doing."
    return f"Your {label.lower()} needs work. Record yourself practicing and listen back — awareness is the first step to improvement."


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

        models['final_model'] = joblib.load("models/xgboost_fair_model.pkl")
        models['top_features'] = joblib.load("models/top_features_fair.pkl")
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
            # If the impact is less than 0.5% (0.005), skip this feature.
            # This prevents the AI from giving advice on "noise."
            if abs(sv) < 0.005:
                continue
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
# SHAP-DRIVEN CATEGORY SENTIMENTS
# ============================================================
SHAP_CATEGORIES = {
    "fluency": {
        "label": "Fluency",
        "features": [
            "voiced_flow",                          # #6
            "vocal_instability",                    # #24
            "shimmerLocaldB_sma3nz_stddevNorm",     # #17
            "StddevUnvoicedSegmentLength",          # #22
            "MeanVoicedSegmentLengthSec",           # #32
        ],
    },
    "energy": {
        "label": "Energy",
        "features": [
            "loudnessPeaksPerSec",                  # #1
            "loudness_sma3_percentile20.0",         # #2
            "loudness_dynamics_power",              # #4
            "vocal_projection",                     # #5
            "loudness_sma3_meanRisingSlope",        # #11
            "loudness_sma3_stddevNorm",             # #9
        ],
    },
    "clarity": {
        "label": "Clarity",
        "features": [
            "F2bandwidth_sma3nz_stddevNorm",        # #8
            "slopeUV500-1500_sma3nz_amean",         # #12
            "HNRdBACF_sma3nz_amean",                # #13
            "F2frequency_sma3nz_stddevNorm",        # #16
            "alphaRatioUV_sma3nz_amean",            # #20
        ],
    },
}


def _build_category_sentiments(all_shap_values: dict, wpm: float = 0.0) -> dict:
    """
    For each category, sum the SHAP contributions of its features,
    determine status, and identify the single biggest driver with
    its human-readable label and coaching tip.
    Also adds a 4th 'pace' category based on WPM thresholds.
    """
    categories = {}
    for key, cat in SHAP_CATEGORIES.items():
        feat_shaps = {f: all_shap_values.get(f, 0.0) for f in cat["features"]}
        shap_sum = sum(feat_shaps.values())

        if shap_sum > 0.01:
            status = "Good"
        elif shap_sum < -0.01:
            status = "Needs Improvement"
        else:
            status = "Neutral"

        top_feat = max(feat_shaps, key=lambda f: abs(feat_shaps[f]))
        top_sv = feat_shaps[top_feat]
        label = FEATURE_LABELS.get(top_feat, top_feat)

        if top_sv > 0:
            direction = "positive"
        elif top_sv < 0:
            direction = "negative"
        else:
            direction = "neutral"

        tip = FEATURE_TIPS.get(top_feat, {}).get(
            direction if direction != "neutral" else "positive",
            get_generic_tip(label, top_sv >= 0),
        )

        categories[key] = {
            "label": cat["label"],
            "status": status,
            "shap_sum": round(shap_sum, 4),
            "top_driver": {
                "feature": top_feat,
                "label": label,
                "shap_value": round(top_sv, 4),
                "direction": direction,
                "tip": tip,
            },
        }
    # --- 4th Box: Speaking Pace (WPM-based, not SHAP) ---
    if wpm > 0:
        if wpm < 110:
            pace_status = "Needs Improvement"
            pace_tip = f"You're speaking at {wpm:.0f} WPM. That's a bit slow. Aim for a brisker pace to keep the interviewer engaged."
        elif wpm > 170:
            pace_status = "Needs Improvement"
            pace_tip = f"You're racing at {wpm:.0f} WPM! Slow down slightly and use pauses to let your key points sink in."
        else:
            pace_status = "Good"
            pace_tip = f"Your pace of {wpm:.0f} WPM is perfect. You sound calm, professional, and easy to follow."

        categories["pace"] = {
            "label": "Speaking Pace",
            "status": pace_status,
            "wpm": round(wpm, 1),
            "top_driver": {
                "feature": "wpm",
                "label": "Words Per Minute",
                "shap_value": 0.0,
                "direction": "neutral",
                "tip": pace_tip,
            },
        }

    # "Rule of One" — find the single most negative SHAP feature across ALL categories
    worst_feat = None
    worst_sv = 0.0
    worst_category = None
    for key, cat_data in categories.items():
        driver = cat_data["top_driver"]
        if driver["shap_value"] < worst_sv:
            worst_sv = driver["shap_value"]
            worst_feat = driver
            worst_category = cat_data["label"]

    primary_goal = None
    if worst_feat and worst_sv < -0.005:
        primary_goal = {
            "feature": worst_feat["feature"],
            "label": worst_feat["label"],
            "category": worst_category,
            "tip": worst_feat["tip"],
        }

    return {"categories": categories, "primary_goal": primary_goal}


# ============================================================
# PREDICTION PIPELINE
# ============================================================
MODEL_MIN = 0.2332
MODEL_MAX = 0.6733

def rescale_score(raw_score: float) -> float:
    scaled = (raw_score - MODEL_MIN) / (MODEL_MAX - MODEL_MIN)
    return max(0.0, min(1.0, scaled))

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
    clipped_raw = float(np.clip(raw_score, 0.0, 1.0))
    confidence_score = rescale_score(clipped_raw)
    logger.info(f"Score rescaling: raw={clipped_raw:.4f} → rescaled={confidence_score:.4f}")

    # 5 — SHAP explanations (use raw score, not rescaled)
    shap_result = generate_shap_explanations(X, clipped_raw)

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
                # SHAP-driven category sentiments for frontend UI boxes
                "ui_sync": _build_category_sentiments(shap["all_shap_values"], prediction["wpm"]),
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
