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
    "loudnessPeaksPerSec":                        "Energy & Emphasis",
    "loudness_sma3_percentile20.0":               "Low-volume Clarity",
    "loudness_dynamics_power":                    "Dynamic Range",
    "vocal_projection":                           "Vocal Presence",
    "loudness_sma3_meanRisingSlope":              "Sentence Energy",
    "loudness_sma3_stddevNorm":                   "Volume Variety",
    "loudness_sma3_pctlrange0-2":                 "Whisper-to-Shout Range",
    "loudness_sma3_meanFallingSlope":             "Sentence Endings",
    "loudness_sma3_stddevRisingSlope":            "Attack Variation",
    # --- Fluency / Steadiness ---
    "voiced_flow":                                "Speaking Flow",
    "vocal_instability":                          "Voice Steadiness",
    "StddevUnvoicedSegmentLength":                "Pause Consistency",
    "shimmerLocaldB_sma3nz_stddevNorm":           "Volume Wobble",
    "MeanVoicedSegmentLengthSec":                 "Phrase Length",
    # --- Clarity / Crispness ---
    "HNRdBACF_sma3nz_amean":                      "Voice Clarity",
    "alphaRatioUV_sma3nz_amean":                  "Consonant Crispness",
    "slopeUV500-1500_sma3nz_amean":               "Voice Brightness",
    "F2bandwidth_sma3nz_stddevNorm":              "Mouth Movement Consistency",
    "F3bandwidth_sma3nz_stddevNorm":              "Pronunciation Consistency",
    # --- Pitch / Intonation ---
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":     "Pitch Variety",
    # --- Breath & Tension ---
    "logRelF0-H1-H2_sma3nz_stddevNorm":          "Breath Control Variation",
    # --- Expressiveness ---
    "spectralFluxV_sma3nz_stddevNorm":            "Expressive Variety",
    "hammarbergIndexUV_sma3nz_amean":             "Spectral Tilt",
    # --- Resonance ---
    "F3amplitudeLogRelF0_sma3nz_stddevNorm":      "Vocal Richness",
    "F3amplitudeLogRelF0_sma3nz_amean":           "Voice Richness",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm":      "Vocal Resonance",
    "F2amplitudeLogRelF0_sma3nz_amean":           "Vowel Power",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm":      "Resonance Variation",
    "F2frequency_sma3nz_stddevNorm":              "Vowel Clarity Range",
    "slopeV500-1500_sma3nz_amean":                "Voice Brightness",
    "F3frequency_sma3nz_stddevNorm":              "Tone Consistency",
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
}

# ============================================================
# COACHING TIPS — Problem → Why it matters → The Drill
# Every tip must describe a PHYSICAL ACTION the user can do.
# No acoustic jargon. If a user can't visualize the body part
# they need to move, the feedback is useless.
# ============================================================
FEATURE_TIPS = {
    "loudnessPeaksPerSec": {
        "positive": "Excellent! You naturally emphasized key words.",
        "negative": "Your volume was quite flat. Easy daily practice: Pick any 3 important words and say them noticeably louder than the rest. Do this 5 times a day.",
    },
    "vocal_projection": {
        "positive": "Your voice carried well.",
        "negative": "Your voice sounded a bit weak. Fix: Sit/stand tall, breathe from your belly, and speak as if the person is 3 meters away.",
    },
    "vocal_instability": {
        "positive": "Your voice was steady.",
        "negative": "There was slight shakiness. Quick fix: Before speaking, exhale slowly through pursed lips for 5 seconds. Repeat 3–4 times.",
    },
    "loudness_dynamics_power": {
        "positive": "Good variation in volume.",
        "negative": "Volume didn't change much. Practice: Say a sentence normally, then say the most important word much louder. Repeat 10 times.",
    },
    "loudness_sma3_meanRisingSlope": {
        "positive": "You started sentences with good energy.",
        "negative": "You tended to start sentences softly. Fix: Make the first 2–3 words of every sentence slightly stronger.",
    },
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": {
        "positive": "Natural pitch changes — sounded conversational.",
        "negative": "Pitch was fairly flat. Easy practice: Let your voice go slightly higher when introducing a new idea.",
    },
    "voiced_flow": {
        "positive": "Smooth speaking flow.",
        "negative": "Sentences felt choppy. Fix: Think 'First... Then... Finally' before speaking and say each part in one breath.",
    },
    "loudness_sma3_percentile20.0": {
        "positive": "Even your quiet moments were clear and audible.",
        "negative": "Your quietest moments were a bit too soft to hear. Fix: Imagine you are speaking to someone at the back of the room, even when lowering your volume for effect.",
    },
    "slopeUV500-1500_sma3nz_amean": {
        "positive": "Your voice sounds bright and energetic.",
        "negative": "Your tone sounded a bit dull. Fix: Try 'smiling with your voice'—literally smiling slightly while you speak can brighten your vocal tone immediately.",
    },
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Good vocal richness and depth.",
        "negative": "Your voice sounded a bit thin and flat. Simple fix: Slightly open the back of your throat (like the start of a yawn) while speaking to add natural warmth.",
    },
    "F2amplitudeLogRelF0_sma3nz_amean": {
        "positive": "Strong vowel projection.",
        "negative": "Your vowels sounded a bit weak. Practice: Over-pronounce vowels like 'I wOrkEd On A prOjEct' for 30 seconds, then speak normally.",
    },
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Your voice has great resonance and clarity.",
        "negative": "Your voice sounded a bit muffled. Practice speaking 'forward'—imagine your voice is hitting your front teeth rather than staying in your throat.",
    },
    "F3frequency_sma3nz_stddevNorm": {
        "positive": "You maintained a very stable and professional tone.",
        "negative": "Your tone shifted slightly during sentences. Focus on maintaining a steady breath until the end of your thought.",
    },
    "HNRdBACF_sma3nz_amean": {
        "positive": "Your voice sounded clean and smooth — easy and pleasant to listen to.",
        "negative": "Your voice sounded breathy or rough. Fix: Hum for 10 seconds to warm up your vocal cords, and support each sentence with a full breath.",
    },
    "StddevUnvoicedSegmentLength": {
        "positive": "Your pauses were consistent and purposeful — structured thinking, not stumbling.",
        "negative": "Your pauses were erratic — some too short, some awkwardly long. Fix: Use deliberate 1-second pauses between ideas.",
    },
    "shimmerLocaldB_sma3nz_stddevNorm": {
        "positive": "Steady volume control — composed and consistent.",
        "negative": "Your volume wobbled unpredictably between words. Fix: Practice sustaining 'ahhh' at one volume for 5 seconds.",
    },
    "loudness_sma3_pctlrange0-2": {
        "positive": "Good whisper-to-shout range — dynamic and engaging delivery.",
        "negative": "You spoke at one volume with almost no range. Fix: Practice reading a paragraph starting quiet, building to medium, finishing strong.",
    },
    "F2bandwidth_sma3nz_stddevNorm": {
        "positive": "Precise mouth movements — clear and well-formed sounds.",
        "negative": "Articulation was inconsistent — some words clear, others mumbled. Fix: Practice tongue twisters for 30 seconds before the interview.",
    },
    "alphaRatioUV_sma3nz_amean": {
        "positive": "Crisp consonants — sharp and clear, making every word distinct.",
        "negative": "Consonants were soft and mushy — words blurred together. Fix: Exaggerate your 'T' and 'K' sounds when practicing.",
    },
    "spectralFluxV_sma3nz_stddevNorm": {
        "positive": "Good expressive variety — your voice shaped each idea differently.",
        "negative": "Your voice stayed too uniform. Fix: Pick one word per sentence to emphasize differently — louder, slower, or higher pitch.",
    },
    "loudness_sma3_stddevNorm": {
        "positive": "Natural volume variation — organic and engaging.",
        "negative": "Volume pattern was flat or erratic. Fix: Record yourself telling a story and mirror that natural energy in interview answers.",
    },
}


def generate_final_summary(score: float, neg_features: list, pos_features: list) -> dict:
    """
    Generate a score-based coaching summary from the top negative/positive SHAP features.
    Returns a dict for inclusion in the result payload.
    """
    if score > 0.6:
        opening = "You have a very strong vocal presence! You sound ready for a high-level pitch."
    elif score > 0.45:
        opening = "Good solid performance. You sound professional and capable."
    else:
        opening = "A great start! With a few vocal tweaks, you can really boost your impact."

    focus_note = None
    best_trait = None

    if neg_features:
        main_fix_label = FEATURE_LABELS.get(neg_features[0]["feature"], "vocal habits")
        focus_note = (
            f"Don't worry too much about the {main_fix_label.lower()} right now. "
            "Everyone's voice has natural 'fingerprints' that show up under AI analysis."
        )
        if pos_features:
            best_trait_label = FEATURE_LABELS.get(pos_features[0]["feature"], "strengths")
            best_trait = f"Focus on leaning into your {best_trait_label.upper()}, which is already working well."

    return {
        "opening": opening,
        "focus_note": focus_note,
        "best_trait": best_trait,
        "reminder": (
            "This is a simulation. The best interviewers look for your passion and skills — "
            "this AI is just here to help you polish the delivery!"
        ),
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

        models['final_model'] = joblib.load("models/xgboost_balance_model.pkl")
        models['top_features'] = joblib.load("models/top_features_balance.pkl")
        print(f"   XGBoost model loaded — {len(models['top_features'])} features")

        models['smile'] = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("   openSMILE eGeMAPSv02 initialized")

        models['shap_explainer'] = shap.TreeExplainer(models['final_model'])
        print("   SHAP TreeExplainer ready")

        models['beta_params'] = joblib.load("models/beta_params_production.pkl")
        bp = models['beta_params']
        print(f"   Beta calibrator loaded — params c={bp[0]:.3f}, d={bp[1]:.3f}, e={bp[2]:.3f}")

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
            if abs(sv) < 0.008:
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
                "category": _FEATURE_TO_CATEGORY.get(feat_name, "Other"),
                "value": round(fval, 4),
                "shap_value": round(sv, 4),
                "direction": "increased" if sv > 0 else "decreased",
                "impact_magnitude": round(abs(sv), 4),
                "explanation": tip,
            })

        contributors.sort(key=lambda x: x["impact_magnitude"], reverse=True)

        # Only include improvements with meaningful negative impact (> 2% of score, aligned with category threshold)
        significant_neg = [c for c in contributors if c["direction"] == "decreased" and c["impact_magnitude"] > 0.02]
        # Cap at top 3 strengths — prevents dilution with low-R² model
        significant_pos = [c for c in contributors if c["direction"] == "increased"][:3]
        top_contributors = significant_neg + significant_pos

        return {
            "base_value": round(base_value, 4),
            "predicted_value": round(predicted_score, 4),
            "top_contributors": top_contributors,
            "top_improvements": significant_neg,
            "top_strengths": significant_pos,
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

# Reverse lookup: feature name → parent category label
_FEATURE_TO_CATEGORY: dict[str, str] = {}
for _cat_key, _cat_def in SHAP_CATEGORIES.items():
    for _feat in _cat_def["features"]:
        _FEATURE_TO_CATEGORY[_feat] = _cat_def["label"]


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

        if shap_sum > 0.02:
            status = "Helped Your Score"
        elif shap_sum < -0.02:
            status = "Held Back Your Score"
        else:
            status = "Minimal Impact"

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
            "impact_pct": round(abs(shap_sum) * 100, 1),
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
    if worst_feat and worst_sv < -0.02:
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
def beta_calibrate(raw_score: float) -> float:
    """
    Logistic-beta calibration fitted on 200 in-house rated clips.
    Mirrors beta_curve from the calibration training script — must
    not deviate, or scores will diverge from the rating team's scale.
    """
    c, d, e = models['beta_params']
    x = float(np.clip(raw_score, 1e-5, 1.0 - 1e-5))
    log_odds = c * np.log(x) + d * np.log(1.0 - x) + e
    return float(1.0 / (1.0 + np.exp(-log_odds)))

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
    confidence_score = beta_calibrate(clipped_raw)
    logger.info(f"Score calibration: raw={clipped_raw:.4f} → calibrated={confidence_score:.4f}")

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

            "modelVersion": "v2.1-egemaps-shap-betacal",

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
                # Coaching summary for report
                "finalSummary": generate_final_summary(
                    prediction["confidence_score"],
                    shap.get("top_improvements", []),
                    shap.get("top_strengths", []),
                ),
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