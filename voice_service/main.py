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

_HERE = Path(__file__).resolve().parent
CALIBRATION_PATH = str(_HERE / "models" / "audio_calibration_data_v1.json")


# ════════════════════════════════════════════════════════════════════════════════
# AUDIO FEEDBACK SYSTEM
# Mirrors the video service's group-based SHAP + golden zone feedback exactly.
# ════════════════════════════════════════════════════════════════════════════════

# ── Friendly names shown to the user ─────────────────────────────────────────
FRIENDLY_NAMES = {
    # Loudness & Energy
    "loudness_sma3_percentile20.0":      "Minimum Loudness",
    "loudness_sma3_percentile50.0":      "Median Loudness",
    "loudness_sma3_percentile80.0":      "Peak Loudness",
    "loudness_sma3_stddevNorm":          "Loudness Variation",
    "loudness_sma3_pctlrange0-2":        "Loudness Dynamic Range",
    "loudness_sma3_meanRisingSlope":     "Loudness Rising Intensity",
    "loudness_sma3_stddevRisingSlope":   "Loudness Rise Consistency",
    "loudness_sma3_meanFallingSlope":    "Loudness Falling Speed",
    "loudness_sma3_stddevFallingSlope":  "Loudness Fall Consistency",
    "loudnessPeaksPerSec":               "Loudness Peaks per Second",
    "loudness_dynamics_power":           "Loudness Dynamics Power",
    "vocal_projection":                  "Vocal Projection",
    # Pitch & Expressiveness
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "Pitch Variation",
    "spectralFlux_sma3_amean":           "Spectral Flux",
    "spectralFlux_sma3_stddevNorm":      "Spectral Flux Variation",
    "alphaRatioV_sma3nz_amean":          "High-Freq Energy Ratio",
    "alphaRatioV_sma3nz_stddevNorm":     "High-Freq Energy Variation",
    "hammarbergIndexV_sma3nz_stddevNorm":"Vocal Brightness Variation",
    "hammarbergIndexUV_sma3nz_amean":    "Unvoiced Brightness",
    "brightness_contrast":               "Spectral Brightness Contrast",
    "slopeV0-500_sma3nz_stddevNorm":     "Low-Freq Slope Variation",
    "slopeUV0-500_sma3nz_amean":         "Unvoiced Low-Freq Slope",
    "slopeUV500-1500_sma3nz_amean":      "Unvoiced Mid-Freq Slope",
    # Voice Quality
    "HNRdBACF_sma3nz_amean":            "Voice Clarity (HNR)",
    "HNRdBACF_sma3nz_stddevNorm":       "Voice Clarity Consistency",
    "shimmerLocaldB_sma3nz_amean":       "Voice Shimmer",
    "shimmerLocaldB_sma3nz_stddevNorm":  "Shimmer Consistency",
    "vocal_instability":                 "Vocal Instability",
    "F2amplitudeLogRelF0_sma3nz_amean":      "F2 Amplitude",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": "F2 Amplitude Variation",
    "F2bandwidth_sma3nz_stddevNorm":          "F2 Bandwidth Variation",
    "F2frequency_sma3nz_stddevNorm":          "F2 Frequency Variation",
    "F3amplitudeLogRelF0_sma3nz_amean":      "F3 Amplitude",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": "F3 Amplitude Variation",
    "F3bandwidth_sma3nz_stddevNorm":          "F3 Bandwidth Variation",
    "F3frequency_sma3nz_stddevNorm":          "F3 Frequency Variation",
    "logRelF0-H1-H2_sma3nz_stddevNorm":      "Harmonic Balance Variation",
    "mfcc1_sma3_stddevNorm":   "Timbre Variation (MFCC1)",
    "mfcc1V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC1)",
    "mfcc2V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC2)",
    "mfcc3_sma3_stddevNorm":   "Timbre Variation (MFCC3)",
    "mfcc4V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC4)",
    "mfcc4_sma3_stddevNorm":   "Timbre Variation (MFCC4)",
    # Fluency & Flow
    "voiced_flow":                       "Voiced Speech Flow",
    "MeanUnvoicedSegmentLength":         "Average Pause Length",
}

# ── Feedback groups — each maps to one UI card ───────────────────────────────
FEEDBACK_GROUPS = {
    "Loudness & Energy": {
        "members": [
            "loudness_sma3_percentile20.0", "loudness_sma3_percentile50.0",
            "loudness_sma3_percentile80.0", "loudness_sma3_stddevNorm",
            "loudness_sma3_pctlrange0-2",   "loudness_sma3_meanRisingSlope",
            "loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope",
            "loudness_sma3_stddevFallingSlope","loudnessPeaksPerSec",
            "loudness_dynamics_power",       "vocal_projection",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0800,
    },
    "Pitch & Expressiveness": {
        "members": [
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "spectralFlux_sma3_amean",       "spectralFlux_sma3_stddevNorm",
            "brightness_contrast",
            "alphaRatioV_sma3nz_amean",      "alphaRatioV_sma3nz_stddevNorm",
            "hammarbergIndexV_sma3nz_stddevNorm", "hammarbergIndexUV_sma3nz_amean",
            "slopeV0-500_sma3nz_stddevNorm",
            "slopeUV0-500_sma3nz_amean",     "slopeUV500-1500_sma3nz_amean",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0600,
    },
    "Voice Quality": {
        "members": [
            "HNRdBACF_sma3nz_amean",      "HNRdBACF_sma3nz_stddevNorm",
            "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm",
            "vocal_instability",
            "F2amplitudeLogRelF0_sma3nz_amean",      "F2amplitudeLogRelF0_sma3nz_stddevNorm",
            "F2bandwidth_sma3nz_stddevNorm",          "F2frequency_sma3nz_stddevNorm",
            "F3amplitudeLogRelF0_sma3nz_amean",      "F3amplitudeLogRelF0_sma3nz_stddevNorm",
            "F3bandwidth_sma3nz_stddevNorm",          "F3frequency_sma3nz_stddevNorm",
            "logRelF0-H1-H2_sma3nz_stddevNorm",
            "mfcc1_sma3_stddevNorm",     "mfcc1V_sma3nz_stddevNorm",
            "mfcc2V_sma3nz_stddevNorm",  "mfcc3_sma3_stddevNorm",
            "mfcc4V_sma3nz_stddevNorm",  "mfcc4_sma3_stddevNorm",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0700,
    },
    "Fluency & Flow": {
        "members": [
            "voiced_flow",
            "MeanUnvoicedSegmentLength",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0400,
    },
}

# Features too technical/noisy to surface as user-facing coaching tips
SKIP_COACHING = {
    "F2frequency_sma3nz_stddevNorm",
    "F3frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "mfcc4_sma3_stddevNorm",
    "mfcc4V_sma3nz_stddevNorm",
}


# ── Coaching tip generator ────────────────────────────────────────────────────
def _get_coaching_tip(feat: str, val: float, zone: dict, direction: str) -> tuple[str, str]:
    """
    Returns (tip_text, status_colour).
    direction: "positive" = SHAP helped score, "negative" = SHAP hurt score.
    status_colour: "green" | "yellow" | "red"
    """
    friendly = FRIENDLY_NAMES.get(feat, feat)
    zone_min = zone.get("min", -99)
    zone_max = zone.get("max",  99)
    feat_dir = zone.get("direction", "INCREASING")
    in_zone  = zone_min <= val <= zone_max

    # ── Positive SHAP ────────────────────────────────────────────────────────
    if direction == "positive":
        if in_zone or val >= zone_min:
            return f"Your {friendly} is a key strength — it's well-calibrated.", "green"
        return f"Your {friendly} is on the right track. Keep improving it.", "yellow"

    # ── Negative SHAP ────────────────────────────────────────────────────────

    # Loudness / Energy
    if "loudness" in feat.lower() or feat == "vocal_projection":
        if "stddev" in feat or "Slope" in feat:
            if val < zone_min:
                return f"Your {friendly} is very flat — vary your volume more to sound dynamic.", "red"
            if val > zone_max:
                return f"Your {friendly} is too erratic — work on smoother volume control.", "red"
        else:
            if feat_dir == "INCREASING" and val < zone_min:
                return f"Your {friendly} is too low — speak louder and project more.", "red"
            if feat_dir == "DECREASING" and val > zone_max:
                return f"Your {friendly} is too intense — dial back the volume slightly.", "red"
        return f"Your {friendly} could be better balanced for a confident delivery.", "yellow"

    if feat == "loudness_dynamics_power":
        if val < zone_min:
            return "Your loudness dynamics are weak — add more peaks and energy variation.", "red"
        return "Your loudness dynamics could be more expressive.", "yellow"

    # Pitch / Expressiveness
    if "F0semitone" in feat:
        if val < zone_min:
            return (
                "Your pitch variation is very low — you sound monotone. "
                "Try raising and lowering your voice to emphasise key points.", "red"
            )
        if val > zone_max:
            return "Your pitch varies too wildly — aim for controlled, purposeful intonation.", "yellow"
        return "Your pitch expressiveness could be slightly improved.", "yellow"

    if "spectralFlux" in feat:
        if val < zone_min:
            return "Your speech lacks tonal variety — it sounds static. Vary your tone and articulation.", "red"
        return "Your tonal variety could be more expressive.", "yellow"

    if "brightness_contrast" in feat:
        if val < zone_min:
            return (
                "Your voice lacks high-frequency brightness — "
                "work on clearer articulation and more forward placement.", "yellow"
            )
        if val > zone_max:
            return "Your voice is overly bright/harsh. A slightly warmer tone will sound more natural.", "yellow"
        return "Your spectral brightness balance could be adjusted.", "yellow"

    if "alphaRatio" in feat:
        if val < zone_min:
            return (
                "Your voice has low high-frequency energy — "
                "it may sound muffled. Work on clear, forward articulation.", "yellow"
            )
        return f"Your {friendly} could be adjusted for a cleaner sound.", "yellow"

    if "hammarberg" in feat.lower() or "slope" in feat.lower():
        return f"Your {friendly} could be better calibrated for confident delivery.", "yellow"

    # Voice Quality
    if "HNR" in feat:
        if val < zone_min:
            return (
                "Your voice clarity (HNR) is low — it sounds breathy or rough. "
                "Support your breath and engage your core when speaking.", "red"
            )
        return "Your voice clarity could be slightly improved.", "yellow"

    if "shimmer" in feat.lower():
        if "stddev" in feat:
            if val > zone_max:
                return "Your shimmer is erratic — focus on steady, supported phonation.", "red"
            if val < zone_min:
                return "Your shimmer consistency is too low — your voice amplitude is unpredictably variable.", "yellow"
        else:
            if val > zone_max:
                return "Your voice shimmer is high — it sounds unsteady. Work on breath support and vocal stability.", "red"
            if val < zone_min:
                return "Your shimmer is unusually low — your voice may sound overly rigid.", "yellow"
        return "Your voice steadiness could be improved.", "yellow"

    if "vocal_instability" in feat:
        if val > zone_max:
            return (
                "Your voice is unstable (high jitter × shimmer). "
                "Focus on steady breath support and relaxed phonation.", "red"
            )
        if val < zone_min:
            return (
                "Your vocal instability is unusually low — "
                "ensure it reflects natural speech rather than suppressed movement.", "yellow"
            )
        return "Your vocal stability could be slightly improved.", "yellow"

    if "mfcc" in feat.lower():
        if val < zone_min:
            return (
                f"Your {friendly} is too uniform — "
                "vary your articulation and resonance for a richer vocal quality.", "yellow"
            )
        if val > zone_max:
            return f"Your {friendly} is too erratic — aim for consistent, controlled articulation.", "red"
        return f"Your {friendly} could be refined for better vocal quality.", "yellow"

    if "F2amplitude" in feat or "F3amplitude" in feat:
        if val < zone_min:
            return f"Your {friendly} is weak — work on clearer vowel resonance.", "yellow"
        if val > zone_max:
            return f"Your {friendly} is too prominent — balance your resonance.", "yellow"
        return f"Your {friendly} could be better balanced.", "yellow"

    # Fluency & Flow
    if "voiced_flow" in feat:
        if val < zone_min:
            return (
                "Your voiced speech flow is low — you have too many pauses or short segments. "
                "Work on longer, connected phrases.", "red"
            )
        if val > zone_max:
            return (
                "Your voiced flow is very high — you may be speaking without enough breathing space. "
                "Allow natural pauses for impact.", "yellow"
            )
        return "Your speech flow could be improved for better rhythm.", "yellow"

    if "MeanUnvoicedSegmentLength" in feat:
        if val > zone_max:
            return (
                "Your pauses are too long — they break your flow and lose listener attention. "
                "Aim for shorter, purposeful pauses.", "red"
            )
        if val < zone_min:
            return "Your pauses are very short — allow brief moments of silence to let key points land.", "yellow"
        return "Your pause length could be better calibrated.", "yellow"

    # Generic fallback
    if val < zone_min:
        if feat_dir == "INCREASING":
            return f"Your {friendly} is below the ideal range — try to increase it.", "red"
        elif feat_dir == "DECREASING":
            return f"Your {friendly} is above the ideal — dial it back slightly.", "red"
        return f"Your {friendly} is outside the ideal range — try to find the sweet spot.", "yellow"
    if val > zone_max:
        if feat_dir == "DECREASING":
            return f"Your {friendly} is above the ideal — dial it back slightly.", "red"
        return f"Your {friendly} is a bit high — a more moderate level works best.", "yellow"

    return f"Your {friendly} could be slightly adjusted.", "yellow"


# ── Eligible feature selector ─────────────────────────────────────────────────
def _get_eligible_features(
    shap_dict: dict,
    group_members: list,
    feature_values: dict,
    direction: str,
    overall_score: float,
    golden_zones: dict,
) -> list:
    """Filters and ranks features for tip generation. Mirrors video service exactly."""
    if direction == "negative":
        if overall_score < 0.65:
            sensitivity = 0.001
        elif overall_score > 0.80:
            sensitivity = 0.015
        else:
            sensitivity = 0.005
    else:
        sensitivity = 0.003

    eligible = []
    for feat, shap_val in shap_dict.items():
        if feat not in group_members:
            continue
        if feat in SKIP_COACHING:
            continue

        val    = feature_values.get(feat, 0)
        zone   = golden_zones.get(feat, {"min": -99, "max": 99})
        in_zone = zone.get("min", -99) <= val <= zone.get("max", 99)

        if direction == "positive":
            if shap_val <= sensitivity:
                continue
        else:
            if shap_val >= -sensitivity:
                continue
            if overall_score > 0.80 and in_zone:
                continue

        eligible.append((feat, shap_val))

    eligible.sort(key=lambda x: abs(x[1]), reverse=True)
    return eligible


# ── Group results builder ─────────────────────────────────────────────────────
def build_audio_group_results(
    shap_dict: dict,
    feature_values: dict,
    overall_score: float,
    base_value: float,
    golden_zones: dict,
) -> dict:
    """
    Mirrors video service's build_group_results() exactly.
    Returns a dict keyed by group name, each with impact_points, status, tips.
    Also returns a _metadata key with baseline and synergy_gap.
    """
    bv = float(base_value[0]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)
    baseline_pct         = bv * 100
    explainer_prediction = baseline_pct
    group_results        = {}

    for group_name, group_info in FEEDBACK_GROUPS.items():
        members = group_info["members"]

        group_shap_sum = sum(v for f, v in shap_dict.items() if f in members)
        impact_points  = group_shap_sum * 100
        explainer_prediction += impact_points

        if   group_shap_sum >= group_info["yellow_threshold"]: status = "green"
        elif group_shap_sum >= group_info["red_threshold"]:    status = "yellow"
        else:                                                  status = "red"

        neg_features = _get_eligible_features(
            shap_dict, members, feature_values, "negative", overall_score, golden_zones
        )
        pos_features = _get_eligible_features(
            shap_dict, members, feature_values, "positive", overall_score, golden_zones
        )

        if   status == "green":  p_limit, n_limit = 3, 1
        elif status == "yellow": p_limit, n_limit = 2, 2
        else:                    p_limit, n_limit = 1, 3

        tips: list = []
        seen: set  = set()

        def process_tips(features, dir_type, limit):
            count = 0
            for feat, shap_val in features:
                if count >= limit or feat in seen:
                    continue
                val  = feature_values.get(feat, 0)
                zone = golden_zones.get(feat, {})
                tip_text, tip_status = _get_coaching_tip(feat, val, zone, dir_type)
                tips.append({
                    "feature":        feat,
                    "friendly":       FRIENDLY_NAMES.get(feat, feat),
                    "shap":           round(float(shap_val), 4),
                    "value":          round(float(val), 4),
                    "direction":      dir_type,
                    "tip":            tip_text,
                    "status":         tip_status,
                    "zone_min":       round(float(zone.get("min", -99)), 4),
                    "zone_max":       round(float(zone.get("max",  99)), 4),
                    "zone_direction": zone.get("direction", "INCREASING"),
                })
                seen.add(feat)
                count += 1

        if status == "red":
            process_tips(neg_features, "negative", n_limit)
            process_tips(pos_features, "positive", p_limit)
        else:
            process_tips(pos_features, "positive", p_limit)
            process_tips(neg_features, "negative", n_limit)

        group_results[group_name] = {
            "impact_points": round(float(impact_points), 1),
            "status":        status,
            "tips":          tips,
        }

    group_results["_metadata"] = {
        "baseline":    round(baseline_pct, 1),
        "synergy_gap": round((overall_score * 100) - explainer_prediction, 1),
    }
    return group_results


# ============================================================
# LEGACY COACHING KNOWLEDGE BASE
# (kept for backward compat — featureExplanations field)
# ============================================================
FEATURE_LABELS = {
    "loudnessPeaksPerSec":                        "Energy & Emphasis",
    "loudness_sma3_percentile20.0":               "Low-volume Clarity",
    "loudness_dynamics_power":                    "Dynamic Range",
    "vocal_projection":                           "Vocal Presence",
    "loudness_sma3_meanRisingSlope":              "Sentence Energy",
    "loudness_sma3_stddevNorm":                   "Volume Variety",
    "loudness_sma3_pctlrange0-2":                 "Whisper-to-Shout Range",
    "loudness_sma3_meanFallingSlope":             "Sentence Endings",
    "loudness_sma3_stddevRisingSlope":            "Attack Variation",
    "voiced_flow":                                "Speaking Flow",
    "vocal_instability":                          "Voice Steadiness",
    "shimmerLocaldB_sma3nz_stddevNorm":           "Volume Wobble",
    "HNRdBACF_sma3nz_amean":                      "Voice Clarity",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":     "Pitch Variety",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm":      "Vocal Richness",
    "F3amplitudeLogRelF0_sma3nz_amean":           "Voice Richness",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm":      "Vocal Resonance",
    "F2amplitudeLogRelF0_sma3nz_amean":           "Vowel Power",
    "mfcc1_sma3_stddevNorm":                      "Voice Texture Variety",
    "mfcc2V_sma3nz_stddevNorm":                   "Voice Warmth Variety",
    "mfcc3_sma3_stddevNorm":                      "Tonal Variety",
    "hammarbergIndexV_sma3nz_stddevNorm":         "Voice Quality Steadiness",
    "alphaRatioV_sma3nz_amean":                   "Spectral Balance",
    "alphaRatioV_sma3nz_stddevNorm":              "Spectral Balance Variation",
    "slopeV0-500_sma3nz_stddevNorm":              "Low-Frequency Variation",
    "spectralFlux_sma3_amean":                    "Expressive Variety",
    "MeanUnvoicedSegmentLength":                  "Pause Length",
    "brightness_contrast":                        "Spectral Brightness",
}

FEATURE_TIPS = {
    "loudnessPeaksPerSec": {
        "positive": "Excellent! You naturally emphasized key words.",
        "negative": "Your volume was quite flat. Practice: Pick 3 important words and say them noticeably louder. Do this 5 times a day.",
    },
    "vocal_projection": {
        "positive": "Your voice carried well.",
        "negative": "Your voice sounded a bit weak. Fix: Sit tall, breathe from your belly, and speak as if the person is 3 meters away.",
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
        "negative": "Your quietest moments were a bit too soft. Fix: Imagine speaking to someone at the back of the room, even when quiet.",
    },
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": {
        "positive": "Good vocal richness and depth.",
        "negative": "Your voice sounded a bit thin. Fix: Slightly open the back of your throat (like the start of a yawn) while speaking.",
    },
    "F2amplitudeLogRelF0_sma3nz_amean": {
        "positive": "Strong vowel projection.",
        "negative": "Your vowels sounded a bit weak. Practice: Over-pronounce vowels for 30 seconds, then speak normally.",
    },
    "HNRdBACF_sma3nz_amean": {
        "positive": "Your voice sounded clean and smooth — easy and pleasant to listen to.",
        "negative": "Your voice sounded breathy or rough. Fix: Hum for 10 seconds to warm up your vocal cords.",
    },
    "shimmerLocaldB_sma3nz_stddevNorm": {
        "positive": "Steady volume control — composed and consistent.",
        "negative": "Your volume wobbled unpredictably between words. Fix: Practice sustaining 'ahhh' at one volume for 5 seconds.",
    },
    "MeanUnvoicedSegmentLength": {
        "positive": "Your pauses were consistent and purposeful.",
        "negative": "Your pauses were too long. Fix: Use deliberate 1-second pauses between ideas only.",
    },
    "mfcc2V_sma3nz_stddevNorm": {
        "positive": "Natural vocal warmth variation — engaging and human.",
        "negative": "Your vocal warmth was too erratic. Focus on consistent breath support throughout each phrase.",
    },
    "alphaRatioV_sma3nz_amean": {
        "positive": "Good spectral balance — your voice sounds clear and forward.",
        "negative": "Your voice lacked high-frequency energy — may sound muffled. Fix: Speak slightly more forward, lips apart.",
    },
}


def _get_generic_tip(label: str, is_positive: bool) -> str:
    if is_positive:
        return f"Your {label.lower()} was a strength — keep doing what you're doing."
    return f"Your {label.lower()} needs work. Record yourself practicing and listen back."


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

        models['final_model']  = joblib.load("models/xgboost_balance_model.pkl")
        models['top_features'] = joblib.load("models/top_features_balance.pkl")
        print(f"   XGBoost model loaded — {len(models['top_features'])} features")

        models['smile'] = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print("   openSMILE eGeMAPSv02 initialized")

        models['shap_proxy']    = joblib.load("models/shap_proxy_calibrated.pkl")
        models['shap_explainer'] = shap.TreeExplainer(models['shap_proxy'])
        print("   SHAP TreeExplainer ready (calibration-aware proxy)")

        models['beta_params'] = joblib.load("models/beta_params_production.pkl")
        bp = models['beta_params']
        print(f"   Beta calibrator loaded — params {bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}")

        # Load golden zones from calibration JSON
        if not Path(CALIBRATION_PATH).exists():
            raise FileNotFoundError(f"Missing calibration file: {CALIBRATION_PATH}. Run calibrate_audio_golden_zones.py first.")
        with open(CALIBRATION_PATH) as f:
            calibration = json.load(f)
        models['golden_zones'] = calibration["golden_zones"]
        models['global_stats'] = calibration.get("global_stats", {})
        print(f"   Audio golden zones loaded: {len(models['golden_zones'])} features")

        # SHAP proxy feature list — always use these for SHAP (45 features)
        models['shap_features'] = list(models['shap_proxy'].feature_names_in_)

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

    if all(c in cols for c in ['slopeV500-1500_sma3nz_amean', 'slopeV0-500_sma3nz_amean']):
        df['brightness_contrast'] = (
            df['slopeV500-1500_sma3nz_amean'] - df['slopeV0-500_sma3nz_amean']
        )

    return df


def compute_wpm(audio_path: str, transcript_text: str | None) -> float:
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
# SCORING
# ============================================================
def beta_calibrate(raw_score: float) -> float:
    """
    Beta CDF calibration: maps raw XGBoost [0,1] → calibrated [0,1].
    Uses scipy-compatible CDF, matching calibrate_audio_golden_zones.py.
    """
    from scipy.stats import beta as beta_dist
    a, b, _ = models['beta_params']
    return float(np.clip(beta_dist.cdf(float(raw_score), float(a), float(b)), 0.0, 1.0))


def score_to_label(score: float) -> str:
    if score >= 0.75:  return "Highly Confident"
    if score >= 0.50:  return "Confident"
    if score >= 0.30:  return "Moderately Confident"
    return "Needs Improvement"


# ============================================================
# SHAP ENGINE  (group-based, mirrors video service)
# ============================================================
def generate_shap_and_groups(
    X: pd.DataFrame,
    calibrated_score: float,
    feature_values_dict: dict,
) -> dict:
    """
    Computes SHAP values and builds group results exactly like the video service.
    Returns both group_results (the structured UI data) and legacy fields.
    """
    try:
        explainer    = models['shap_explainer']
        shap_features = models['shap_features']
        golden_zones  = models['golden_zones']

        # Align X to shap_features order, fill missing with 0
        X_shap = pd.DataFrame(index=[0], columns=shap_features, dtype=float)
        for col in shap_features:
            X_shap[col] = feature_values_dict.get(col, 0.0)

        X_np      = X_shap.to_numpy(dtype=np.float64)
        shap_vals = explainer.shap_values(X_np)[0]
        base_value = float(explainer.expected_value)

        shap_dict = {feat: float(sv) for feat, sv in zip(shap_features, shap_vals)}

        # ── Group results (the main structured output) ───────────────────────
        group_results = build_audio_group_results(
            shap_dict      = shap_dict,
            feature_values = feature_values_dict,
            overall_score  = calibrated_score,
            base_value     = base_value,
            golden_zones   = golden_zones,
        )

        # ── Legacy top_contributors list (for featureExplanations) ───────────
        top_contributors = []
        for feat, sv in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(sv) < 0.008:
                continue
            fval      = feature_values_dict.get(feat, 0.0)
            label     = FEATURE_LABELS.get(feat, FRIENDLY_NAMES.get(feat, feat))
            direction = "positive" if sv > 0 else "negative"
            tip       = FEATURE_TIPS.get(feat, {}).get(direction, _get_generic_tip(label, sv > 0))
            top_contributors.append({
                "feature":          feat,
                "label":            label,
                "value":            round(float(fval), 4),
                "shap_value":       round(sv, 4),
                "direction":        "increased" if sv > 0 else "decreased",
                "impact_magnitude": round(abs(sv), 4),
                "explanation":      tip,
            })

        neg_contributors = [c for c in top_contributors if c["direction"] == "decreased"]
        pos_contributors = [c for c in top_contributors if c["direction"] == "increased"]

        _max_improvements = (
            1 if calibrated_score >= 0.70 else
            2 if calibrated_score >= 0.50 else
            len(neg_contributors)
        )

        return {
            "group_results":     group_results,
            "shap_dict":         shap_dict,
            "base_value":        round(base_value, 4),
            "top_contributors":  top_contributors,
            "top_improvements":  neg_contributors[:_max_improvements],
            "top_strengths":     pos_contributors[:3],
            "all_shap_values":   {f: round(v, 4) for f, v in shap_dict.items()},
        }

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return {
            "group_results":    {},
            "shap_dict":        {},
            "base_value":       0.0,
            "top_contributors": [],
            "top_improvements": [],
            "top_strengths":    [],
            "all_shap_values":  {},
        }


# ============================================================
# PREDICTION PIPELINE
# ============================================================
def predict_confidence(audio_path: str, transcript_text: str | None = None) -> dict:
    logger.info(f"Starting voice analysis: {audio_path}")

    # 1 & 2 — extract + engineer features
    features_df = extract_opensmile_features(audio_path)
    features_df = add_engineered_features(features_df)

    # 3 — select top 45 features for the main model
    top_features = models['top_features']
    X = pd.DataFrame(index=[0], columns=top_features, dtype=float)
    for col in top_features:
        X[col] = features_df[col].iloc[0] if col in features_df.columns else 0.0

    # 4 — predict + calibrate
    raw_score       = float(np.clip(models['final_model'].predict(X)[0], 0.0, 1.0))
    calibrated_score = beta_calibrate(raw_score)
    logger.info(f"Score: raw={raw_score:.4f} → calibrated={calibrated_score:.4f}")

    # Build a flat dict of all available feature values (for group results lookup)
    feature_values_dict = features_df.iloc[0].to_dict()
    # Add engineered features that were computed
    for col in top_features:
        if col not in feature_values_dict:
            feature_values_dict[col] = float(X[col].iloc[0])

    # 5 — SHAP + group results
    shap_result = generate_shap_and_groups(X, calibrated_score, feature_values_dict)

    # 6 — WPM
    wpm = compute_wpm(audio_path, transcript_text)

    return {
        "confidence_score": calibrated_score,
        "raw_score":        raw_score,
        "confidence_label": score_to_label(calibrated_score),
        "wpm":              wpm,
        "shap":             shap_result,
        "raw_features":     features_df.iloc[0].to_dict(),
    }


def generate_final_summary(score: float, neg_features: list, pos_features: list) -> dict:
    if score > 0.6:
        opening = "You have a very strong vocal presence! You sound ready for a high-level pitch."
    elif score > 0.45:
        opening = "Good solid performance. You sound professional and capable."
    else:
        opening = "A great start! With a few vocal tweaks, you can really boost your impact."

    focus_note = best_trait = None
    if neg_features:
        main_fix_label = FEATURE_LABELS.get(neg_features[0].get("feature",""), "vocal habits")
        focus_note = (
            f"Don't worry too much about the {main_fix_label.lower()} right now. "
            "Everyone's voice has natural 'fingerprints' that show up under AI analysis."
        )
        if pos_features:
            best_trait_label = FEATURE_LABELS.get(pos_features[0].get("feature",""), "strengths")
            best_trait = f"Focus on leaning into your {best_trait_label.upper()}, which is already working well."

    return {
        "opening":    opening,
        "focus_note": focus_note,
        "best_trait": best_trait,
        "reminder": (
            "This is a simulation. The best interviewers look for your passion and skills — "
            "this AI is just here to help you polish the delivery!"
        ),
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
    try:
        print(f"\n{'='*60}")
        print(f"VOICE ANALYSIS — Turn {turn_id} | Session {interview_id}")
        print(f"{'='*60}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prediction = predict_confidence(audio_path, transcript_text)
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        raw         = prediction["raw_features"]
        shap        = prediction["shap"]
        group_results = shap["group_results"]

        # Derived backward-compat fields
        vsps            = raw.get("VoicedSegmentsPerSec", 1.0) or 1.0
        mvsl            = raw.get("MeanVoicedSegmentLengthSec", 0.3) or 0.3
        voiced_fraction = min(1.0, float(vsps) * float(mvsl))
        pause_ratio_approx = max(0.0, round(1.0 - voiced_fraction, 4))

        result = {
            "interviewTurnId": turn_id,

            # ── Core scores ──────────────────────────────────────────────────
            "confidenceLevel":     round(prediction["confidence_score"], 4),
            "confidenceLabelText": prediction["confidence_label"],

            # ── Backward-compat voice metrics (Prisma VoiceAnalysis schema) ─
            "speakingQuality":  round(prediction["confidence_score"], 4),
            "vocalStability":   round(max(0.0, 1.0 - float(raw.get("jitterLocal_sma3nz_amean", 0) or 0)), 4),
            "speakingFluency":  round(voiced_fraction, 4),
            "pitchMean":        round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0) or 0), 4),
            "pitchStd":         round(float(raw.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0) or 0), 4),
            "energyLevel":      round(float(raw.get("equivalentSoundLevel_dBp", 0) or 0), 4),
            "wordsPerMinute":   prediction["wpm"],
            "pauseRatio":       pause_ratio_approx,
            "jitter":           round(float(raw.get("jitterLocal_sma3nz_amean", 0) or 0), 6),
            "shimmer":          round(float(raw.get("shimmerLocaldB_sma3nz_amean", 0) or 0), 6),

            "modelVersion": "v2.2-egemaps-shap-betacal-zones",

            # ── SHAP values (stored in allProbabilities for DB) ──────────────
            "allProbabilities": shap["all_shap_values"],

            # ── Group results (top-level — mirrors video service exactly) ────
            # The frontend reads this the same way it reads videoAnalysis.groupResults
            "groupResults": group_results,

            # ── rawFeatures: full eGeMAPS + explainability data ──────────────
            "rawFeatures": {
                **{
                    k: (round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in raw.items()
                },
                # Structured group feedback (same as groupResults above, also embedded here)
                "groupResults": group_results,
                # Legacy fields kept for any existing frontend consumers
                "shapExplanations": shap["top_contributors"],
                "shapBaseValue":    shap["base_value"],
                "featureExplanations": [
                    c["explanation"]
                    for c in shap["top_improvements"]
                    if c.get("explanation")
                ],
                "finalSummary": generate_final_summary(
                    prediction["confidence_score"],
                    shap["top_improvements"],
                    shap["top_strengths"],
                ),
            },

            "status":           "completed",
            "processingTimeMs": elapsed_ms,
            "processedAt":      datetime.now().isoformat(),
        }

        # Console summary
        print(f"Confidence  : {result['confidenceLevel']*100:.1f}% ({result['confidenceLabelText']})")
        print(f"WPM         : {result['wordsPerMinute']}")
        print(f"Group statuses:")
        for gname, gdata in group_results.items():
            if gname == "_metadata":
                continue
            icon = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(gdata["status"],"?")
            print(f"  {icon} {gname}: {gdata['impact_points']:+.1f}%  ({len(gdata['tips'])} tips)")
        print(f"Processing  : {elapsed_ms}ms")
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
            "status":          "failed",
            "errorMessage":    str(e),
            "processedAt":     datetime.now().isoformat(),
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
        "status":       "ok",
        "service":      "Hirely Voice Analysis v2.2",
        "models_loaded": bool(models.get("final_model")),
        "shap_ready":   bool(models.get("shap_explainer")),
        "zones_loaded": bool(models.get("golden_zones")),
        "timestamp":    datetime.now().isoformat(),
    }


class AnalysisRequest(BaseModel):
    turn_id:      int
    interview_id: str
    audio_path:   str
    transcript:   str | None = None


@app.post("/analyze-voice")
async def analyze_voice(request_data: AnalysisRequest, background_tasks: BackgroundTasks):
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
            "status":    "queued",
            "turn_id":   request_data.turn_id,
            "message":   "Voice analysis queued.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return {"status": "error", "error": str(e)}, 500


@app.get("/result/{interview_id}/{turn_id}")
def get_result(interview_id: str, turn_id: int):
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
    print("Starting Hirely Voice Analysis Service v2.2")
    print("Docs: http://localhost:8001/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)