"""
audio_feedback_pipeline.py
────────────────────────────────────────────────────────────────────────────
Audio confidence feedback pipeline.

Model stack:
  1. XGBoost regressor  (shap_proxy_calibrated.pkl)  → raw score [0, 1]
  2. Beta CDF calibration (beta_params_production.pkl) → calibrated score [0, 1]
  3. SHAP TreeExplainer → per-feature attribution
  4. Golden zones (audio_calibration_data_v1.json) → coaching tips

USAGE:
    python audio_feedback_pipeline.py \
        --features path/to/audio_features.csv \
        --calibration audio_calibration_data_v1.json

The input CSV must contain the same 45 columns the model was trained on.
One row = one audio sample.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import beta as beta_dist

# ── Paths (edit to match your layout) ────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

MODEL_PATH       = str(_HERE / "shap_proxy_calibrated.pkl")
BETA_PARAMS_PATH = str(_HERE / "beta_params_production.pkl")
CALIBRATION_PATH = str(_HERE / "audio_calibration_data_v1.json")

# ── Load artefacts ────────────────────────────────────────────────────────────
print("[INIT] Loading model artefacts...")

model       = joblib.load(MODEL_PATH)
beta_params = joblib.load(BETA_PARAMS_PATH)   # [alpha, beta_b, scale]
ALPHA_B, BETA_B, SCALE = float(beta_params[0]), float(beta_params[1]), float(beta_params[2])

if not Path(CALIBRATION_PATH).exists():
    print(f"[ERROR] Missing calibration file: {CALIBRATION_PATH}")
    print("        Run calibrate_audio_golden_zones.py first.")
    sys.exit(1)

with open(CALIBRATION_PATH) as f:
    CALIBRATION = json.load(f)

GOLDEN_ZONES = CALIBRATION["golden_zones"]
GLOBAL_STATS = CALIBRATION["global_stats"]

MODEL_FEATURES = list(model.feature_names_in_)

# ── SHAP explainer ────────────────────────────────────────────────────────────
shap_explainer = shap.TreeExplainer(model)

print(f"[INIT] Model features  : {len(MODEL_FEATURES)}")
print(f"[INIT] Golden zones    : {len(GOLDEN_ZONES)}")
print(f"[INIT] Beta params     : alpha={ALPHA_B:.4f}  beta={BETA_B:.4f}  scale={SCALE:.4f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_score(raw_score: float) -> float:
    """
    Maps XGBoost raw prediction [0, 1] → calibrated score [0, 1]
    using the fitted Beta CDF.

    Beta(alpha=6.33, beta=2.29) peaks near 0.75, meaning the calibration
    is intentionally strict: only truly strong raw scores translate to high
    calibrated scores.  This prevents score inflation.
    """
    return float(np.clip(beta_dist.cdf(raw_score, ALPHA_B, BETA_B), 0.0, 1.0))


def score_sample(feature_dict: dict) -> dict:
    """
    Score one audio sample from a flat feature dict.
    Returns raw score, calibrated score, and SHAP dict.
    """
    X = pd.DataFrame([feature_dict])[MODEL_FEATURES]
    raw_score     = float(np.clip(model.predict(X)[0], 0.0, 1.0))
    cal_score     = calibrate_score(raw_score)
    shap_vals     = shap_explainer.shap_values(X)[0]
    shap_dict     = {feat: float(v) for feat, v in zip(MODEL_FEATURES, shap_vals)}
    base_value    = float(shap_explainer.expected_value)

    return {
        "raw_score":        raw_score,
        "calibrated_score": cal_score,
        "shap_dict":        shap_dict,
        "base_value":       base_value,
        "feature_values":   dict(zip(MODEL_FEATURES, X.iloc[0].values)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FRIENDLY NAMES  (what users see instead of raw feature names)
# ══════════════════════════════════════════════════════════════════════════════

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

    # Voice Quality
    "HNRdBACF_sma3nz_amean":            "Voice Clarity (HNR)",
    "HNRdBACF_sma3nz_stddevNorm":       "Voice Clarity Consistency",
    "shimmerLocaldB_sma3nz_amean":       "Voice Shimmer",
    "shimmerLocaldB_sma3nz_stddevNorm":  "Shimmer Consistency",
    "vocal_instability":                 "Vocal Instability",

    # Fluency & Flow
    "voiced_flow":                       "Voiced Speech Flow",
    "MeanUnvoicedSegmentLength":         "Average Pause Length",

    # Spectral / Timbre
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

    # Formants
    "F2amplitudeLogRelF0_sma3nz_amean":      "F2 Amplitude",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm": "F2 Amplitude Variation",
    "F2bandwidth_sma3nz_stddevNorm":          "F2 Bandwidth Variation",
    "F2frequency_sma3nz_stddevNorm":          "F2 Frequency Variation",
    "F3amplitudeLogRelF0_sma3nz_amean":      "F3 Amplitude",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm": "F3 Amplitude Variation",
    "F3bandwidth_sma3nz_stddevNorm":          "F3 Bandwidth Variation",
    "F3frequency_sma3nz_stddevNorm":          "F3 Frequency Variation",
    "logRelF0-H1-H2_sma3nz_stddevNorm":      "Harmonic Balance Variation",

    # MFCCs
    "mfcc1_sma3_stddevNorm":   "Timbre Variation (MFCC1)",
    "mfcc1V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC1)",
    "mfcc2V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC2)",
    "mfcc3_sma3_stddevNorm":   "Timbre Variation (MFCC3)",
    "mfcc4V_sma3nz_stddevNorm":"Voiced Timbre Variation (MFCC4)",
    "mfcc4_sma3_stddevNorm":   "Timbre Variation (MFCC4)",
}


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK GROUPS
# ══════════════════════════════════════════════════════════════════════════════
# Each group has member feature prefixes, and thresholds on the group-level
# SHAP sum to determine green/yellow/red status.

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
            "spectralFlux_sma3_amean",  "spectralFlux_sma3_stddevNorm",
            "brightness_contrast",
            "alphaRatioV_sma3nz_amean", "alphaRatioV_sma3nz_stddevNorm",
            "hammarbergIndexV_sma3nz_stddevNorm", "hammarbergIndexUV_sma3nz_amean",
            "slopeV0-500_sma3nz_stddevNorm",
            "slopeUV0-500_sma3nz_amean", "slopeUV500-1500_sma3nz_amean",
        ],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.0600,
    },
    "Voice Quality": {
        "members": [
            "HNRdBACF_sma3nz_amean",     "HNRdBACF_sma3nz_stddevNorm",
            "shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm",
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

# Features to never surface as coaching tips (too technical / noisy / unactionable)
SKIP_COACHING = {
    # Raw formant frequencies — anatomy, not behaviour
    "F2frequency_sma3nz_stddevNorm",
    "F3frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_stddevNorm",
    # Raw harmonic features — very technical
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    # Raw unvoiced spectral slopes — not intuitive
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    # Low-level MFCC details — confusing without context
    "mfcc4_sma3_stddevNorm",
    "mfcc4V_sma3nz_stddevNorm",
}


# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def get_coaching_tip(
    feat: str,
    val: float,
    zone: dict,
    direction: str,  # "positive" = this SHAP helped score; "negative" = hurt score
) -> tuple[str, str]:
    """
    Returns (tip_text, status_colour) for one feature.

    status_colour is one of: "green", "yellow", "red"
    direction reflects SHAP sign, NOT the feature's natural direction.
    """
    friendly = FRIENDLY_NAMES.get(feat, feat)
    zone_min = zone.get("min", -99)
    zone_max = zone.get("max",  99)
    feat_dir = zone.get("direction", "INCREASING")
    in_zone  = zone_min <= val <= zone_max

    # ── Positive SHAP: feature is helping the score ───────────────────────────
    if direction == "positive":
        if in_zone or val >= zone_min:
            return f"Your {friendly} is a strength — it's well-calibrated.", "green"
        return f"Your {friendly} is on the right track. Keep improving it.", "yellow"

    # ── Negative SHAP: feature is hurting the score ───────────────────────────

    # ── Loudness / Energy tips ────────────────────────────────────────────────
    if "loudness" in feat.lower() or "vocal_projection" in feat:
        if "stddev" in feat or "variation" in feat.lower() or "Slope" in feat:
            if val < zone_min:
                return (
                    f"Your {friendly} is very flat — vary your volume more to sound dynamic.",
                    "red",
                )
            if val > zone_max:
                return (
                    f"Your {friendly} is too erratic — work on smoother volume control.",
                    "red",
                )
        else:
            if feat_dir == "INCREASING" and val < zone_min:
                return f"Your {friendly} is too low — speak louder and project more.", "red"
            if feat_dir == "DECREASING" and val > zone_max:
                return f"Your {friendly} is too intense — dial back the volume slightly.", "red"
        return f"Your {friendly} could be better balanced for a confident delivery.", "yellow"

    if feat == "loudness_dynamics_power":
        if val < zone_min:
            return (
                "Your loudness dynamics are weak — add more peaks and energy variation.",
                "red",
            )
        return "Your loudness dynamics could be more expressive.", "yellow"

    # ── Pitch / Expressiveness tips ───────────────────────────────────────────
    if "F0semitone" in feat:
        if val < zone_min:
            return (
                "Your pitch variation is very low — you sound monotone. "
                "Try raising and lowering your voice to emphasise key points.",
                "red",
            )
        if val > zone_max:
            return (
                "Your pitch varies too wildly — aim for controlled, purposeful intonation.",
                "yellow",
            )
        return "Your pitch expressiveness could be slightly improved.", "yellow"

    if "spectralFlux" in feat:
        if val < zone_min:
            return (
                "Your speech lacks spectral variety — it sounds static. "
                "Vary your tone and articulation.",
                "red",
            )
        return "Your tonal variety could be more expressive.", "yellow"

    if "brightness_contrast" in feat:
        if val < zone_min:
            return (
                "Your voice lacks high-frequency brightness — "
                "work on clearer articulation and more forward placement.",
                "yellow",
            )
        if val > zone_max:
            return (
                "Your voice is overly bright/harsh. "
                "A slightly warmer tone will sound more natural.",
                "yellow",
            )
        return "Your spectral brightness balance could be adjusted.", "yellow"

    if "alphaRatio" in feat:
        if val < zone_min:
            return (
                "Your voice has low high-frequency energy — "
                "it may sound muffled. Work on clear, forward articulation.",
                "yellow",
            )
        return f"Your {friendly} could be adjusted for a cleaner sound.", "yellow"

    if "hammarberg" in feat.lower() or "slope" in feat.lower():
        return f"Your {friendly} could be better calibrated for confident delivery.", "yellow"

    # ── Voice Quality tips ────────────────────────────────────────────────────
    if "HNR" in feat:
        if val < zone_min:
            return (
                "Your voice clarity (HNR) is low — it sounds breathy or rough. "
                "Support your breath and engage your core when speaking.",
                "red",
            )
        return "Your voice clarity could be slightly improved.", "yellow"

    if "shimmer" in feat.lower():
        if "stddev" in feat:
            if val < zone_min:
                return (
                    "Your shimmer consistency is too low — "
                    "your voice amplitude is unpredictably variable.",
                    "yellow",
                )
            if val > zone_max:
                return (
                    "Your shimmer is erratic — focus on steady, supported phonation.",
                    "red",
                )
        else:
            if val > zone_max:
                return (
                    "Your voice shimmer is high — it sounds unsteady. "
                    "Work on breath support and vocal stability.",
                    "red",
                )
            if val < zone_min:
                return (
                    "Your shimmer is unusually low — your voice may sound overly rigid.",
                    "yellow",
                )
        return "Your voice steadiness could be improved.", "yellow"

    if "vocal_instability" in feat:
        if val > zone_max:
            return (
                "Your voice is unstable (high jitter × shimmer). "
                "Focus on steady breath support and relaxed phonation.",
                "red",
            )
        if val < zone_min:
            return (
                "Your vocal instability is unusually low — "
                "ensure it reflects natural speech rather than suppressed movement.",
                "yellow",
            )
        return "Your vocal stability could be slightly improved.", "yellow"

    if "mfcc" in feat.lower():
        if val < zone_min:
            return (
                f"Your {friendly} is too uniform — "
                "vary your articulation and resonance for a richer vocal quality.",
                "yellow",
            )
        if val > zone_max:
            return (
                f"Your {friendly} is too erratic — "
                "aim for consistent, controlled articulation.",
                "red",
            )
        return f"Your {friendly} could be refined for better vocal quality.", "yellow"

    if "F2amplitude" in feat or "F3amplitude" in feat:
        if val < zone_min:
            return f"Your {friendly} is weak — work on clearer vowel resonance.", "yellow"
        if val > zone_max:
            return f"Your {friendly} is too prominent — balance your resonance.", "yellow"
        return f"Your {friendly} could be better balanced.", "yellow"

    # ── Fluency & Flow tips ───────────────────────────────────────────────────
    if "voiced_flow" in feat:
        if val < zone_min:
            return (
                "Your voiced speech flow is low — you have too many pauses or short segments. "
                "Work on longer, connected phrases.",
                "red",
            )
        if val > zone_max:
            return (
                "Your voiced flow is very high — you may be speaking without enough breathing space. "
                "Allow natural pauses for impact.",
                "yellow",
            )
        return "Your speech flow could be improved for better rhythm.", "yellow"

    if "MeanUnvoicedSegmentLength" in feat:
        if val > zone_max:
            return (
                "Your pauses are too long — they break your flow and lose listener attention. "
                "Aim for shorter, purposeful pauses.",
                "red",
            )
        if val < zone_min:
            return (
                "Your pauses are very short — allow brief moments of silence "
                "to let key points land.",
                "yellow",
            )
        return "Your pause length could be better calibrated.", "yellow"

    # ── Generic fallback ──────────────────────────────────────────────────────
    if val < zone_min:
        if feat_dir == "INCREASING":
            return f"Your {friendly} is below the ideal range — try to increase it.", "red"
        elif feat_dir == "DECREASING":
            return f"Your {friendly} is below the ideal range — this needs adjustment.", "yellow"
        return f"Your {friendly} is outside the ideal range — try to find the sweet spot.", "yellow"

    if val > zone_max:
        if feat_dir == "DECREASING":
            return f"Your {friendly} is above the ideal — dial it back slightly.", "red"
        return f"Your {friendly} is a bit high — a more moderate level works best.", "yellow"

    return f"Your {friendly} could be slightly adjusted.", "yellow"


# ══════════════════════════════════════════════════════════════════════════════
# ELIGIBLE FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_eligible_features(
    shap_dict: dict,
    group_members: list,
    feature_values: dict,
    direction: str,
    overall_score: float,
) -> list:
    """
    Filters and sorts features for coaching tip generation.
    direction: "positive" or "negative"
    """
    if direction == "negative":
        sensitivity = 0.001 if overall_score < 0.55 else (0.010 if overall_score > 0.80 else 0.004)
    else:
        sensitivity = 0.003

    eligible = []
    for feat, shap_val in shap_dict.items():
        if feat not in group_members:
            continue
        if feat in SKIP_COACHING:
            continue

        val     = feature_values.get(feat, 0)
        zone    = GOLDEN_ZONES.get(feat, {"min": -99, "max": 99})
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


# ══════════════════════════════════════════════════════════════════════════════
# GROUP RESULTS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_group_results(
    shap_dict: dict,
    feature_values: dict,
    overall_score: float,
    base_value: float,
) -> dict:
    group_results = {}
    baseline_pct       = base_value * 100
    explainer_prediction = baseline_pct

    for group_name, group_info in FEEDBACK_GROUPS.items():
        members = group_info["members"]

        group_shap_sum = sum(
            v for f, v in shap_dict.items() if f in members
        )
        impact_points = group_shap_sum * 100
        explainer_prediction += impact_points

        if   group_shap_sum >= group_info["yellow_threshold"]: status = "green"
        elif group_shap_sum >= group_info["red_threshold"]:    status = "yellow"
        else:                                                  status = "red"

        neg_features = get_eligible_features(
            shap_dict, members, feature_values, "negative", overall_score
        )
        pos_features = get_eligible_features(
            shap_dict, members, feature_values, "positive", overall_score
        )

        if   status == "green":  p_limit, n_limit = 3, 1
        elif status == "yellow": p_limit, n_limit = 2, 2
        else:                    p_limit, n_limit = 1, 3

        tips = []
        seen: set = set()

        def process_tips(features, dir_type, limit):
            count = 0
            for feat, shap_val in features:
                if count >= limit or feat in seen:
                    continue
                val  = feature_values.get(feat, 0)
                zone = GOLDEN_ZONES.get(feat, {})
                tip_text, tip_status = get_coaching_tip(feat, val, zone, dir_type)
                tips.append({
                    "feature":   feat,
                    "friendly":  FRIENDLY_NAMES.get(feat, feat),
                    "shap":      round(shap_val, 4),
                    "value":     round(val, 4),
                    "direction": dir_type,
                    "tip":       tip_text,
                    "status":    tip_status,
                })
                seen.add(feat)
                count += 1

        if status == "red":
            process_tips(neg_features, "negative", n_limit)
            process_tips(pos_features, "positive", p_limit)
        else:
            process_tips(pos_features, "positive", p_limit)
            process_tips(neg_features, "negative", n_limit)

        # Debug table
        debug_info = []
        for feat in shap_dict:
            if feat in members and feat not in SKIP_COACHING:
                zone = GOLDEN_ZONES.get(feat, {})
                debug_info.append({
                    "feat": feat,
                    "val":  round(feature_values.get(feat, 0), 4),
                    "shap": round(shap_dict[feat], 4),
                    "min":  round(zone.get("min", 0), 4),
                    "max":  round(zone.get("max", 0), 4),
                    "dir":  zone.get("direction", "N/A"),
                })

        group_results[group_name] = {
            "impact_points": round(impact_points, 1),
            "status":        status,
            "tips":          tips,
            "debug":         debug_info,
        }

    group_results["_metadata"] = {
        "baseline":    round(baseline_pct, 1),
        "synergy_gap": round((overall_score * 100) - explainer_prediction, 1),
    }
    return group_results


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_results(result: dict):
    raw_score = result["raw_score"]
    cal_score = result["calibrated_score"]
    metadata  = result["group_results"].get("_metadata", {})

    print(f"\n   Raw Score        : {raw_score * 100:.1f}%")
    print(f"   Calibrated Score : {cal_score * 100:.1f}%")
    print(f"   {'━' * 100}")

    for name, data in result["group_results"].items():
        if name == "_metadata":
            continue
        icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[data["status"]]
        print(f"\n   {icon} {name} ({data['impact_points']:+.1f}%)")

        for t in data["tips"]:
            marker = {"green": "✓", "yellow": "→"}.get(t["status"], "×")
            print(f"      {marker} [{t['friendly']}] {t['tip']}")

        # Engineering debug table
        print(f"\n      [DEBUG — {name}]")
        print(f"      {'Feature':<50} | {'Value':>8} | {'Zone':>20} | {'SHAP':>8} | Dir")
        print(f"      {'-'*50}-+-{'-'*8}-+-{'-'*20}-+-{'-'*8}-+-{'-'*12}")
        for d in sorted(data["debug"], key=lambda x: abs(x["shap"]), reverse=True):
            zone_str = f"[{d['min']}, {d['max']}]"
            print(
                f"      {d['feat']:<50} | {d['val']:>8.4f} | "
                f"{zone_str:<20} | {d['shap']:>8.4f} | {d['dir']}"
            )

    print(f"\n   {'━' * 100}")
    print(f"      Baseline    : {metadata.get('baseline')}%")
    print(f"      Synergy Gap : {metadata.get('synergy_gap')}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(feature_dict: dict) -> dict:
    """
    Full pipeline for one audio sample (already feature-extracted).
    Returns the complete result dict (score + group_results).
    """
    scored = score_sample(feature_dict)

    group_results = build_group_results(
        shap_dict     = scored["shap_dict"],
        feature_values= scored["feature_values"],
        overall_score = scored["calibrated_score"],
        base_value    = scored["base_value"],
    )

    return {
        "raw_score":        scored["raw_score"],
        "calibrated_score": scored["calibrated_score"],
        "group_results":    group_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", required=True,
        help="CSV file with one row per audio sample (must have the 45 model features)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} sample(s) from {args.features}\n")

    for i, row in df.iterrows():
        file_id = row.get("file", f"sample_{i}")
        feature_dict = row.drop(labels=["file", "confidence_score"], errors="ignore").to_dict()

        print(f"\n{'═' * 110}")
        print(f"  Analysing: {file_id}")
        result = process_sample(feature_dict)
        display_results(result)


if __name__ == "__main__":
    main()