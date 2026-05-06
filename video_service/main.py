from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import json
import os
import time
from pathlib import Path
from datetime import datetime
from upstash_redis import Redis
import logging
import warnings
import joblib

warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = {}

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "models"

LSTM_MODEL_PATH         = str(_MODEL_DIR / "video_lstm_v2.onnx")
EXPLAINER_MODEL_PATH    = str(_MODEL_DIR / "video_explainer_model_v2.pkl")
EXPLAINER_FEATURES_PATH = str(_MODEL_DIR / "video_explainer_features_v2.pkl")
SCALER_PATH             = str(_MODEL_DIR / "video_scaler_v2.pkl")
CALIBRATION_PATH        = str(_MODEL_DIR / "calibration_data_v8.json")

# ── Friendly names shown to the user ─────────────────────────────────────────
FRIENDLY_NAMES = {
    "mouth_expressiveness":      "Mouth Expressiveness",
    "head_nodding":              "Head Nodding",
    "head_tilt":                 "Head Position",
    "shoulder_alignment":        "Shoulder Posture",
    "forward_lean":              "Leaning Forward",
    "left_hand_velocity":        "Left Hand Movement",
    "right_hand_velocity":       "Right Hand Movement",
    "left_hand_expressiveness":  "Left Hand Gestures",
    "right_hand_expressiveness": "Right Hand Gestures",
    "body_stillness":            "Body Stillness",
    "hand_gesture_range":        "Gesture Range",
    "smile_intensity":           "Warmth & Smile",
    "gaze_consistency":          "Conversational Orientation",
}

SKIP_COACHING = {
    "hand_gesture_range_mean", "hand_gesture_range_std",
    "hand_gesture_range_min", "hand_gesture_range_max",
    "hand_gesture_range_cv", "hand_gesture_range_peak_ratio",
    "hand_gesture_range_skew_approx", "hand_gesture_range_stability",
    "left_hand_velocity_min", "right_hand_velocity_min",
    "left_hand_expressiveness_min", "right_hand_expressiveness_min",
    "body_stillness_min",
    "gaze_consistency_cv", "gaze_consistency_peak_ratio",
    "gaze_consistency_skew_approx", "gaze_consistency_stability",
}

ENGINEERED_SUFFIXES = ("_mean", "_std", "_max", "_cv", "_peak_ratio", "_skew_approx", "_stability")

FEEDBACK_GROUPS = {
    "Facial Engagement": {
        "members": ["mouth_expressiveness", "head_nodding", "smile_intensity", "gaze_consistency"],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.1600,
    },
    "Hand Gestures": {
        "members": ["left_hand_velocity", "right_hand_velocity",
                    "left_hand_expressiveness", "right_hand_expressiveness",
                    "hand_gesture_range"],
        "yellow_threshold": -0.0001,
        "red_threshold":    -0.1200,
    },
    "Posture & Presence": {
        "members": ["head_tilt", "shoulder_alignment", "forward_lean", "body_stillness"],
        "yellow_threshold": -0.0200,
        "red_threshold":    -0.0900,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE LABEL
# ═══════════════════════════════════════════════════════════════════════════════

def get_confidence_label(score: float) -> str:
    if score >= 0.75:  return "Highly Confident"
    if score >= 0.50:  return "Confident"
    if score >= 0.30:  return "Moderately Confident"
    return "Needs Improvement"


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_aggregated_stats(stats: dict) -> dict:
    enriched = dict(stats)
    base_names: set = set()
    for key in stats:
        for suffix in ("_mean", "_std", "_max"):
            if key.endswith(suffix):
                base_names.add(key[: -len(suffix)])

    for base in base_names:
        mean_v = stats.get(f"{base}_mean")
        std_v  = stats.get(f"{base}_std")
        max_v  = stats.get(f"{base}_max")
        if None in (mean_v, std_v, max_v):
            continue
        enriched[f"{base}_cv"]          = std_v / (abs(mean_v) + 1e-6)
        enriched[f"{base}_peak_ratio"]  = max_v / (abs(mean_v) + 1e-6)
        enriched[f"{base}_skew_approx"] = (max_v - mean_v) / (std_v + 1e-6)
        enriched[f"{base}_stability"]   = 1.0 / (std_v + 1e-6)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# COACHING FEEDBACK LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _direction_nudge(val: float, zone_min: float, zone_max: float, direction: str) -> str | None:
    """
    Returns a nudge toward the elite zone based on feature direction type.
    Used for POSITIVE SHAP + outside zone.
    """
    if direction == "INCREASING":
        if val < zone_min:
            return "You're doing well — push this a little further to reach the elite range."
        if val > zone_max:
            return "You're doing well — dial this back slightly to land in the elite range."

    elif direction == "DECREASING":
        if val > zone_max:
            return "You're doing well — reduce this slightly to land in the elite range."
        if val < zone_min:
            return "You're doing well — bring this back up slightly to reach the elite range."

    elif direction == "INVERTED_U":
        if val < zone_min:
            return "You're doing well — push this up slightly to reach the elite range."
        if val > zone_max:
            return "You're doing well — dial this back slightly to reach the elite range."

    return None  # already in zone


def _get_engineered_tip(culprit_feat: str, base_feature: str,
                         val: float, zone: dict, direction: str) -> tuple[str, str]:
    friendly  = FRIENDLY_NAMES.get(base_feature, base_feature)
    zone_min  = zone.get("min", -99)
    zone_max  = zone.get("max",  99)
    feat_dir  = zone.get("direction", "INCREASING")
    in_zone   = zone_min <= val <= zone_max

    # ── Positive SHAP: praise + nudge if outside zone ────────────────────────
    def positive_tip(base_praise: str) -> tuple[str, str]:
        if in_zone:
            return base_praise, "green"
        nudge = _direction_nudge(val, zone_min, zone_max, feat_dir)
        if nudge:
            return f"{base_praise} {nudge}", "yellow"
        return base_praise, "green"

    # ── Negative SHAP helpers ─────────────────────────────────────────────────
    def negative_tip_increasing() -> tuple[str, str]:
        if in_zone:
            return None, None
        if val < zone_min:
            return (
                f"Your {friendly} is below the elite range — "
                f"try to increase this to reach the ideal level.", "red"
            )
        return (
            f"Your {friendly} is past the elite range — "
            f"dial it back slightly to land in the ideal zone.", "yellow"
        )

    def negative_tip_decreasing() -> tuple[str, str]:
        if in_zone:
            return None, None
        if val > zone_max:
            return (
                f"Your {friendly} is too high — "
                f"reduce this to reach the elite range.", "red"
            )
        return (
            f"Your {friendly} is below the elite range — "
            f"bring it back up slightly toward the ideal level.", "yellow"
        )

    def negative_tip_inverted() -> tuple[str, str]:
        if in_zone:
            return None, None
        if val < zone_min:
            return (
                f"Your {friendly} is too low — "
                f"bring it up toward the middle elite range.", "red"
            )
        return (
            f"Your {friendly} is too high — "
            f"bring it down toward the middle elite range.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _mean
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_mean"):
        if direction == "positive":
            return positive_tip(f"Your overall {friendly} level is well-calibrated — a real strength.")
        if feat_dir == "INCREASING":
            return negative_tip_increasing()
        if feat_dir == "DECREASING":
            return negative_tip_decreasing()
        return negative_tip_inverted()

    # ══════════════════════════════════════════════════════════════════════════
    # _std
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_std"):
        if direction == "positive":
            return positive_tip(f"Your {friendly} has a natural, healthy variation throughout.")
        if feat_dir == "INCREASING":
            if in_zone:
                return None, None
            if val < zone_min:
                return (
                    f"Your {friendly} barely varies — it feels flat and robotic. "
                    f"Let it move more naturally as you speak.", "red"
                )
            return (
                f"Your {friendly} is swinging too much — the variation is distracting. "
                f"Focus on keeping it smoother and more controlled.", "yellow"
            )
        if feat_dir == "DECREASING":
            if in_zone:
                return None, None
            if val > zone_max:
                return (
                    f"Your {friendly} variation is too high — "
                    f"try to keep it more controlled and consistent.", "red"
                )
            return (
                f"Your {friendly} has almost no variation — "
                f"allow a little natural movement.", "yellow"
            )
        # INVERTED_U
        if in_zone:
            return None, None
        if val < zone_min:
            return (
                f"Your {friendly} barely varies — it feels unnatural. "
                f"Let it shift a little more freely.", "red"
            )
        return (
            f"Your {friendly} is varying too much — "
            f"aim for smoother, more consistent delivery.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _max
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_max"):
        if direction == "positive":
            return positive_tip(f"Your {friendly} reaches strong peak moments — great emphasis.")
        if feat_dir == "INCREASING":
            if in_zone:
                return None, None
            if val < zone_min:
                return (
                    f"Your {friendly} never really peaks — your delivery lacks emphasis moments. "
                    f"Push it further at key points to hold attention.", "red"
                )
            return (
                f"Your {friendly} peaks too high at times — it can feel overwhelming. "
                f"Keep your strongest moments more controlled.", "yellow"
            )
        if feat_dir == "DECREASING":
            if in_zone:
                return None, None
            if val > zone_max:
                return (
                    f"Your {friendly} peaks too high — "
                    f"reduce the intensity at your strongest moments.", "red"
                )
            return (
                f"Your {friendly} peak moments are very low — "
                f"allow slightly more range at key points.", "yellow"
            )
        # INVERTED_U
        if in_zone:
            return None, None
        if val < zone_min:
            return (
                f"Your {friendly} peak moments are too low — "
                f"push it a little further at key points.", "red"
            )
        return (
            f"Your {friendly} peaks too high — "
            f"keep your emphasis moments more controlled.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _stability
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_stability"):
        if direction == "positive":
            return positive_tip(f"Your {friendly} is impressively consistent throughout — a real strength.")
        if in_zone:
            return None, None
        if feat_dir == "INCREASING":
            # higher stability = more consistent = better
            if val < zone_min:
                return (
                    f"Your {friendly} varies quite a lot between moments — it can look uncontrolled. "
                    f"Focus on keeping movements smooth and intentional.", "red"
                )
            return (
                f"Your {friendly} is almost completely frozen — it can look stiff. "
                f"Allow a little natural, gentle variation.", "yellow"
            )
        if feat_dir == "DECREASING":
            # lower stability = more variation = better for this feature
            if val < zone_min:
                # below zone on DECREASING = too stable/frozen
                return (
                    f"Your {friendly} is almost completely frozen — it can look stiff. "
                    f"Allow a little natural, gentle variation.", "yellow"
                )
            # above zone = too erratic
            return (
                f"Your {friendly} varies quite a lot between moments — it can look uncontrolled. "
                f"Focus on keeping movements smooth and intentional.", "red"
            )
        # INVERTED_U — middle is best
        if val < zone_min:
            return (
                f"Your {friendly} is almost completely frozen — "
                f"allow a little more natural variation.", "yellow"
            )
        return (
            f"Your {friendly} varies too much — "
            f"aim for smoother, more consistent delivery.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _cv
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_cv"):
        if direction == "positive":
            return positive_tip(f"The rhythm of your {friendly} is well-balanced — varied but controlled.")
        if in_zone:
            return None, None
        if feat_dir == "INCREASING":
            # higher cv = more relative variation = better
            if val < zone_min:
                return (
                    f"Your {friendly} barely changes at all. "
                    f"Let it vary naturally as you shift between ideas.", "yellow"
                )
            return (
                f"Your {friendly} is all over the place — it varies too unpredictably. "
                f"Aim for consistent, purposeful movements.", "red"
            )
        if feat_dir == "DECREASING":
            # lower cv = less variation = better
            if val > zone_max:
                return (
                    f"Your {friendly} is all over the place — it varies too unpredictably. "
                    f"Aim for consistent, purposeful movements.", "red"
                )
            return (
                f"Your {friendly} barely changes at all. "
                f"Let it vary naturally as you shift between ideas.", "yellow"
            )
        # INVERTED_U
        if val < zone_min:
            return (
                f"Your {friendly} barely changes at all. "
                f"Let it vary naturally as you shift between ideas.", "yellow"
            )
        return (
            f"Your {friendly} is all over the place — it varies too unpredictably. "
            f"Aim for consistent, purposeful movements.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _peak_ratio
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_peak_ratio"):
        if direction == "positive":
            return positive_tip(f"Your {friendly} peaks are well-proportioned — you have clear emphasis moments.")
        if in_zone:
            return None, None
        if feat_dir == "INCREASING":
            # higher peak ratio = stronger peaks relative to baseline = better
            if val < zone_min:
                return (
                    f"Your {friendly} lacks standout moments. "
                    f"Add occasional emphasis at key points to hold attention.", "red"
                )
            return (
                f"Your {friendly} spikes very high at times but stays low otherwise. "
                f"Aim for a steadier, higher baseline level.", "red"
            )
        if feat_dir == "DECREASING":
            # lower peak ratio = better
            if val > zone_max:
                return (
                    f"Your {friendly} spikes very high at times but stays low otherwise. "
                    f"Aim for a steadier, higher baseline level.", "red"
                )
            return (
                f"Your {friendly} lacks standout moments. "
                f"Add occasional emphasis at key points to hold attention.", "red"
            )
        # INVERTED_U
        if val < zone_min:
            return (
                f"Your {friendly} lacks standout moments. "
                f"Add occasional emphasis at key points to hold attention.", "red"
            )
        return (
            f"Your {friendly} spikes very high at times but stays low otherwise. "
            f"Aim for a steadier, higher baseline level.", "red"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # _skew_approx
    # ══════════════════════════════════════════════════════════════════════════
    if culprit_feat.endswith("_skew_approx"):
        if direction == "positive":
            return positive_tip(f"The distribution of your {friendly} is well-balanced.")
        if in_zone:
            return None, None
        if feat_dir == "INCREASING":
            if val < zone_min:
                return (
                    f"Your {friendly} is at its peak almost constantly. "
                    f"Vary it more so emphasis moments actually stand out.", "yellow"
                )
            return (
                f"Your {friendly} stays flat most of the time with sudden spikes. "
                f"Try to maintain a higher, steadier baseline instead.", "red"
            )
        if feat_dir == "DECREASING":
            # lower skew = better
            if val > zone_max:
                return (
                    f"Your {friendly} stays flat most of the time with sudden spikes. "
                    f"Try to maintain a higher, steadier baseline instead.", "red"
                )
            return (
                f"Your {friendly} is at its peak almost constantly. "
                f"Vary it more so emphasis moments actually stand out.", "yellow"
            )
        # INVERTED_U
        if val < zone_min:
            return (
                f"Your {friendly} is at its peak almost constantly. "
                f"Vary it more so emphasis moments actually stand out.", "yellow"
            )
        return (
            f"Your {friendly} stays flat most of the time with sudden spikes. "
            f"Try to maintain a higher, steadier baseline instead.", "red"
        )

    return f"Your {friendly} could be adjusted for better impact.", "yellow"


def get_granular_feedback(culprit_feat: str, base_feature: str, val: float,
                           zone: dict, direction: str = "negative") -> tuple[str, str]:
    friendly = FRIENDLY_NAMES.get(base_feature, base_feature)
    is_hand  = "hand" in base_feature
    zone_min = zone.get("min", -99) if zone else -99
    zone_max = zone.get("max",  99) if zone else  99
    in_zone  = zone_min <= val <= zone_max

    # ── Hand not detected — check FIRST before anything else ─────────────────
    if is_hand and zone and val <= (zone_min + 0.10):
        if direction == "positive":
            return (
                f"Your hand movement was very minimal during the interview. "
                f"Subtle gestures still help — try letting your hands move naturally as you speak.", "yellow"
            )
        return (
            f"Your hand gestures were very limited during the interview. "
            f"Natural hand movement while speaking helps you look more engaged and confident — "
            f"try letting your hands move freely as you would in a normal conversation.", "red"
        )

    # ── Route ALL engineered features including _mean, _std, _max ────────────
    if any(culprit_feat.endswith(s) for s in ENGINEERED_SUFFIXES):
        tip, status = _get_engineered_tip(culprit_feat, base_feature, val, zone, direction)
        if tip is None:
            return (
                f"Your {friendly} is within the ideal range — keep doing what you're doing.", "yellow"
            )
        return tip, status

    # ── Negative SHAP + already in zone ──────────────────────────────────────
    if direction == "negative" and in_zone:
        return (
            f"Your {friendly} is within the ideal range — keep doing what you're doing.", "yellow"
        )

    # ── Positive SHAP tips ────────────────────────────────────────────────────
    if direction == "positive":
        nudge = None if in_zone else _direction_nudge(val, zone_min, zone_max, zone.get("direction", "INCREASING"))
        if base_feature == "mouth_expressiveness":
            base = "Your face is expressive and animated — you look engaged and natural."
        elif base_feature == "head_nodding":
            base = "You nod at the right moments — you look attentive and engaged."
        elif base_feature == "smile_intensity":
            base = "Your warmth comes through naturally — you look approachable and genuine."
        elif base_feature == "gaze_consistency":
            base = "Your eye contact is steady and natural — you look focused and confident."
        elif base_feature in ("left_hand_velocity", "right_hand_velocity"):
            side = "left" if "left" in base_feature else "right"
            base = f"Your {side} hand moves at a good pace — your gestures feel natural and purposeful."
        elif base_feature in ("left_hand_expressiveness", "right_hand_expressiveness"):
            side = "left" if "left" in base_feature else "right"
            base = f"Your {side} hand gestures add emphasis and make you look engaged."
        elif base_feature == "hand_gesture_range":
            base = "Your gesture range is well-sized — not too small, not too large."
        elif base_feature == "body_stillness":
            base = "You hold yourself steady — you look composed and in control."
        elif base_feature == "shoulder_alignment":
            base = "Your shoulders are well-aligned — your posture looks professional and open."
        elif base_feature == "forward_lean":
            base = "Your forward lean shows engagement — you look interested and present."
        elif base_feature == "head_tilt":
            base = "Your head position is well-balanced — you look attentive and even-keeled."
        else:
            base = f"Your {friendly} is a key strength — it's well-calibrated and professional."
        if nudge:
            return f"{base} {nudge}", "yellow"
        return base, "green"

    # ══════════════════════════════════════════════════════════════════════════
    # NEGATIVE SHAP TIPS — specific, plain-English, actionable
    # ══════════════════════════════════════════════════════════════════════════

    # ── FACIAL EXPRESSIVENESS ─────────────────────────────────────────────────
    if base_feature == "mouth_expressiveness":
        if val < zone_min:
            return (
                "Your face looks quite still during the interview — minimal expression. "
                "Try letting your face react naturally to what you're saying: raise your eyebrows "
                "slightly on a key point, nod as you listen, let a small smile sit naturally. "
                "Expressiveness signals enthusiasm and keeps the interviewer engaged.", "red"
            )
        if val > zone_max:
            return (
                "Your facial expressions are very intense — it can feel overdone on camera. "
                "React naturally rather than performing — subtle, genuine reactions "
                "read better than big, exaggerated ones.", "yellow"
            )
        return "Your facial expressiveness could be slightly more animated for a warmer impression.", "yellow"

    # ── HEAD NODDING ─────────────────────────────────────────────────────────
    if base_feature == "head_nodding":
        if val < zone_min:
            return (
                "You're barely nodding throughout the interview. "
                "A slow, deliberate nod when the interviewer speaks — or when you make a key point — "
                "signals active listening and confidence. You don't need to nod constantly, "
                "just at meaningful moments.", "red"
            )
        if val > zone_max:
            return (
                "You're nodding very frequently — it can look nervous or overly eager. "
                "Reserve head nods for when you genuinely agree or want to show you're listening. "
                "Stillness between nods looks much more confident.", "yellow"
            )
        return "Your head nodding could be slightly more deliberate and purposeful.", "yellow"

    # ── SMILE INTENSITY ───────────────────────────────────────────────────────
    if base_feature == "smile_intensity":
        if val < zone_min:
            return (
                "You're barely smiling throughout the interview — "
                "you may come across as cold or disengaged, even if that's not your intention. "
                "You don't need to force it — just let a slight, genuine smile sit on your face "
                "while you're listening. It makes a big difference to how approachable you look.", "red"
            )
        if val > zone_max:
            return (
                "You're smiling almost constantly — which can feel unnatural or performative on camera. "
                "Let your expression match the moment: smile warmly when appropriate, "
                "but allow a neutral, attentive look when discussing serious points.", "yellow"
            )
        return "Your smile could be a little more natural and consistent throughout.", "yellow"

    # ── EYE CONTACT ───────────────────────────────────────────────────────────
    if base_feature == "gaze_consistency":
        if val < zone_min:
            return (
                "Your gaze is moving around quite a bit — away from the camera. "
                "In a video interview, looking at the camera lens is the equivalent of eye contact. "
                "Try placing a small sticker just above your camera as a reminder to look there, "
                "especially when answering questions.", "red"
            )
        if val > zone_max:
            return (
                "Your gaze is very fixed — it can look intense or unnatural. "
                "It's okay to look away briefly when thinking. "
                "Natural eye contact involves occasional breaks, then returning to the camera.", "yellow"
            )
        return "Your eye contact could be slightly more consistent throughout the interview.", "yellow"

    # ── HAND VELOCITY ─────────────────────────────────────────────────────────
    if base_feature in ("left_hand_velocity", "right_hand_velocity"):
        side = "left" if "left" in base_feature else "right"
        if val < zone_min:
            return (
                f"Your {side} hand is mostly still during the interview. "
                f"Natural hand movement while speaking helps emphasise your points and "
                f"makes you look more engaged. Try letting your hands move freely "
                f"as you would in a normal face-to-face conversation.", "red"
            )
        if val > zone_max:
            return (
                f"Your {side} hand is moving very quickly and frequently — "
                f"it can be distracting. Slow down your gestures and make them more deliberate. "
                f"A few well-timed, controlled movements land better than constant motion.", "yellow"
            )
        return f"Your {side} hand movement could be slightly more deliberate.", "yellow"

    # ── HAND EXPRESSIVENESS ───────────────────────────────────────────────────
    if base_feature in ("left_hand_expressiveness", "right_hand_expressiveness"):
        side = "left" if "left" in base_feature else "right"
        if val < zone_min:
            return (
                f"Your {side} hand isn't contributing much — it's not adding any gestural emphasis. "
                f"Try using it to illustrate your points: "
                f"open palm when explaining, a gentle chop motion to emphasise, "
                f"counting on fingers for lists.", "red"
            )
        if val > zone_max:
            return (
                f"Your {side} hand is very active — the gestures are drawing attention away "
                f"from what you're saying. Pull back slightly and make each gesture deliberate.", "yellow"
            )
        return f"Your {side} hand expressiveness could be slightly more purposeful.", "yellow"

    # ── HAND GESTURE RANGE ────────────────────────────────────────────────────
    if base_feature == "hand_gesture_range":
        if val < zone_min:
            return (
                "Your hand gestures are very small and tight — "
                "they're barely visible and have little impact. "
                "Try opening your gestures up: let your hands move to chest height "
                "and out from your body slightly. Bigger but controlled gestures "
                "project more confidence.", "red"
            )
        if val > zone_max:
            return (
                "Your gestures are very large and wide — "
                "they can feel overwhelming on a video call. "
                "Keep movements closer to your body and at chest level for a more "
                "controlled, professional look.", "yellow"
            )
        return "Your gesture range could be slightly better calibrated.", "yellow"

    # ── BODY STILLNESS ────────────────────────────────────────────────────────
    if base_feature == "body_stillness":
        if val < zone_min:
            return (
                "You're moving around quite a bit in your seat — "
                "swaying, shifting, or fidgeting. This can make you look nervous or distracted. "
                "Try to sit still and upright, with your feet flat on the floor. "
                "Stillness reads as confidence on camera.", "red"
            )
        if val > zone_max:
            return (
                "You're almost completely frozen — which can look tense or uncomfortable. "
                "It's okay to shift slightly between points. "
                "A relaxed, natural stillness looks better than a rigid one.", "yellow"
            )
        return "Your body stillness could be slightly more grounded.", "yellow"

    # ── SHOULDER ALIGNMENT ────────────────────────────────────────────────────
    if base_feature == "shoulder_alignment":
        if val < zone_min:
            return (
                "Your shoulders are noticeably uneven — one is higher than the other. "
                "Try to sit squarely facing the camera, with both shoulders level. "
                "Uneven posture can look like tension or discomfort.", "red"
            )
        if val > zone_max:
            return (
                "Your shoulders are very tense or raised — it can look like you're stressed. "
                "Try dropping your shoulders, take a breath, and let them relax before answering.", "yellow"
            )
        return "Your shoulder posture could be slightly more open and relaxed.", "yellow"

    # ── FORWARD LEAN ─────────────────────────────────────────────────────────
    if base_feature == "forward_lean":
        if val < zone_min:
            return (
                "You're leaning back or sitting very upright — "
                "you may come across as disengaged or distant. "
                "Try leaning in very slightly when answering — "
                "just enough to show you're present and interested.", "red"
            )
        if val > zone_max:
            return (
                "You're leaning quite far forward — it can look intense or uncomfortable on camera. "
                "Sit back slightly to a neutral, upright position. "
                "A gentle lean-in at key moments is powerful; constant forward lean is overwhelming.", "yellow"
            )
        return "Your forward lean could be slightly more natural.", "yellow"

    # ── HEAD TILT ─────────────────────────────────────────────────────────────
    if base_feature == "head_tilt":
        if val < zone_min:
            return (
                "Your head is tilting quite significantly to one side. "
                "A slight tilt can show curiosity, but too much can look uncertain. "
                "Try to keep your head more level — especially when making important points.", "red"
            )
        if val > zone_max:
            return (
                "Your head is tilted very far — it can look awkward or uncomfortable. "
                "Try to keep your head relatively level and centered.", "red"
            )
        return "Your head position could be slightly more centered.", "yellow"

    # ── Generic fallback ──────────────────────────────────────────────────────
    if val < zone_min:
        return (
            f"Your {friendly} is lower than the ideal range — "
            f"try to show a bit more of this throughout the interview.", "red"
        )
    if val > zone_max:
        return (
            f"Your {friendly} is a bit higher than the ideal range — "
            f"dialling it back slightly will make your delivery feel more natural.", "yellow"
        )
    return f"Your {friendly} could be slightly adjusted to feel more natural.", "yellow"


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def get_eligible_features(shap_dict, group_members, scaled_stats_dict, direction, overall_score):
    eligible = []
    if direction == "negative":
        if overall_score < 0.65:
            sensitivity = 0.001
        elif overall_score > 0.80:
            sensitivity = 0.015
        else:
            sensitivity = 0.005
    else:
        sensitivity = 0.003

    for feat, shap_val in shap_dict.items():
        if not any(feat.startswith(m) for m in group_members):
            continue
        if feat in SKIP_COACHING:
            continue

        val     = scaled_stats_dict.get(feat, 0)
        zone    = models["golden_zones"].get(feat, {"min": -99, "max": 99})
        in_zone = (zone["min"] <= val <= zone["max"])

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


def build_group_results(shap_values_dict, scaled_stats_dict, overall_score, base_value):
    bv = float(base_value[0]) if isinstance(base_value, (list, np.ndarray)) else float(base_value)
    group_results = {}

    for group_name, group_info in FEEDBACK_GROUPS.items():
        members = group_info["members"]
        group_shap_sum = sum(
            v for f, v in shap_values_dict.items()
            if any(f.startswith(m) for m in members)
        )
        impact_points = group_shap_sum * 100

        if group_shap_sum >= group_info["yellow_threshold"]:
            status = "green"
        elif group_shap_sum >= group_info["red_threshold"]:
            status = "yellow"
        else:
            status = "red"

        neg_features = get_eligible_features(
            shap_values_dict, members, scaled_stats_dict, "negative", overall_score)
        pos_features = get_eligible_features(
            shap_values_dict, members, scaled_stats_dict, "positive", overall_score)

        if status == "green":
            p_limit, n_limit = 4, 1
        elif status == "yellow":
            p_limit, n_limit = 3, 3
        else:
            p_limit, n_limit = 1, 4

        tips = []
        seen_bases: set = set()

        def process_tips(features, dir_type, limit):
            count = 0
            for feat, shap_val in features:
                if count >= limit:
                    break
                base = next((m for m in members if feat.startswith(m)), feat)
                if base in seen_bases:
                    continue

                culprit_val  = scaled_stats_dict.get(feat, 0)
                culprit_zone = models["golden_zones"].get(feat, {})

                tip_text, tip_status = get_granular_feedback(
                    feat, base, culprit_val, culprit_zone, direction=dir_type)
                if tip_text is None:
                    continue

                tips.append({
                    "feature":        feat,
                    "base":           base,
                    "friendly":       FRIENDLY_NAMES.get(base, base),
                    "shap":           round(float(shap_val), 4),
                    "direction":      dir_type,
                    "tip":            tip_text,
                    "status":         tip_status,
                    "val":            round(float(culprit_val), 4),
                    "zone_min":       round(float(culprit_zone.get("min", -99)), 4),
                    "zone_max":       round(float(culprit_zone.get("max",  99)), 4),
                    "zone_direction": culprit_zone.get("direction", "INCREASING"),
                })
                seen_bases.add(base)
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

    return group_results


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_video_v2(video_path: str) -> dict:
    from feature_extraction_v2 import (
        LandmarkExtractor, SmileExtractor,
        extract_all_frames, normalize, engineer_features, aggregate_features,
        WINDOW_SIZE, STEP_SIZE, FACE_LANDMARKER_PATH,
    )

    all_frames = extract_all_frames(video_path)
    if len(all_frames) < WINDOW_SIZE:
        all_frames = np.concatenate(
            [all_frames, np.repeat(all_frames[-1:], WINDOW_SIZE - len(all_frames), axis=0)],
            axis=0,
        )

    extractor     = LandmarkExtractor()
    all_landmarks = extractor.extract(list(all_frames))
    extractor.close()

    smile_ext  = SmileExtractor(model_path=FACE_LANDMARKER_PATH)
    all_smiles = smile_ext.extract_smile_scores(all_frames)
    smile_ext.close()

    lstm_session       = models["lstm_session"]
    scaler             = models["scaler"]
    explainer_features = models["explainer_features"]
    shap_explainer     = models["shap_explainer"]

    master_features, window_scores = [], []
    starts = list(range(0, len(all_frames) - WINDOW_SIZE + 1, STEP_SIZE)) or [0]

    for start in starts:
        end          = start + WINDOW_SIZE
        norm_dicts, scales = normalize(all_landmarks[start:end])
        feat_array   = scaler.transform(
            engineer_features(norm_dicts, scales, all_smiles[start:end])
        )
        input_tensor = feat_array[np.newaxis, :, :].astype(np.float32)
        raw = float(lstm_session.run(None, {"video_features": input_tensor})[0][0])

        min_lstm = models["min_lstm"]
        max_lstm = models["max_lstm"]
        window_scores.append(np.clip((raw - min_lstm) / (max_lstm - min_lstm), 0.0, 1.0))
        master_features.append(feat_array)

    raw_stats      = aggregate_features(np.vstack(master_features))
    enriched_stats = enrich_aggregated_stats(raw_stats)

    X_single  = pd.DataFrame([enriched_stats])[explainer_features]
    shap_vals = shap_explainer.shap_values(X_single)[0]
    shap_dict = {feat: float(val) for feat, val in zip(explainer_features, shap_vals)}

    overall_score = float(np.mean(window_scores))

    group_results = build_group_results(
        shap_dict,
        enriched_stats,
        overall_score,
        shap_explainer.expected_value,
    )

    return {
        "scaled_score":  overall_score,
        "group_results": group_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("VIDEO ANALYSIS SERVICE V2 STARTING...")
    print("=" * 60)
    try:
        import onnxruntime as ort
        import shap

        models["lstm_session"] = ort.InferenceSession(LSTM_MODEL_PATH)
        print(f"   ONNX LSTM v2 loaded")

        expl_model                   = joblib.load(EXPLAINER_MODEL_PATH)
        models["explainer_features"] = joblib.load(EXPLAINER_FEATURES_PATH)
        models["scaler"]             = joblib.load(SCALER_PATH)
        models["shap_explainer"]     = shap.TreeExplainer(expl_model)
        print(f"   XGBoost explainer loaded ({len(models['explainer_features'])} features)")

        with open(CALIBRATION_PATH, "r") as f:
            calibration = json.load(f)
        models["min_lstm"]     = calibration["global_stats"]["score_min"]
        models["max_lstm"]     = calibration["global_stats"]["score_max"]
        models["golden_zones"] = calibration["golden_zones"]
        print(f"   Calibration loaded: {len(models['golden_zones'])} golden zones")

        print("\nSERVICE READY!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        raise

    yield
    print("\nVideo analysis service shutting down...")


app = FastAPI(
    title="Hirely Video Analysis Service",
    description="Analyzes candidate body language from interview video with SHAP explainability",
    lifespan=lifespan,
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


async def run_video_analysis(turn_id: int, video_path: str, interview_id: str, queued_at: datetime):
    start_time = time.time()
    try:
        logger.info(f"Processing video — Turn {turn_id}, file: {video_path}")

        result_data  = process_video_v2(video_path)
        scaled_score = result_data["scaled_score"]
        label        = get_confidence_label(scaled_score)
        elapsed_ms   = int((time.time() - start_time) * 1000)

        result = {
            "interviewTurnId":     turn_id,
            "confidenceLevel":     round(scaled_score, 4),
            "confidenceLabelText": label,
            "rawScore":            round(scaled_score, 4),
            "modelVersion":        "v2.0-lstm-shap",
            "status":              "completed",
            "processingTimeMs":    elapsed_ms,
            "processedAt":         datetime.now().isoformat(),
            "groupResults":        result_data["group_results"],
        }

        logger.info(f"Turn {turn_id} done — score={scaled_score:.4f}, label={label}, {elapsed_ms}ms")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Turn {turn_id} failed: {e}", exc_info=True)
        result = {
            "interviewTurnId":  turn_id,
            "status":           "failed",
            "errorMessage":     str(e),
            "processingTimeMs": elapsed_ms,
            "processedAt":      datetime.now().isoformat(),
        }

    if redis_client:
        try:
            redis_client.set(
                f"video_analysis:{interview_id}:{turn_id}",
                json.dumps(result),
                ex=86400,
            )
        except Exception as re:
            logger.error(f"Redis write failed: {re}")


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "service":       "Hirely Video Analysis v2.0",
        "models_loaded": bool(models.get("lstm_session")),
        "shap_ready":    bool(models.get("shap_explainer")),
        "timestamp":     datetime.now().isoformat(),
    }


class VideoAnalysisRequest(BaseModel):
    turn_id:      int
    interview_id: str
    video_path:   str


@app.post("/analyze-video")
async def analyze_video(request_data: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Queued video analysis — Turn {request_data.turn_id}")
        background_tasks.add_task(
            run_video_analysis,
            request_data.turn_id,
            request_data.video_path,
            request_data.interview_id,
            datetime.now(),
        )
        return {
            "status":    "queued",
            "turn_id":   request_data.turn_id,
            "message":   "Video analysis queued (v2 with SHAP).",
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
        result = redis_client.get(f"video_analysis:{interview_id}:{turn_id}")
        if result is None:
            return {"status": "pending"}
        return json.loads(result) if isinstance(result, str) else result
    except Exception as e:
        logger.error(f"Get result failed: {e}")
        return {"status": "error", "error": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Starting Hirely Video Analysis Service v2.0")
    print("Docs: http://localhost:8002/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)