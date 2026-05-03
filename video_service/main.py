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

LSTM_MODEL_PATH = str(_MODEL_DIR / "video_lstm_v2.onnx")
EXPLAINER_MODEL_PATH = str(_MODEL_DIR / "video_explainer_model_v2.pkl")
EXPLAINER_FEATURES_PATH = str(_MODEL_DIR / "video_explainer_features_v2.pkl")
SCALER_PATH = str(_MODEL_DIR / "video_scaler_v2.pkl")
CALIBRATION_PATH = str(_MODEL_DIR / "calibration_data_v8.json")

# ── Friendly names for coaching tips ─────────────────────────────────────────
FRIENDLY_NAMES = {
    "mouth_expressiveness": "Mouth Expressiveness",
    "head_nodding": "Head Nodding",
    "head_tilt": "Head Tilt",
    "shoulder_alignment": "Shoulder Alignment",
    "forward_lean": "Forward Lean",
    "left_hand_velocity": "Left Hand Gestures",
    "right_hand_velocity": "Right Hand Gestures",
    "left_hand_expressiveness": "Left Hand Expressiveness",
    "right_hand_expressiveness": "Right Hand Expressiveness",
    "body_stillness": "Body Stillness",
    "hand_gesture_range": "Hand Gesture Range",
    "smile_intensity": "Smile Intensity",
    "gaze_consistency": "Conversational Orientation",
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

ENGINEERED_SUFFIXES = ("_cv", "_peak_ratio", "_skew_approx", "_stability")

FEEDBACK_GROUPS = {
    "Facial Engagement": {
        "members": ["mouth_expressiveness", "head_nodding", "smile_intensity", "gaze_consistency"],
        "yellow_threshold": -0.0001,
        "red_threshold": -0.1600,
    },
    "Hand Gestures": {
        "members": ["left_hand_velocity", "right_hand_velocity",
                     "left_hand_expressiveness", "right_hand_expressiveness",
                     "hand_gesture_range"],
        "yellow_threshold": -0.0001,
        "red_threshold": -0.1200,
    },
    "Posture & Presence": {
        "members": ["head_tilt", "shoulder_alignment", "forward_lean", "body_stillness"],
        "yellow_threshold": -0.0200,
        "red_threshold": -0.0900,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE RESCALING
# ═══════════════════════════════════════════════════════════════════════════════

def get_confidence_label(score: float) -> str:
    if score >= 0.75:
        return "Highly Confident"
    if score >= 0.50:
        return "Confident"
    if score >= 0.30:
        return "Moderately Confident"
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
        std_v = stats.get(f"{base}_std")
        max_v = stats.get(f"{base}_max")
        if None in (mean_v, std_v, max_v):
            continue
        enriched[f"{base}_cv"] = std_v / (abs(mean_v) + 1e-6)
        enriched[f"{base}_peak_ratio"] = max_v / (abs(mean_v) + 1e-6)
        enriched[f"{base}_skew_approx"] = (max_v - mean_v) / (std_v + 1e-6)
        enriched[f"{base}_stability"] = 1.0 / (std_v + 1e-6)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# COACHING FEEDBACK LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _get_engineered_tip(culprit_feat, base_feature, val, zone, direction):
    friendly = FRIENDLY_NAMES.get(base_feature, base_feature)
    zone_min = zone.get("min", -99)
    zone_max = zone.get("max", 99)
    in_zone = zone_min <= val <= zone_max

    if culprit_feat.endswith("_stability"):
        if direction == "positive":
            if in_zone or val > zone_max:
                return f"Your {friendly} is impressively consistent — a real strength.", "green"
            return f"Your {friendly} is growing in consistency. Keep it steady.", "yellow"
        else:
            if val < zone_min:
                return (f"Your {friendly} is too erratic between moments. "
                        f"Focus on smoother, more controlled movements."), "red"
            if val > zone_max:
                return (f"Your {friendly} is almost rigid. "
                        f"Allow a little natural variation to feel human."), "yellow"
        return f"Your {friendly} consistency could be slightly improved.", "yellow"

    if culprit_feat.endswith("_cv"):
        if direction == "positive":
            if in_zone or val > zone_max:
                return f"The rhythm of your {friendly} is well-balanced.", "green"
            return f"Your {friendly} variability is improving — keep building rhythm.", "yellow"
        else:
            if val > zone_max:
                return (f"Your {friendly} is highly inconsistent — "
                        f"the variation is distracting. Aim for smoother delivery."), "red"
            if val < zone_min:
                return (f"Your {friendly} is too uniform. "
                        f"Let it vary a little more to feel natural."), "yellow"
        return f"The rhythm of your {friendly} could feel more natural.", "yellow"

    if culprit_feat.endswith("_peak_ratio"):
        if direction == "positive":
            if in_zone or val > zone_max:
                return f"Your {friendly} peaks are well-proportioned to your baseline.", "green"
            return f"Good energy spikes in your {friendly} — keep calibrating them.", "yellow"
        else:
            if val > zone_max:
                return (f"Your {friendly} spikes dramatically at times but stays low otherwise. "
                        f"Try to maintain a higher baseline instead of occasional bursts."), "red"
            if val < zone_min:
                return (f"Your {friendly} lacks peak moments — "
                        f"add occasional emphasis to keep the audience engaged."), "red"
        return f"Your {friendly} peak moments could be better calibrated.", "yellow"

    if culprit_feat.endswith("_skew_approx"):
        if direction == "positive":
            if in_zone or val > zone_max:
                return f"The distribution of your {friendly} is well-balanced.", "green"
            return f"Your {friendly} is becoming more balanced — keep going.", "yellow"
        else:
            if val > zone_max:
                return (f"Your {friendly} is mostly flat with occasional bursts — "
                        f"try to sustain a higher baseline level."), "red"
            if val < zone_min:
                return (f"Your {friendly} stays near its peak constantly — "
                        f"vary it more so emphasis moments stand out."), "yellow"
        return f"The balance of your {friendly} could be improved.", "yellow"

    return f"Your {friendly} could be adjusted for better impact.", "yellow"


def get_granular_feedback(culprit_feat, base_feature, val, zone, direction="negative"):
    friendly = FRIENDLY_NAMES.get(base_feature, base_feature)
    is_hand = "hand" in base_feature

    if any(culprit_feat.endswith(s) for s in ENGINEERED_SUFFIXES):
        return _get_engineered_tip(culprit_feat, base_feature, val, zone, direction)

    if is_hand and zone and val <= (zone.get("min", 0) + 0.10):
        if direction == "positive":
            return (f"Your {friendly} weren't visible in frame. "
                    f"This kept your posture stable, but consider showing them to add impact."), "yellow"
        return (f"Your {friendly} weren't detected. "
                f"Bring your hands into the frame to show engagement."), "red"

    if direction == "positive":
        if val < zone.get("min", -99):
            return f"You're on the right track with {friendly}! Keep it up to reach the elite standard.", "yellow"
        return f"Your {friendly} is a key strength — it's well-calibrated and professional.", "green"

    feat_dir = zone.get("direction", "INCREASING") if zone else "INCREASING"

    if "_std" in culprit_feat:
        if val < zone.get("min", -99):
            return f"Your {friendly} feels robotic. Let it move more naturally.", "red"
        if val > zone.get("max", 99):
            return f"Your {friendly} is too erratic. Focus on keeping movements smooth.", "red"

    if val < zone.get("min", -99):
        if feat_dir == "INCREASING":
            return f"Your overall {friendly} level is low. Show more of it.", "red"
        elif feat_dir == "DECREASING":
            return f"Your {friendly} is tilting too far to one side. Try to keep it more centered.", "red"
        else:
            return f"Your {friendly} is too low — try to find the middle range.", "red"

    if val > zone.get("max", 99):
        if feat_dir == "DECREASING":
            return f"Your {friendly} is excessive. Dial it back.", "red"
        elif feat_dir == "INVERTED_U":
            return f"Your {friendly} is past the sweet spot — dial it back slightly.", "red"
        return f"Your {friendly} is a bit too intense. Bring it down slightly for a more natural feel.", "red"

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
        val = scaled_stats_dict.get(feat, 0)
        zone = models["golden_zones"].get(feat, {"min": -99, "max": 99})
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

        tips = []
        seen_bases: set = set()

        if status == "green":
            p_limit, n_limit = 4, 1
        elif status == "yellow":
            p_limit, n_limit = 3, 3
        else:
            p_limit, n_limit = 1, 4

        def process_tips(features, dir_type, limit):
            count = 0
            for feat, shap_val in features:
                if count >= limit:
                    break
                base = next((m for m in members if feat.startswith(m)), feat)
                if base in seen_bases:
                    continue

                val = scaled_stats_dict.get(feat, 0)
                zone = models["golden_zones"].get(feat, {})

                tip_text, tip_status = get_granular_feedback(
                    feat, base, val, zone, direction=dir_type)

                tips.append({
                    "feature": feat,
                    "base": base,
                    "friendly": FRIENDLY_NAMES.get(base, base),
                    "shap": round(float(shap_val), 4),
                    "direction": dir_type,
                    "tip": tip_text,
                    "status": tip_status,
                    "val": round(float(val), 4),
                    "zone_min": round(float(zone.get("min", -99)), 4),
                    "zone_max": round(float(zone.get("max", 99)), 4),
                    "zone_direction": zone.get("direction", "INCREASING"),
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
            "status": status,
            "tips": tips,
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

    extractor = LandmarkExtractor()
    all_landmarks = extractor.extract(list(all_frames))
    extractor.close()

    smile_ext = SmileExtractor(model_path=FACE_LANDMARKER_PATH)
    all_smiles = smile_ext.extract_smile_scores(all_frames)
    smile_ext.close()

    lstm_session = models["lstm_session"]
    scaler = models["scaler"]
    explainer_features = models["explainer_features"]
    shap_explainer = models["shap_explainer"]

    master_features, window_scores = [], []
    starts = list(range(0, len(all_frames) - WINDOW_SIZE + 1, STEP_SIZE)) or [0]

    for start in starts:
        end = start + WINDOW_SIZE
        norm_dicts, scales = normalize(all_landmarks[start:end])
        feat_array = scaler.transform(
            engineer_features(norm_dicts, scales, all_smiles[start:end])
        )
        input_tensor = feat_array[np.newaxis, :, :].astype(np.float32)
        raw = float(lstm_session.run(None, {"video_features": input_tensor})[0][0])

        min_lstm = models["min_lstm"]
        max_lstm = models["max_lstm"]
        window_scores.append(np.clip((raw - min_lstm) / (max_lstm - min_lstm), 0.0, 1.0))
        master_features.append(feat_array)

    raw_stats = aggregate_features(np.vstack(master_features))
    enriched_stats = enrich_aggregated_stats(raw_stats)

    X_single = pd.DataFrame([enriched_stats])[explainer_features]
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
        "scaled_score": overall_score,
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
        print(f"   ONNX LSTM v2 loaded: {LSTM_MODEL_PATH}")

        expl_model = joblib.load(EXPLAINER_MODEL_PATH)
        models["explainer_features"] = joblib.load(EXPLAINER_FEATURES_PATH)
        models["scaler"] = joblib.load(SCALER_PATH)
        models["shap_explainer"] = shap.TreeExplainer(expl_model)
        print(f"   XGBoost explainer loaded ({len(models['explainer_features'])} features)")

        with open(CALIBRATION_PATH, "r") as f:
            calibration = json.load(f)
        models["min_lstm"] = calibration["global_stats"]["score_min"]
        models["max_lstm"] = calibration["global_stats"]["score_max"]
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


async def run_video_analysis(
    turn_id: int,
    video_path: str,
    interview_id: str,
    queued_at: datetime,
):
    start_time = time.time()
    try:
        logger.info(f"Processing video — Turn {turn_id}, file: {video_path}")

        result_data = process_video_v2(video_path)
        scaled_score = result_data["scaled_score"]
        label = get_confidence_label(scaled_score)
        elapsed_ms = int((time.time() - start_time) * 1000)

        result = {
            "interviewTurnId": turn_id,
            "confidenceLevel": round(scaled_score, 4),
            "confidenceLabelText": label,
            "rawScore": round(scaled_score, 4),
            "modelVersion": "v2.0-lstm-shap",
            "status": "completed",
            "processingTimeMs": elapsed_ms,
            "processedAt": datetime.now().isoformat(),
            "groupResults": result_data["group_results"],
        }

        logger.info(
            f"Turn {turn_id} done — score={scaled_score:.4f}, "
            f"label={label}, {elapsed_ms}ms"
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Turn {turn_id} failed: {e}", exc_info=True)
        result = {
            "interviewTurnId": turn_id,
            "status": "failed",
            "errorMessage": str(e),
            "processingTimeMs": elapsed_ms,
            "processedAt": datetime.now().isoformat(),
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
        "status": "ok",
        "service": "Hirely Video Analysis v2.0",
        "models_loaded": bool(models.get("lstm_session")),
        "shap_ready": bool(models.get("shap_explainer")),
        "timestamp": datetime.now().isoformat(),
    }


class VideoAnalysisRequest(BaseModel):
    turn_id: int
    interview_id: str
    video_path: str


@app.post("/analyze-video")
async def analyze_video(
    request_data: VideoAnalysisRequest, background_tasks: BackgroundTasks
):
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
            "status": "queued",
            "turn_id": request_data.turn_id,
            "message": "Video analysis queued (v2 with SHAP).",
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
