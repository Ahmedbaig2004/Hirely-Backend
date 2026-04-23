from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import json
import os
import time
from datetime import datetime
from upstash_redis import Redis
import logging
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = {}

OBSERVED_MIN = 0.2396
OBSERVED_MAX = 0.7526


def rescale_score(raw_score: float) -> float:
    rescaled = (raw_score - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)
    return float(np.clip(rescaled, 0.0, 1.0))


def get_confidence_label(score: float) -> str:
    if score >= 0.75:
        return "Highly Confident"
    if score >= 0.50:
        return "Confident"
    if score >= 0.30:
        return "Moderately Confident"
    return "Needs Improvement"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("VIDEO ANALYSIS SERVICE STARTING...")
    print("=" * 60)
    try:
        import onnxruntime as ort
        model_path = os.path.join(os.path.dirname(__file__), "models", "video_lstm.onnx")
        models["session"] = ort.InferenceSession(model_path)
        print(f"   ONNX model loaded: {model_path}")

        from landmark_extractor_legacy import LandmarkExtractor
        models["extractor"] = LandmarkExtractor()
        print("   MediaPipe holistic initialized")

        print("\nSERVICE READY!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        raise

    yield

    if models.get("extractor"):
        models["extractor"].close()
    print("\nVideo analysis service shutting down...")


app = FastAPI(
    title="Hirely Video Analysis Service",
    description="Analyzes candidate body language from interview video",
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


def score_video(video_path: str) -> float:
    import cv2
    from frame_extractor import extract_frames, TARGET_FPS, TARGET_FRAMES
    from normalizer import normalize
    from feature_engineering import engineer_features

    frames = extract_frames(video_path, TARGET_FPS, TARGET_FRAMES)

    h, w = frames[0].shape[:2]
    frames = [
        f if (f.shape[0] == h and f.shape[1] == w) else cv2.resize(f, (w, h))
        for f in frames
    ]

    landmarks = models["extractor"].extract(frames)

    valid_count = sum(1 for lm in landmarks if lm.get("valid", False))
    if valid_count == 0:
        raise ValueError(
            "No body landmarks detected in any frame — "
            "ensure face and upper body are visible"
        )

    norm_landmarks = normalize(landmarks)
    features = engineer_features(norm_landmarks)

    input_array = features[np.newaxis, :, :]
    result = models["session"].run(None, {"video_features": input_array})
    return float(result[0][0])


async def process_video_analysis(
    turn_id: int,
    video_path: str,
    interview_id: str,
    queued_at: datetime,
):
    start_time = time.time()
    try:
        logger.info(f"Processing video — Turn {turn_id}, file: {video_path}")

        raw_score = score_video(video_path)
        rescaled = rescale_score(raw_score)
        label = get_confidence_label(rescaled)
        elapsed_ms = int((time.time() - start_time) * 1000)

        result = {
            "interviewTurnId": turn_id,
            "confidenceLevel": round(rescaled, 4),
            "confidenceLabelText": label,
            "rawScore": round(raw_score, 4),
            "modelVersion": "v1.0-lstm-video",
            "status": "completed",
            "processingTimeMs": elapsed_ms,
            "processedAt": datetime.now().isoformat(),
        }

        logger.info(
            f"Turn {turn_id} done — raw={raw_score:.4f}, "
            f"rescaled={rescaled:.4f}, label={label}, {elapsed_ms}ms"
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Turn {turn_id} failed: {e}")
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
        "service": "Hirely Video Analysis v1.0",
        "models_loaded": bool(models.get("session")),
        "mediapipe_ready": bool(models.get("extractor")),
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
            process_video_analysis,
            request_data.turn_id,
            request_data.video_path,
            request_data.interview_id,
            datetime.now(),
        )
        return {
            "status": "queued",
            "turn_id": request_data.turn_id,
            "message": "Video analysis queued.",
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
    print("Starting Hirely Video Analysis Service v1.0")
    print("Docs: http://localhost:8002/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)
