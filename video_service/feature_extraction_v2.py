# ─────────────────────────────────────────────────────────────────────────────
# feature_extraction.py
# All extraction, normalization, and feature engineering for Hirely Video V3
# Import this in your pipeline scripts
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]     = "3"

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Run: pip install mediapipe")

try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("Run: pip install decord")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_FPS    = 5
WINDOW_SIZE   = 75      # frames = 15 seconds at 5fps
STEP_SIZE     = 25      # frames = 5 seconds step
TARGET_WIDTH  = 640
TARGET_HEIGHT = 480

FACE_LANDMARKER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "face_landmarker.task"
).replace("\\", "/")
# Landmark indices
_POSE_NOSE    = 0
_POSE_L_SHLDR = 11
_POSE_R_SHLDR = 12

_FACE_INDICES = [
    0, 4,       # Nose region
    13, 14,     # Lips
    61, 291,    # Mouth corners
    234, 454,   # Face edges
    10, 152     # Forehead/Chin
]
N_FACE = len(_FACE_INDICES)   # 10
N_HAND = 21

_ZERO3     = np.zeros(3,           dtype=np.float32)
_ZERO_FACE = np.zeros((N_FACE, 3), dtype=np.float32)
_ZERO_HAND = np.zeros((N_HAND, 3), dtype=np.float32)

# Feature engineering constants
N_FEATURES         = 13
N_FRAMES           = 75
SMOOTH_WINDOW      = 3
CLIP_RANGE         = 10.0
STILLNESS_WINDOW   = 10
VELOCITY_THRESHOLD = 0.05
MOUTH_THRESHOLD    = 0.005

# Face array indices
_F_NOSE_TIP    = 0
_F_UPPER_LIP   = 2
_F_LOWER_LIP   = 3
_F_L_FACE_EDGE = 6
_F_R_FACE_EDGE = 7
_F_FOREHEAD    = 8
_F_CHIN        = 9
_HAND_WRIST    = 0

FEATURE_NAMES = [
    "mouth_expressiveness",       # F0
    "head_nodding",               # F1
    "head_tilt",                  # F2
    "shoulder_alignment",         # F3
    "forward_lean",               # F4
    "left_hand_velocity",         # F5
    "right_hand_velocity",        # F6
    "left_hand_expressiveness",   # F7
    "right_hand_expressiveness",  # F8
    "body_stillness",             # F9
    "hand_gesture_range",         # F10
    "smile_intensity",            # F11
    "gaze_consistency",           # F12
]


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkExtractor:
    def __init__(self):
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self._holistic.close()

    def extract(self, frames):
        raw = [self._process_frame(f) for f in frames]
        return self._impute_missing(raw)

    def _process_frame(self, frame):
        result = self._holistic.process(frame)
        out    = {
            "nose": None, "left_shoulder": None, "right_shoulder": None,
            "face": None, "left_hand": None, "right_hand": None, "valid": False
        }

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            if len(lms) > max(_POSE_NOSE, _POSE_L_SHLDR, _POSE_R_SHLDR):
                out["nose"]           = np.array([lms[_POSE_NOSE].x,    lms[_POSE_NOSE].y,    lms[_POSE_NOSE].z],    dtype=np.float32)
                out["left_shoulder"]  = np.array([lms[_POSE_L_SHLDR].x, lms[_POSE_L_SHLDR].y, lms[_POSE_L_SHLDR].z], dtype=np.float32)
                out["right_shoulder"] = np.array([lms[_POSE_R_SHLDR].x, lms[_POSE_R_SHLDR].y, lms[_POSE_R_SHLDR].z], dtype=np.float32)
                out["valid"] = True

        if result.face_landmarks:
            flms = result.face_landmarks.landmark
            if len(flms) > max(_FACE_INDICES):
                out["face"] = np.stack([
                    np.array([flms[fi].x, flms[fi].y, flms[fi].z], dtype=np.float32)
                    for fi in _FACE_INDICES
                ])

        if result.left_hand_landmarks and len(result.left_hand_landmarks.landmark) >= N_HAND:
            out["left_hand"] = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark[:N_HAND]],
                dtype=np.float32
            )

        if result.right_hand_landmarks and len(result.right_hand_landmarks.landmark) >= N_HAND:
            out["right_hand"] = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark[:N_HAND]],
                dtype=np.float32
            )

        return out

    def _impute_missing(self, raw):
        for key in ("nose", "left_shoulder", "right_shoulder"):
            self._fill(raw, key, lambda: _ZERO3.copy())
        self._fill(raw, "face", lambda: _ZERO_FACE.copy())
        for key in ("left_hand", "right_hand"):
            for i in range(len(raw)):
                if raw[i][key] is None:
                    raw[i][key] = _ZERO_HAND.copy()
        return raw

    def _fill(self, dicts, key, zero_fn):
        n    = len(dicts)
        last = None
        for i in range(n):
            if dicts[i][key] is not None: last = dicts[i][key]
            elif last is not None:        dicts[i][key] = last.copy()
        last = None
        for i in range(n-1, -1, -1):
            if dicts[i][key] is not None: last = dicts[i][key]
            elif last is not None:        dicts[i][key] = last.copy()
        for i in range(n):
            if dicts[i][key] is None:
                dicts[i][key] = zero_fn()


# ═══════════════════════════════════════════════════════════════════════════════
# SMILE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════
print(f"FACE_LANDMARKER_PATH = {FACE_LANDMARKER_PATH}")

class SmileExtractor:

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = FACE_LANDMARKER_PATH
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), model_path
            )
        model_path = str(model_path).replace("\\", "/")

        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions           = mp.tasks.BaseOptions
        RunningMode           = mp.tasks.vision.RunningMode

        # Load model bytes directly — bypasses MediaPipe Windows path bug
        with open(model_path, "rb") as f:
            model_data = f.read()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            output_face_blendshapes=True,
            running_mode=RunningMode.IMAGE,
        )
        self._detector = FaceLandmarker.create_from_options(options)

    def close(self):
        self._detector.close()

    def extract_smile_scores(self, frames_rgb):
        scores = np.zeros(len(frames_rgb), dtype=np.float32)
        for i, frame_rgb in enumerate(frames_rgb):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result   = self._detector.detect(mp_image)
            if not result.face_blendshapes:
                continue
            blendshapes = result.face_blendshapes[0]
            try:
                left_smile  = next(b.score for b in blendshapes if b.category_name == "mouthSmileLeft")
                right_smile = next(b.score for b in blendshapes if b.category_name == "mouthSmileRight")
                scores[i]   = float((left_smile + right_smile) / 2.0)
            except StopIteration:
                scores[i] = 0.0
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all_frames(video_path):
    """Extract ALL frames at TARGET_FPS — no cap on total count."""
    vr         = VideoReader(str(video_path), ctx=cpu(0), width=TARGET_WIDTH, height=TARGET_HEIGHT)
    native_fps = vr.get_avg_fps()
    total      = len(vr)
    step       = native_fps / TARGET_FPS
    indices    = []
    i          = 0.0
    while round(i) < total:
        indices.append(min(round(i), total - 1))
        i += step
    return vr.get_batch(indices).asnumpy()   # RGB uint8


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(landmark_dicts):
    dicts      = deepcopy(landmark_dicts)
    raw_scales = [float(np.linalg.norm(d["left_shoulder"] - d["right_shoulder"])) for d in dicts]
    raw_scales = [s if s >= 1e-6 else 0.0 for s in raw_scales]

    scales = list(raw_scales)
    last   = None
    for i in range(len(scales)):
        if scales[i] > 0: last = scales[i]
        elif last:         scales[i] = last
    last = None
    for i in range(len(scales)-1, -1, -1):
        if scales[i] > 0: last = scales[i]
        elif last:         scales[i] = last
    final_scales = np.array([s if s > 0 else 1.0 for s in scales])

    for d, scale in zip(dicts, final_scales):
        origin = d["nose"].copy()
        inv    = 1.0 / scale
        for k in ("nose", "left_shoulder", "right_shoulder"):
            d[k] = ((d[k] - origin) * inv).astype(np.float32)
        for k in ("face", "left_hand", "right_hand"):
            d[k] = ((d[k] - origin[np.newaxis, :]) * inv).astype(np.float32)
        d["nose"] = np.zeros(3, dtype=np.float32)
    return dicts, final_scales


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_angle(a):
    if a >  90: return a - 180
    if a < -90: return a + 180
    return a



def engineer_features(norm_dicts, scales, smile_scores):
    raw          = np.zeros((N_FRAMES, N_FEATURES), dtype=np.float64)
    yaw_sequence = np.zeros(N_FRAMES,               dtype=np.float64)

    prev_l_wrist     = None
    prev_r_wrist     = None
    prev_l_vel       = 0.0
    prev_r_vel       = 0.0
    all_wrist_pts    = []
    midpoint_history = []

    for i, d in enumerate(norm_dicts):
        face, ls, rs = d["face"], d["left_shoulder"], d["right_shoulder"]

        # F0: Mouth Expressiveness
        m_dist    = float(np.linalg.norm(face[_F_UPPER_LIP, :2] - face[_F_LOWER_LIP, :2]))
        raw[i, 0] = m_dist if m_dist > MOUTH_THRESHOLD else 0.0

        # F2: Head Tilt & F3: Shoulder Alignment
        f_dy       = float(face[_F_L_FACE_EDGE, 1] - face[_F_R_FACE_EDGE, 1])
        f_dx       = float(face[_F_L_FACE_EDGE, 0] - face[_F_R_FACE_EDGE, 0]) + 1e-9
        face_angle = wrap_angle(np.degrees(np.arctan2(f_dy, f_dx)))
        s_dy       = float(ls[1] - rs[1])
        s_dx       = float(ls[0] - rs[0]) + 1e-9
        sh_angle   = wrap_angle(np.degrees(np.arctan2(s_dy, s_dx)))
        rel_tilt   = face_angle - sh_angle
        raw[i, 2]  = float(rel_tilt) if abs(rel_tilt) < abs(face_angle) else float(face_angle)
        raw[i, 3]  = float(ls[1] - rs[1])

        # --- IMPROVED HAND LOGIC ---
        curr_l = d["left_hand"][_HAND_WRIST, :2]
        curr_r = d["right_hand"][_HAND_WRIST, :2]
        
        # Kill noise near origin [0,0]
        l_active = bool(np.any(d["left_hand"] != 0.0)) and not np.allclose(curr_l, [0, 0], atol=1e-3)
        r_active = bool(np.any(d["right_hand"] != 0.0)) and not np.allclose(curr_r, [0, 0], atol=1e-3)

        # DEADZONE (Outer 5%)
        l_safe = (0.02 < curr_l[0] < 0.98) and (0.02 < curr_l[1] < 0.98)
        r_safe = (0.02 < curr_r[0] < 0.98) and (0.02 < curr_r[1] < 0.98)

        if l_active and l_safe: all_wrist_pts.append(curr_l)
        if r_active and r_safe: all_wrist_pts.append(curr_r)

        # ONLY append to Range list if in the safe zone
        if l_active and l_safe: all_wrist_pts.append(curr_l)
        if r_active and r_safe: all_wrist_pts.append(curr_r)

        # VELOCITY
        l_vel = 0.0
        if l_active and prev_l_wrist is not None:
            l_dist = min(float(np.linalg.norm(curr_l - prev_l_wrist)), 0.35)
            l_vel_raw = (l_dist * 12.0) if (l_dist * 12.0) > VELOCITY_THRESHOLD else 0.0
            l_vel = (l_vel_raw * 0.8) + (prev_l_vel * 0.2)
        raw[i, 5] = l_vel

        r_vel = 0.0
        if r_active and prev_r_wrist is not None:
            r_dist = min(float(np.linalg.norm(curr_r - prev_r_wrist)), 0.35)
            r_vel_raw = (r_dist * 12.0) if (r_dist * 12.0) > VELOCITY_THRESHOLD else 0.0
            r_vel = (r_vel_raw * 0.8) + (prev_r_vel * 0.2)
        raw[i, 6] = r_vel

        # EXPRESSIVENESS
        if l_active:
            raw[i, 7] = l_vel + abs(l_vel - prev_l_vel)
        else:
            raw[i, 7] = 0.0

        if r_active:
            raw[i, 8] = r_vel + abs(r_vel - prev_r_vel)
        else:
            raw[i, 8] = 0.0

        # F9: Body Stillness
        mx, my = float((ls[0] + rs[0]) / 2.0), float((ls[1] + rs[1]) / 2.0)
        midpoint_history.append((mx, my))
        if len(midpoint_history) > STILLNESS_WINDOW: midpoint_history.pop(0)
        if len(midpoint_history) > 1:
            raw[i, 9] = (float(np.var([p[0] for p in midpoint_history])) + 
                         float(np.var([p[1] for p in midpoint_history]))) * 1000.0

        raw[i, 11] = float(smile_scores[i])

        # Yaw Sequence
        fw = float(face[_F_R_FACE_EDGE, 0] - face[_F_L_FACE_EDGE, 0])
        yaw_sequence[i] = ((float(face[_F_NOSE_TIP, 0]) - float(face[_F_L_FACE_EDGE, 0])) / fw 
                          if abs(fw) > 1e-6 else 0.5)

        # Roll state
        prev_l_wrist, prev_r_wrist = (curr_l if l_active else None), (curr_r if r_active else None)
        prev_l_vel, prev_r_vel = (l_vel if l_active else 0.0), (r_vel if r_active else 0.0)

    # --- POST-LOOP ---
    nose_y   = np.array([d["nose"][1] for d in norm_dicts])
    sh_mid_y = np.array([(d["left_shoulder"][1] + d["right_shoulder"][1]) / 2.0 for d in norm_dicts])
    raw[:, 1] = np.abs(np.gradient(sh_mid_y - nose_y))
    raw[:, 4] = np.abs(np.gradient(scales)) * 8.0

    # F10: CLEAN RANGE
# F10: Hand Gesture Range (Tuned for Reactivity)
    # 1. Lowered requirement to 5 frames (1 second of movement)
    if len(all_wrist_pts) > 5:
        pts = np.array(all_wrist_pts)
        
        # 2. Reverted to Max/Min but with the Safe Zone protection
        # We don't need percentiles if our "Safe Zone" is working correctly.
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        
        # 3. Adjusted Multiplier to 8.0
        # This makes it easier to see movement without instantly hitting 10.
        dist_raw = float(np.sqrt(x_range**2 + y_range**2))
        raw[:, 10] = dist_raw * 8.0 
    else:
        raw[:, 10] = 0.0

    for i in range(N_FRAMES):
        raw[i, 12] = float(np.var(yaw_sequence[max(0, i-10):min(N_FRAMES, i+11)])) * 1000.0

    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    for f in {1, 2, 3, 4}: raw[:, f] = np.convolve(raw[:, f], kernel, mode="same")

    return np.clip(np.nan_to_num(raw), -CLIP_RANGE, CLIP_RANGE).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_features(arr):
    """Convert (75, 13) temporal array into flat feature dict for XGBoost."""
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        stats[f"{name}_mean"] = float(np.mean(col))
        stats[f"{name}_std"]  = float(np.std(col))
        stats[f"{name}_min"]  = float(np.min(col))
        stats[f"{name}_max"]  = float(np.max(col))
    return stats