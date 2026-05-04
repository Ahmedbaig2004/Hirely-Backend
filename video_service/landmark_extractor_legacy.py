# ─────────────────────────────────────────────────────────────────────────────
# landmark_extractor_legacy.py
# Uses mp.solutions.holistic (legacy API) instead of Tasks API
# For inference only — avoids the "packet is empty" crash on Windows
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp

log = logging.getLogger(__name__)

_POSE_NOSE    = 0
_POSE_L_SHLDR = 11
_POSE_R_SHLDR = 12

_FACE_INDICES = [0, 4, 13, 14, 33, 133, 362, 263, 61, 291]
N_FACE = len(_FACE_INDICES)
N_HAND = 21

_ZERO3     = np.zeros(3,           dtype=np.float32)
_ZERO_FACE = np.zeros((N_FACE, 3), dtype=np.float32)
_ZERO_HAND = np.zeros((N_HAND, 3), dtype=np.float32)


class LandmarkExtractor:
    def __init__(self, **kwargs):
        # kwargs accepted for compatibility but ignored
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def extract(self, frames: List[np.ndarray]) -> List[Dict]:
        raw = []
        for i, frame in enumerate(frames):
            d = self._process_frame(frame, i)
            raw.append(d)
        return _impute_missing(raw)

    def _process_frame(self, frame: np.ndarray, idx: int) -> Dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        out = {
            "nose":           None,
            "left_shoulder":  None,
            "right_shoulder": None,
            "face":           None,
            "left_hand":      None,
            "right_hand":     None,
            "valid":          False,
        }

        # Pose
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            if len(lms) > max(_POSE_NOSE, _POSE_L_SHLDR, _POSE_R_SHLDR):
                out["nose"]           = _lm3(lms[_POSE_NOSE])
                out["left_shoulder"]  = _lm3(lms[_POSE_L_SHLDR])
                out["right_shoulder"] = _lm3(lms[_POSE_R_SHLDR])
                out["valid"] = True

        # Face
        if result.face_landmarks:
            flms = result.face_landmarks.landmark
            if len(flms) > max(_FACE_INDICES):
                face_pts = [_lm3(flms[fi]) for fi in _FACE_INDICES]
                out["face"] = np.stack(face_pts, axis=0).astype(np.float32)

        # Hands
        if result.left_hand_landmarks:
            lh = result.left_hand_landmarks.landmark
            if len(lh) >= N_HAND:
                out["left_hand"] = np.array(
                    [_lm3(lh[j]) for j in range(N_HAND)], dtype=np.float32
                )

        if result.right_hand_landmarks:
            rh = result.right_hand_landmarks.landmark
            if len(rh) >= N_HAND:
                out["right_hand"] = np.array(
                    [_lm3(rh[j]) for j in range(N_HAND)], dtype=np.float32
                )

        return out


def _lm3(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _impute_missing(raw: List[Dict]) -> List[Dict]:
    for key in ("nose", "left_shoulder", "right_shoulder"):
        _fill_field(raw, key, lambda: _ZERO3.copy())
    _fill_field(raw, "face", lambda: _ZERO_FACE.copy())
    for key in ("left_hand", "right_hand"):
        for i in range(len(raw)):
            if raw[i][key] is None:
                raw[i][key] = _ZERO_HAND.copy()
    return raw


def _fill_field(dicts, key, zero_factory):
    n = len(dicts)
    last = None
    for i in range(n):
        if dicts[i][key] is not None:
            last = dicts[i][key]
        elif last is not None:
            dicts[i][key] = last.copy()
    last = None
    for i in range(n - 1, -1, -1):
        if dicts[i][key] is not None:
            last = dicts[i][key]
        elif last is not None:
            dicts[i][key] = last.copy()
    for i in range(n):
        if dicts[i][key] is None:
            dicts[i][key] = zero_factory()