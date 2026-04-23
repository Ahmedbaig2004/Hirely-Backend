"""
feature_engineering.py
======================
Stage 4 — Compute exactly 20 behavioural features per frame.

Compatible: Python 3.10.11, numpy 1.24+

Output: np.ndarray  shape (75, 20)  dtype float32  no NaN/Inf

Feature index reference (FIXED ORDER — never reorder):
    ── HEAD / FACE (6) ──────────────────────────────────────────────────
     0  mouth_openness          vertical lip distance (upper/lower lip)
     1  eye_aspect_ratio_left   horizontal span of left eye
     2  eye_aspect_ratio_right  horizontal span of right eye
     3  head_tilt_angle         shoulder line angle from horizontal (deg)
     4  forward_head_posture    nose x-offset from shoulder midpoint
     5  head_nod_angle          nose y-offset from shoulder midpoint

    ── SHOULDERS (3) ────────────────────────────────────────────────────
     6  shoulder_tilt           Y-diff left minus right shoulder
     7  shoulder_symmetry       |left_y - right_y|  (lower = symmetric)
     8  shoulder_width          L↔R distance (≈1 at calibration baseline)

    ── HANDS — POSITION (5) ─────────────────────────────────────────────
     9  left_hand_face_dist     Euclidean: left wrist → nose (origin)
    10  right_hand_face_dist    Euclidean: right wrist → nose
    11  left_hand_shoulder_dist Euclidean: left wrist → left shoulder
    12  right_hand_shoulder_dist
    13  hand_symmetry           |left_hand_face_dist - right_hand_face_dist|

    ── HANDS — MOTION (4) ───────────────────────────────────────────────
    14  left_hand_velocity      frame-to-frame wrist displacement magnitude
    15  right_hand_velocity
    16  left_hand_acceleration  1st derivative of velocity
    17  right_hand_acceleration

    ── BODY DYNAMICS (2) ────────────────────────────────────────────────
    18  posture_stability       rolling variance of shoulder midpoint
    19  gaze_proxy              face nose-tip z-coordinate (depth cue)

Processing:
    - All features computed on normalized (scale-invariant) coordinates
    - Velocity and acceleration via np.gradient (central finite differences)
    - Smoothed with rolling mean (window=3)
    - Clipped to [-10, 10] to suppress extreme outliers
    - Final NaN/Inf replaced with 0.0
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

log = logging.getLogger(__name__)

N_FEATURES: int    = 20
N_FRAMES: int      = 75
SMOOTH_WINDOW: int = 3
CLIP_RANGE: float  = 10.0

# Face landmark local indices (within the 10-point subset)
# Matches _FACE_INDICES = [0,4,13,14,33,133,362,263,61,291]
_F_NOSE_TIP   = 0   # original index 0  — nose tip
_F_NOSE_BOT   = 1   # original index 4  — nose bottom
_F_UPPER_LIP  = 2   # original index 13 — upper lip
_F_LOWER_LIP  = 3   # original index 14 — lower lip
_F_LE_OUTER   = 4   # original index 33 — left eye outer
_F_LE_INNER   = 5   # original index 133 — left eye inner
_F_RE_INNER   = 6   # original index 362 — right eye inner
_F_RE_OUTER   = 7   # original index 263 — right eye outer
_F_L_MOUTH    = 8   # original index 61  — left mouth corner
_F_R_MOUTH    = 9   # original index 291 — right mouth corner

_HAND_WRIST   = 0   # wrist = landmark 0 in MediaPipe hand model


def engineer_features(norm_dicts: List[Dict]) -> np.ndarray:
    """
    Compute exactly 20 features for each of 75 normalized frames.

    Parameters
    ----------
    norm_dicts : 75 normalized landmark dicts (output of normalizer.normalize)

    Returns
    -------
    np.ndarray  shape (75, 20)  dtype float32  no NaN/Inf
    """
    assert len(norm_dicts) == N_FRAMES, \
        f"Expected {N_FRAMES} dicts, got {len(norm_dicts)}"

    raw = np.zeros((N_FRAMES, N_FEATURES), dtype=np.float64)

    for i, d in enumerate(norm_dicts):
        raw[i] = _static_features(d)

    raw = _temporal_features(raw, norm_dicts)

    raw[:, 18] = _posture_stability(norm_dicts)

    smoothed = _smooth(raw, SMOOTH_WINDOW)
    smoothed = np.clip(smoothed, -CLIP_RANGE, CLIP_RANGE)
    smoothed = np.nan_to_num(
        smoothed, nan=0.0, posinf=CLIP_RANGE, neginf=-CLIP_RANGE
    )

    result = smoothed.astype(np.float32)
    assert result.shape == (N_FRAMES, N_FEATURES), \
        f"BUG: output shape {result.shape}"
    return result


# ── Per-frame static features ─────────────────────────────────────────────────
def _static_features(d: Dict) -> np.ndarray:
    feats = np.zeros(N_FEATURES, dtype=np.float64)

    face = d["face"]           # (10, 3) float32
    ls   = d["left_shoulder"]  # (3,)
    rs   = d["right_shoulder"] # (3,)
    lh   = d["left_hand"]      # (21, 3)
    rh   = d["right_hand"]     # (21, 3)
    # nose is (0,0,0) post-normalization

    sh_mid = (ls + rs) / 2.0   # shoulder midpoint

    # 0: mouth openness — vertical distance between upper and lower lip
    upper_lip = face[_F_UPPER_LIP, :2]
    lower_lip = face[_F_LOWER_LIP, :2]
    feats[0] = float(np.linalg.norm(upper_lip - lower_lip))

    # 1: left eye aspect ratio — horizontal span (outer to inner corner)
    feats[1] = float(np.linalg.norm(
        face[_F_LE_OUTER, :2] - face[_F_LE_INNER, :2]
    ))

    # 2: right eye aspect ratio
    feats[2] = float(np.linalg.norm(
        face[_F_RE_INNER, :2] - face[_F_RE_OUTER, :2]
    ))

    # 3: head tilt angle — angle of shoulder line from horizontal (degrees)
    dy = float(ls[1] - rs[1])
    dx = float(ls[0] - rs[0]) + 1e-9
    feats[3] = float(np.degrees(np.arctan2(dy, dx)))

    # 4: forward head posture — nose x relative to shoulder midpoint
    # nose is at origin (0,0,0); shoulder midpoint x is how far
    # the shoulders are displaced in x from nose → negated = head forward
    feats[4] = float(-sh_mid[0])

    # 5: head nod angle — nose y relative to shoulder midpoint y
    feats[5] = float(-sh_mid[1])

    # 6: shoulder tilt (signed)
    feats[6] = float(ls[1] - rs[1])

    # 7: shoulder symmetry (unsigned)
    feats[7] = float(abs(ls[1] - rs[1]))

    # 8: shoulder width (post-norm; should be ~1 at calibration)
    feats[8] = float(np.linalg.norm(ls[:2] - rs[:2]))

    # 9: left wrist → nose (origin = 0)
    lw = lh[_HAND_WRIST, :2]
    feats[9] = float(np.linalg.norm(lw))

    # 10: right wrist → nose
    rw = rh[_HAND_WRIST, :2]
    feats[10] = float(np.linalg.norm(rw))

    # 11: left wrist → left shoulder
    feats[11] = float(np.linalg.norm(lw - ls[:2]))

    # 12: right wrist → right shoulder
    feats[12] = float(np.linalg.norm(rw - rs[:2]))

    # 13: hand symmetry
    feats[13] = float(abs(feats[9] - feats[10]))

    # 14, 15: velocity — filled by _temporal_features
    # 16, 17: acceleration — filled by _temporal_features
    # 18: posture_stability — filled by _posture_stability

    # 19: gaze proxy — face nose-tip z (depth relative to shoulder plane)
    feats[19] = float(face[_F_NOSE_TIP, 2])

    return feats


# ── Temporal features (velocity + acceleration) ───────────────────────────────
def _temporal_features(raw: np.ndarray, dicts: List[Dict]) -> np.ndarray:
    """Fill columns 14-17 with wrist velocity and acceleration."""
    lw_pos = np.array(
        [d["left_hand"][_HAND_WRIST, :2]  for d in dicts], dtype=np.float64
    )  # (75, 2)
    rw_pos = np.array(
        [d["right_hand"][_HAND_WRIST, :2] for d in dicts], dtype=np.float64
    )

    # Velocity: L2 magnitude of central finite difference
    lv = np.linalg.norm(np.gradient(lw_pos, axis=0), axis=1)  # (75,)
    rv = np.linalg.norm(np.gradient(rw_pos, axis=0), axis=1)

    # Acceleration: derivative of velocity
    la = np.abs(np.gradient(lv))
    ra = np.abs(np.gradient(rv))

    raw[:, 14] = lv
    raw[:, 15] = rv
    raw[:, 16] = la
    raw[:, 17] = ra
    return raw


# ── Posture stability ─────────────────────────────────────────────────────────
def _posture_stability(dicts: List[Dict], window: int = 5) -> np.ndarray:
    """Rolling variance of shoulder midpoint x-y over `window` frames."""
    mid = np.array(
        [(d["left_shoulder"][:2] + d["right_shoulder"][:2]) / 2.0
         for d in dicts],
        dtype=np.float64,
    )  # (75, 2)

    out = np.zeros(len(dicts), dtype=np.float64)
    half = window // 2
    n = len(dicts)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.var(mid[lo:hi]))
    return out


# ── Smoothing ─────────────────────────────────────────────────────────────────
def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Per-feature rolling mean, same-length output (edge = partial windows)."""
    out = np.empty_like(arr)
    kernel = np.ones(window) / window
    for f in range(arr.shape[1]):
        out[:, f] = np.convolve(arr[:, f], kernel, mode="same")
    return out


# ── Validation ────────────────────────────────────────────────────────────────
def validate_features(arr: np.ndarray) -> None:
    """Assert invariants on the feature matrix."""
    assert arr.shape == (N_FRAMES, N_FEATURES), \
        f"Expected ({N_FRAMES},{N_FEATURES}), got {arr.shape}"
    assert arr.dtype == np.float32, \
        f"dtype {arr.dtype} != float32"
    assert not np.any(np.isnan(arr)), "NaN in feature matrix"
    assert not np.any(np.isinf(arr)), "Inf in feature matrix"
    assert np.all(arr >= -CLIP_RANGE - 1e-4), "Value below clip range"
    assert np.all(arr <=  CLIP_RANGE + 1e-4), "Value above clip range"


# ── Feature name manifest (fixed) ────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "mouth_openness",
    "eye_aspect_ratio_left",
    "eye_aspect_ratio_right",
    "head_tilt_angle",
    "forward_head_posture",
    "head_nod_angle",
    "shoulder_tilt",
    "shoulder_symmetry",
    "shoulder_width",
    "left_hand_face_dist",
    "right_hand_face_dist",
    "left_hand_shoulder_dist",
    "right_hand_shoulder_dist",
    "hand_symmetry",
    "left_hand_velocity",
    "right_hand_velocity",
    "left_hand_acceleration",
    "right_hand_acceleration",
    "posture_stability",
    "gaze_proxy",
]
assert len(FEATURE_NAMES) == N_FEATURES, \
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries, expected {N_FEATURES}"
