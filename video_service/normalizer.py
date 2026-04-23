"""
normalizer.py
=============
Stage 3 — Ego-centric normalization.

Compatible: Python 3.10.11, numpy 1.24+

Two-step normalization per frame:
    Step 1 — Centering   : subtract nose → nose becomes (0, 0, 0)
    Step 2 — Scaling     : divide by shoulder width (Euclidean distance L↔R)

Edge cases:
    - Shoulder width ≈ 0 (collapsed / occluded) → use nearest valid scale
    - All frames have zero shoulder width         → scale = 1.0 (identity)
    - Any NaN/Inf produced                        → replaced with 0.0
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

EPS: float = 1e-6  # minimum shoulder width to be considered valid


def normalize(landmark_dicts: List[Dict]) -> List[Dict]:
    """
    Apply ego-centric normalization to each of 75 frames independently.

    Parameters
    ----------
    landmark_dicts : output of LandmarkExtractor.extract() — 75 dicts

    Returns
    -------
    List[Dict]  — 75 normalized dicts, same structure, all float32, no NaN/Inf
    """
    assert len(landmark_dicts) == 75, \
        f"Expected 75 dicts, got {len(landmark_dicts)}"

    dicts = deepcopy(landmark_dicts)

    raw_scales = [_shoulder_width(d) for d in dicts]
    scales     = _fill_scales(raw_scales)

    for d, scale in zip(dicts, scales):
        _normalize_frame(d, scale)

    _assert_no_nan_inf(dicts)
    return dicts


def _shoulder_width(d: Dict) -> float:
    """Euclidean distance L↔R shoulder in original (pre-center) coordinates."""
    dist = float(np.linalg.norm(d["left_shoulder"] - d["right_shoulder"]))
    return dist if dist >= EPS else 0.0


def _fill_scales(raw: List[float]) -> List[float]:
    """Forward-fill then backward-fill zero scales; absolute fallback = 1.0."""
    n = len(raw)
    scales = list(raw)

    last: Optional[float] = None
    for i in range(n):
        if scales[i] > 0:
            last = scales[i]
        elif last is not None:
            scales[i] = last

    last = None
    for i in range(n - 1, -1, -1):
        if scales[i] > 0:
            last = scales[i]
        elif last is not None:
            scales[i] = last

    scales = [s if s > 0 else 1.0 for s in scales]

    n_invalid = sum(1 for r in raw if r == 0)
    if n_invalid > 0:
        log.debug(
            f"[Normalizer] {n_invalid}/{n} frames had zero shoulder distance; "
            "filled from neighbours."
        )
    return scales


def _normalize_frame(d: Dict, scale: float) -> None:
    """In-place: center on nose, then scale by 1/shoulder_width."""
    origin = d["nose"].copy()  # (3,)
    inv    = 1.0 / scale

    scalar_keys = ("nose", "left_shoulder", "right_shoulder")
    array_keys  = ("face", "left_hand", "right_hand")

    for k in scalar_keys:
        d[k] = ((d[k] - origin) * inv).astype(np.float32)

    for k in array_keys:
        d[k] = ((d[k] - origin[np.newaxis, :]) * inv).astype(np.float32)

    # Enforce exact zero for nose (eliminates float rounding residuals)
    d["nose"] = np.zeros(3, dtype=np.float32)


def _assert_no_nan_inf(dicts: List[Dict]) -> None:
    all_keys = ("nose", "left_shoulder", "right_shoulder",
                "face", "left_hand", "right_hand")
    for i, d in enumerate(dicts):
        for k in all_keys:
            arr = d[k]
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                log.warning(
                    f"[Normalizer] frame {i} '{k}': NaN/Inf detected → replaced with 0"
                )
                d[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def validate_normalized(dicts: List[Dict]) -> None:
    """Assert invariants on normalized landmark list."""
    assert len(dicts) == 75
    for i, d in enumerate(dicts):
        assert np.allclose(d["nose"], 0.0, atol=1e-5), \
            f"Frame {i}: nose is not (0,0,0): {d['nose']}"
        for k in ("left_shoulder", "right_shoulder", "face",
                  "left_hand", "right_hand"):
            assert not np.any(np.isnan(d[k])), f"Frame {i}: NaN in '{k}'"
            assert not np.any(np.isinf(d[k])), f"Frame {i}: Inf in '{k}'"
