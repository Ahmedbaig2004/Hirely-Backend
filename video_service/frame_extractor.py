"""
frame_extractor.py
==================
Stage 1 — Deterministic 5fps frame extraction, exactly 75 frames output.

Compatible: Python 3.10.11, OpenCV 4.x, numpy 1.24+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

log = logging.getLogger(__name__)

TARGET_FPS: int = 5
TARGET_FRAMES: int = 75
MAX_DECODE_ATTEMPTS: int = 3


def extract_frames(
    video_path: str,
    target_fps: int = TARGET_FPS,
    target_frames: int = TARGET_FRAMES,
) -> List[np.ndarray]:
    """
    Extract exactly `target_frames` BGR frames sampled at `target_fps`.

    Strategy:
        1. Read native FPS from video metadata.
        2. Compute frame-index step = native_fps / target_fps.
        3. Seek to each required index; retry on decode failure.
        4. Pad with last good frame if total < target_frames.
        5. Truncate to target_frames if total > target_frames.

    Parameters
    ----------
    video_path    : str path to video file
    target_fps    : desired sampling rate (default 5)
    target_frames : fixed output length   (default 75)

    Returns
    -------
    List[np.ndarray]  length == target_frames, each shape (H, W, 3) uint8

    Raises
    ------
    FileNotFoundError : video_path does not exist
    RuntimeError      : video cannot be opened at all
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {video_path}")

    try:
        return _extract(cap, target_fps, target_frames, str(p))
    finally:
        cap.release()


def extract_frames_batch(
    video_paths: List[str],
    target_fps: int = TARGET_FPS,
    target_frames: int = TARGET_FRAMES,
) -> List[Tuple[str, object]]:
    """
    Process multiple videos. Returns list of (path, frames_or_exception).
    Per-video failures are captured so one bad file does not abort the batch.
    """
    results: List[Tuple[str, object]] = []
    for path in video_paths:
        try:
            frames = extract_frames(path, target_fps, target_frames)
            results.append((str(path), frames))
        except Exception as exc:
            log.error(f"[FrameExtractor] Failed {path}: {exc}")
            results.append((str(path), exc))
    return results


# ── internal ──────────────────────────────────────────────────────────────────
def _extract(
    cap: cv2.VideoCapture,
    target_fps: int,
    target_frames: int,
    label: str,
) -> List[np.ndarray]:
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_native <= 0:
        log.warning(
            f"[FrameExtractor] {label}: frame count metadata={total_native}; "
            "reading until EOF."
        )
        total_native = int(1e9)

    step = native_fps / target_fps
    wanted_indices = [round(i * step) for i in range(target_frames)]
    wanted_indices = [min(idx, max(total_native - 1, 0)) for idx in wanted_indices]

    frames: List[Optional[np.ndarray]] = []
    last_good: Optional[np.ndarray] = None

    for native_idx in wanted_indices:
        frame = _read_frame_at(cap, native_idx, label)
        if frame is None:
            frame = last_good
        if frame is not None:
            last_good = frame
            frames.append(cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT)))
        else:
            frames.append(None)

    # Resolve any leading Nones using first real frame (backward pass)
    first_real = next((f for f in frames if f is not None), None)
    if first_real is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        log.error(
            f"[FrameExtractor] {label}: zero decodeable frames — "
            "returning black placeholder frames."
        )
        return [placeholder] * target_frames

    frames = [f if f is not None else first_real for f in frames]

    while len(frames) < target_frames:
        frames.append(frames[-1].copy())  # type: ignore[arg-type]
    frames = frames[:target_frames]

    assert len(frames) == target_frames, \
        f"BUG: expected {target_frames}, got {len(frames)}"

    log.debug(
        f"[FrameExtractor] {label}: {len(frames)} frames extracted "
        f"(native_fps={native_fps:.1f}, step={step:.2f})"
    )
    return frames  # type: ignore[return-value]


def _read_frame_at(
    cap: cv2.VideoCapture,
    target_idx: int,
    label: str,
) -> Optional[np.ndarray]:
    for attempt in range(MAX_DECODE_ATTEMPTS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame
        log.debug(
            f"[FrameExtractor] {label}: seek={target_idx} "
            f"attempt {attempt + 1}/{MAX_DECODE_ATTEMPTS} failed"
        )
    return None


def validate_frames(frames: List[np.ndarray]) -> None:
    """Assert shape / dtype invariants on the returned frame list."""
    assert len(frames) == TARGET_FRAMES, \
        f"Expected {TARGET_FRAMES} frames, got {len(frames)}"
    for i, f in enumerate(frames):
        assert isinstance(f, np.ndarray), f"Frame {i} is not ndarray"
        assert f.ndim == 3 and f.shape[2] == 3, \
            f"Frame {i} shape {f.shape} is not (H, W, 3)"
        assert f.dtype == np.uint8, f"Frame {i} dtype {f.dtype} != uint8"
