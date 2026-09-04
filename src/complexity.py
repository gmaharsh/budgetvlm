"""Video complexity estimators (no VLM required)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .video_io import load_frames


def _to_gray_norm(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        # ITU-R BT.601
        g = (
            0.299 * frame[..., 0].astype(np.float32)
            + 0.587 * frame[..., 1].astype(np.float32)
            + 0.114 * frame[..., 2].astype(np.float32)
        )
    else:
        g = frame.astype(np.float32)
    return g / 255.0


def frame_difference_complexity(frames: list[np.ndarray]) -> float:
    """Mean absolute frame-to-frame difference after normalization."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    prev = _to_gray_norm(frames[0])
    for fr in frames[1:]:
        cur = _to_gray_norm(fr)
        # resize-safe: if shapes differ, crop to min
        h = min(prev.shape[0], cur.shape[0])
        w = min(prev.shape[1], cur.shape[1])
        diffs.append(float(np.mean(np.abs(cur[:h, :w] - prev[:h, :w]))))
        prev = cur
    return float(np.mean(diffs)) if diffs else 0.0


def scene_change_score(frames: list[np.ndarray], threshold: float = 0.35) -> float:
    """Fraction of consecutive pairs whose diff exceeds threshold."""
    if len(frames) < 2:
        return 0.0
    diffs = []
    prev = _to_gray_norm(frames[0])
    for fr in frames[1:]:
        cur = _to_gray_norm(fr)
        h = min(prev.shape[0], cur.shape[0])
        w = min(prev.shape[1], cur.shape[1])
        diffs.append(float(np.mean(np.abs(cur[:h, :w] - prev[:h, :w]))))
        prev = cur
    if not diffs:
        return 0.0
    return float(np.mean([d >= threshold for d in diffs]))


@dataclass
class ComplexityResult:
    video_id: str
    complexity: float
    motion: float
    scene: float
    n_frames: int


def score_video(
    video_path: str,
    video_id: str,
    num_frames: int = 8,
    alpha: float = 0.7,
    beta: float = 0.3,
    scene_threshold: float = 0.35,
    use_scene: bool = False,
) -> ComplexityResult:
    frames, _ = load_frames(video_path, num_frames=num_frames, resize_max=256)
    motion = frame_difference_complexity(frames)
    scene = scene_change_score(frames, threshold=scene_threshold) if use_scene else 0.0
    if use_scene:
        c = alpha * motion + beta * scene
    else:
        c = motion
    return ComplexityResult(
        video_id=video_id,
        complexity=float(c),
        motion=float(motion),
        scene=float(scene),
        n_frames=len(frames),
    )
