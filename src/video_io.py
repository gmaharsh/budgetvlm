"""Video loading and uniform frame sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    duration_sec: float
    fps: float
    n_frames_total: int
    width: int
    height: int
    # Indices into the original video for the returned frames (uniform sample).
    sampled_indices: list[int] | None = None


def probe_video(path: str | Path) -> VideoMeta:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (n / fps) if fps > 0 else 0.0
    cap.release()
    return VideoMeta(path=path, duration_sec=duration, fps=fps, n_frames_total=n, width=w, height=h)


def _sample_indices(n_total: int, n_keep: int) -> list[int]:
    if n_total <= 0:
        raise ValueError("video has no frames")
    n_keep = max(1, min(n_keep, n_total))
    if n_keep == 1:
        return [n_total // 2]
    return [int(round(i * (n_total - 1) / (n_keep - 1))) for i in range(n_keep)]


def _maybe_resize(rgb: np.ndarray, resize_max: int | None) -> np.ndarray:
    if resize_max is None:
        return rgb
    h, w = rgb.shape[:2]
    scale = resize_max / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(
        rgb,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def load_frames(
    path: str | Path,
    num_frames: int,
    resize_max: int | None = None,
) -> tuple[list[np.ndarray], VideoMeta]:
    """Return RGB uint8 frames (H, W, 3) sampled uniformly (OpenCV)."""
    meta = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []

    sampled_indices: list[int] = []
    if meta.n_frames_total > 0:
        idxs = _sample_indices(meta.n_frames_total, num_frames)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            frames.append(_maybe_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resize_max))
            sampled_indices.append(idx)
    else:
        # Some containers report 0 frame count; sequential decode then subsample.
        raw: list[np.ndarray] = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            raw.append(_maybe_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), resize_max))
        if raw:
            idxs = _sample_indices(len(raw), num_frames)
            frames = [raw[i] for i in idxs]
            sampled_indices = list(idxs)
            meta = VideoMeta(
                path=meta.path,
                duration_sec=len(raw) / meta.fps if meta.fps > 0 else 0.0,
                fps=meta.fps,
                n_frames_total=len(raw),
                width=raw[0].shape[1],
                height=raw[0].shape[0],
                sampled_indices=sampled_indices,
            )

    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode any frames from {path}")
    if meta.sampled_indices is None:
        meta = VideoMeta(
            path=meta.path,
            duration_sec=meta.duration_sec,
            fps=meta.fps,
            n_frames_total=meta.n_frames_total,
            width=meta.width,
            height=meta.height,
            sampled_indices=sampled_indices,
        )
    return frames, meta


def frames_to_temp_video(
    frames: Sequence[np.ndarray],
    out_path: str | Path,
    fps: float = 2.0,
) -> Path:
    """Write RGB frames to an mp4 for backends that require a file path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for rgb in frames:
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path
