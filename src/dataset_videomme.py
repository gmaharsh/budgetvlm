"""Video-MME dataset loader + synthetic subset for smoke tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, project_path

log = get_logger("dataset")


@dataclass
class QAExample:
    video_id: str
    question_id: str
    video_path: str
    question: str
    options: list[str]
    answer: str  # A/B/C/D
    duration: str = ""
    domain: str = ""
    task_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_options(row: dict[str, Any]) -> list[str]:
    opts = row.get("options")
    if opts is None:
        opts = row.get("candidates")
    if opts is None:
        # common HF columns: option_A ...
        letters = ["A", "B", "C", "D"]
        built = []
        for L in letters:
            key = f"option_{L}" if f"option_{L}" in row else L
            if key in row and row[key] is not None:
                val = str(row[key])
                built.append(val if val.strip().upper().startswith(L) else f"{L}. {val}")
        if built:
            return built
        raise KeyError("Could not find options in row")
    if isinstance(opts, str):
        import ast

        opts = ast.literal_eval(opts)
    # pandas/pyarrow may give numpy arrays
    opts = list(opts)
    out = []
    for i, o in enumerate(opts):
        L = chr(ord("A") + i)
        s = str(o)
        out.append(s if s.strip().upper().startswith(L) else f"{L}. {s}")
    return out


def _normalize_answer(ans: Any) -> str:
    s = str(ans).strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    if s.startswith(("A", "B", "C", "D")) and len(s) <= 3:
        return s[0]
    # sometimes answer is full option text — leave as-is for letter extract
    return s[0] if s and s[0] in "ABCD" else s


def load_videomme_annotations(
    root: str | Path | None = None,
    limit: int | None = None,
    video_ids: list[str] | None = None,
) -> list[QAExample]:
    """Load Video-MME QA rows. Prefers local parquet/csv under data/videomme."""
    root = Path(root) if root else project_path("data", "videomme")
    ensure_dir(root)

    parquet = root / "videomme.parquet"
    csv_path = root / "videomme.csv"
    if not parquet.exists() and not csv_path.exists():
        raise FileNotFoundError(
            f"No Video-MME annotations at {root}. "
            "Run: python -m src.download_videomme --annotations-only"
        )

    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        df = pd.read_csv(csv_path)

    rows: list[QAExample] = []
    video_dir = root / "videos"
    for i, rec in df.iterrows():
        d = rec.to_dict()
        vid = str(d.get("video_id") or d.get("videoID") or d.get("video") or "")
        if video_ids is not None and vid not in video_ids:
            continue
        qid = str(d.get("question_id") or d.get("question_id") or d.get("id") or f"{vid}_{i}")
        question = str(d.get("question") or d.get("Question") or "")
        answer = _normalize_answer(d.get("answer") or d.get("Answer") or d.get("response") or "")
        options = _normalize_options(d)

        # locate video file
        candidates = [
            video_dir / f"{vid}.mp4",
            video_dir / vid / f"{vid}.mp4",
            video_dir / f"{vid}.webm",
            Path(str(d.get("video_path") or "")),
        ]
        vpath = next((str(p) for p in candidates if p and Path(p).exists()), "")
        rows.append(
            QAExample(
                video_id=vid,
                question_id=qid,
                video_path=vpath,
                question=question,
                options=options,
                answer=answer,
                duration=str(d.get("duration") or d.get("duration_category") or ""),
                domain=str(d.get("domain") or ""),
                task_type=str(d.get("task_type") or d.get("sub_category") or ""),
            )
        )

    if limit is not None:
        keep: list[QAExample] = []
        seen: list[str] = []
        for r in rows:
            if r.video_id not in seen:
                if len(seen) >= limit:
                    break
                seen.append(r.video_id)
            keep.append(r)
        rows = keep

    log.info("Loaded %d QA pairs across %d videos", len(rows), len({r.video_id for r in rows}))
    return rows


def make_synthetic_dataset(
    n_videos: int = 10,
    questions_per_video: int = 1,
    out_dir: str | Path | None = None,
    seed: int = 42,
) -> list[QAExample]:
    """Create tiny synthetic videos + MCQs so the pipeline runs without GPU/HF."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir) if out_dir else project_path("data", "synthetic")
    video_dir = ensure_dir(out_dir / "videos")
    examples: list[QAExample] = []

    for v in range(n_videos):
        # complexity proxy baked into motion amplitude
        motion = float(rng.uniform(0.02, 0.85))
        vid = f"synth_{v:03d}"
        path = video_dir / f"{vid}.mp4"
        _write_synth_video(path, motion=motion, seed=seed + v)

        # Label from measured complexity so mock GT aligns with the estimator
        from .complexity import score_video

        cx = score_video(str(path), vid, num_frames=8, use_scene=False).complexity
        answer = "A" if cx < 0.15 else ("B" if cx < 0.40 else "C")
        options = [
            "A. Low motion / mostly static scene",
            "B. Moderate motion / some scene change",
            "C. High motion / frequent scene change",
            "D. Audio-only content with no visuals",
        ]
        for q in range(questions_per_video):
            examples.append(
                QAExample(
                    video_id=vid,
                    question_id=f"{vid}_q{q}",
                    video_path=str(path),
                    question="Which description best matches the visual dynamics of this video?",
                    options=options,
                    answer=answer,
                    duration="short",
                    domain="synthetic",
                    task_type="perception",
                )
            )

    # also write a parquet for the regular loader
    pd.DataFrame([e.to_dict() for e in examples]).to_parquet(out_dir / "videomme.parquet")
    log.info("Synthetic dataset: %d videos -> %s", n_videos, out_dir)
    return examples


def _write_synth_video(path: Path, motion: float, seed: int, n_frames: int = 48, size: int = 128) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (size, size),
    )
    base = rng.integers(40, 200, size=(size, size, 3), dtype=np.uint8)
    for t in range(n_frames):
        frame = base.copy()
        shift = int(motion * 40 * np.sin(t / 3.0))
        frame = np.roll(frame, shift, axis=1)
        # occasional hard cut for high motion
        if motion > 0.5 and t % max(3, int(12 * (1 - motion))) == 0:
            frame = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        noise = rng.normal(0, 8 + 40 * motion, size=frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def iter_unique_videos(examples: list[QAExample]) -> Iterator[tuple[str, list[QAExample]]]:
    by: dict[str, list[QAExample]] = {}
    for e in examples:
        by.setdefault(e.video_id, []).append(e)
    for vid, qs in by.items():
        yield vid, qs
