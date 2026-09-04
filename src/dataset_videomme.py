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


def _index_local_videos(video_dir: Path) -> dict[str, Path]:
    """Map file stem -> path for all mp4/webm under video_dir (recursive)."""
    if not video_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for pattern in ("*.mp4", "*.webm", "*.mkv"):
        for p in video_dir.rglob(pattern):
            if p.is_file():
                out.setdefault(p.stem, p)
    return out


def resolve_video_path(
    video_dir: Path,
    *,
    video_id: str,
    youtube_id: str = "",
    row_path: str = "",
    index: dict[str, Path] | None = None,
) -> str:
    """Resolve on-disk path for a Video-MME row.

    Official files are named ``{videoID}.mp4`` (YouTube id), not ``{video_id}``.
    Never returns ``.`` / cwd — empty string means missing.
    """
    if row_path:
        rp = Path(row_path)
        if rp.is_file():
            return str(rp)

    names: list[str] = []
    for n in (youtube_id, video_id):
        n = str(n or "").strip()
        if n and n not in names:
            names.append(n)

    index = index if index is not None else _index_local_videos(video_dir)
    for n in names:
        hit = index.get(n)
        if hit is not None and hit.is_file():
            return str(hit)

    # Shallow fallbacks (no empty Path — Path("") == "." and exists!)
    for n in names:
        for p in (
            video_dir / f"{n}.mp4",
            video_dir / "data" / f"{n}.mp4",
            video_dir / n / f"{n}.mp4",
            video_dir / f"{n}.webm",
        ):
            if p.is_file():
                return str(p)
    return ""


def load_videomme_annotations(
    root: str | Path | None = None,
    limit: int | None = None,
    video_ids: list[str] | None = None,
    require_video: bool = False,
) -> list[QAExample]:
    """Load Video-MME QA rows. Prefers local parquet/csv under data/videomme.

    If ``limit`` is set, prefer videos that exist on disk so chunk-1 experiments
    do not silently pick annotation rows whose mp4s were never downloaded.
    """
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

    video_dir = root / "videos"
    index = _index_local_videos(video_dir)
    if index:
        log.info("Indexed %d local video files under %s", len(index), video_dir)

    rows: list[QAExample] = []
    for i, rec in df.iterrows():
        d = rec.to_dict()
        # video_id = dataset ordinal; videoID = YouTube id used for filenames
        vid = str(d.get("video_id") or d.get("video") or "")
        yt = str(d.get("videoID") or d.get("video_ID") or "")
        if not vid:
            vid = yt
        if video_ids is not None and vid not in video_ids and yt not in video_ids:
            continue
        qid = str(d.get("question_id") or d.get("id") or f"{vid}_{i}")
        question = str(d.get("question") or d.get("Question") or "")
        answer = _normalize_answer(d.get("answer") or d.get("Answer") or d.get("response") or "")
        options = _normalize_options(d)

        vpath = resolve_video_path(
            video_dir,
            video_id=vid,
            youtube_id=yt,
            row_path=str(d.get("video_path") or ""),
            index=index,
        )
        if require_video and not vpath:
            continue
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
        # Prefer rows with on-disk videos so chunk-1 runs are usable.
        ordered = sorted(rows, key=lambda r: (0 if r.video_path else 1))
        keep: list[QAExample] = []
        seen: list[str] = []
        for r in ordered:
            if r.video_id not in seen:
                if len(seen) >= limit:
                    break
                if not r.video_path:
                    # skip missing until we have enough present, unless none exist
                    continue
                seen.append(r.video_id)
            if r.video_id in seen:
                keep.append(r)
        if not keep:
            # Fall back to first N annotation videos (paths may still be empty).
            keep = []
            seen = []
            for r in rows:
                if r.video_id not in seen:
                    if len(seen) >= limit:
                        break
                    seen.append(r.video_id)
                keep.append(r)
        rows = keep

    n_with = sum(1 for r in rows if r.video_path)
    log.info(
        "Loaded %d QA pairs across %d videos (%d rows with local mp4)",
        len(rows),
        len({r.video_id for r in rows}),
        n_with,
    )
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
