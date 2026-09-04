"""Download Video-MME annotations (and optionally a small video chunk)."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from .utils import ensure_dir, get_logger, load_config, project_path

log = get_logger("download")


def download_annotations(root: Path) -> Path:
    """Pull the HF parquet/arrow annotations without all video zips."""
    from datasets import load_dataset

    ensure_dir(root)
    log.info("Downloading lmms-lab/Video-MME annotations...")
    ds = load_dataset("lmms-lab/Video-MME", split="test")
    out = root / "videomme.parquet"
    ds.to_pandas().to_parquet(out)
    log.info("Wrote %s (%d rows)", out, len(ds))
    return out


def download_first_video_chunk(root: Path) -> Path:
    """Download videos_chunked_01.zip (~first videos) for small-scale experiments."""
    from huggingface_hub import hf_hub_download

    ensure_dir(root / "videos")
    log.info("Downloading videos_chunked_01.zip (this can take a while)...")
    zpath = hf_hub_download(
        repo_id="lmms-lab/Video-MME",
        filename="videos_chunked_01.zip",
        repo_type="dataset",
        local_dir=str(root / "hf"),
    )
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(root / "videos")
    log.info("Extracted videos to %s", root / "videos")
    return root / "videos"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--with-videos-chunk1", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_path(*(cfg["dataset"]["root"].split("/")))
    download_annotations(root)
    if args.with_videos_chunk1 and not args.annotations_only:
        download_first_video_chunk(root)
    elif not args.annotations_only:
        log.info("Annotations ready. Pass --with-videos-chunk1 to fetch the first video zip.")


if __name__ == "__main__":
    main()
