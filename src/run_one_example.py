"""Run one Video-MME (or synthetic) example through the selected backend."""
from __future__ import annotations

import argparse
import json

from .backends import get_backend
from .dataset_videomme import load_videomme_annotations, make_synthetic_dataset
from .infer import run_one
from .utils import get_logger, load_config, project_path, set_seed

log = get_logger("one")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default=None, help="mock | vllm")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--pruning-rate", type=float, default=0.0)
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    backend_name = args.backend or cfg["serving"]["backend"]

    if args.synthetic or backend_name == "mock":
        examples = make_synthetic_dataset(n_videos=3, seed=cfg["seed"])
    else:
        examples = load_videomme_annotations(
            root=project_path(*(cfg["dataset"]["root"].split("/"))),
            limit=5,
            video_ids=[args.video_id] if args.video_id else None,
        )

    if args.video_id:
        examples = [e for e in examples if e.video_id == args.video_id]
    if not examples:
        raise SystemExit("No examples found")
    ex = examples[0]
    if not ex.video_path:
        raise SystemExit(f"Video file missing for {ex.video_id}")

    backend = get_backend(backend_name, cfg, pruning_rate=args.pruning_rate)
    try:
        row = run_one(
            backend,
            ex,
            pruning_rate=args.pruning_rate,
            baseline_num_frames=int(cfg["video"]["baseline_num_frames"]),
            max_pixels=cfg["video"].get("max_pixels"),
        )
    finally:
        backend.close()
    print(json.dumps(row, indent=2))
    log.info(
        "prediction=%s gt=%s correct=%s latency=%.3fs frames=%s tokens_pre=%s tokens_ret_est=%s",
        row["prediction"],
        row["ground_truth"],
        row["correct"],
        row["latency_sec"],
        row["n_frames"],
        row.get("visual_tokens_pre"),
        row.get("visual_tokens_retained_est"),
    )


if __name__ == "__main__":
    main()
