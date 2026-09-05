"""Phase 7–8: compute video complexity scores."""
from __future__ import annotations

import argparse

from .complexity import score_video
from .dataset_videomme import iter_unique_videos, load_videomme_annotations, make_synthetic_dataset
from .utils import get_logger, load_config, project_path, read_jsonl, write_jsonl

log = get_logger("complexity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--backend",
        default=None,
        help="mock → synthetic videos; vllm → Video-MME (defaults to config serving.backend)",
    )
    ap.add_argument(
        "--predictions",
        default=None,
        help="If set, only score video_ids present in this predictions jsonl (recommended)",
    )
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--use-scene", action="store_true", help="Phase 8 combined metric")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    limit = args.limit if args.limit is not None else int(cfg["dataset"]["default_limit"])
    backend_name = (args.backend or cfg["serving"]["backend"]).lower()

    video_ids = None
    if args.predictions:
        video_ids = sorted({str(r["video_id"]) for r in read_jsonl(project_path(args.predictions))})
        log.info("Restricting complexity to %d videos from predictions", len(video_ids))

    if args.synthetic or backend_name == "mock":
        examples = make_synthetic_dataset(n_videos=limit, seed=cfg["seed"])
    else:
        examples = load_videomme_annotations(
            root=project_path(*(cfg["dataset"]["root"].split("/"))),
            limit=None if video_ids is not None else limit,
            video_ids=video_ids,
            require_video=True,
        )

    rows = []
    for vid, qs in iter_unique_videos(examples):
        path = qs[0].video_path
        if not path:
            log.warning("skip %s: missing video", vid)
            continue
        r = score_video(
            path,
            vid,
            num_frames=int(cfg["complexity"]["num_frames"]),
            alpha=float(cfg["complexity"]["alpha_motion"]),
            beta=float(cfg["complexity"]["beta_scene"]),
            scene_threshold=float(cfg["complexity"]["scene_threshold"]),
            use_scene=args.use_scene,
        )
        rows.append(
            {
                "video_id": r.video_id,
                "complexity": r.complexity,
                "motion": r.motion,
                "scene": r.scene,
                "n_frames": r.n_frames,
            }
        )
        log.info("%s complexity=%.4f motion=%.4f scene=%.4f", vid, r.complexity, r.motion, r.scene)

    out = project_path(args.out) if args.out else project_path("results", "metrics", "complexity.jsonl")
    write_jsonl(out, rows)
    log.info("Wrote %s (%d videos)", out, len(rows))


if __name__ == "__main__":
    main()
