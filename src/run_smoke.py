"""Phase 1–3 smoke: synthetic videos + mock backend, end-to-end."""
from __future__ import annotations

import argparse

from .backends import get_backend
from .compare_policies import build_comparison
from .complexity import score_video
from .dataset_videomme import iter_unique_videos, make_synthetic_dataset
from .evaluate import accuracy, summarize_by_pruning
from .infer import run_matrix
from .policy import (
    adaptive_pruning_rate,
    aggregate_video_correctness,
    max_observed_correct_rate,
    max_safe_pruning_rate,
)
from .utils import ensure_dir, get_logger, load_config, project_path, set_seed, write_json, write_jsonl

log = get_logger("smoke")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-videos", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    examples = make_synthetic_dataset(
        n_videos=args.n_videos, questions_per_video=1, seed=cfg["seed"]
    )
    cfg["serving"]["backend"] = "mock"
    backend = get_backend("mock", cfg)

    rates = list(cfg["pruning"]["rates"])
    out = project_path("results", "predictions", "smoke_fixed.jsonl")
    rows = run_matrix(
        backend,
        examples,
        rates,
        baseline_num_frames=int(cfg["video"]["baseline_num_frames"]),
        max_pixels=cfg["video"].get("max_pixels"),
        out_jsonl=str(out),
    )

    summary = summarize_by_pruning(rows)
    ensure_dir(project_path("results", "metrics"))
    summary.to_csv(project_path("results", "metrics", "smoke_by_rate.csv"), index=False)
    log.info("Accuracy overall=%.3f", accuracy(rows))
    log.info("\n%s", summary.to_string(index=False))

    by_vid = aggregate_video_correctness(rows)
    tolerance = []
    for vid, outcomes in by_vid.items():
        tolerance.append(
            {
                "video_id": vid,
                "max_safe_pruning_rate": max_safe_pruning_rate(outcomes, rates),
                "max_observed_correct_rate": max_observed_correct_rate(outcomes, rates),
                **{f"all_correct_p{int(r * 100):02d}": outcomes.get(r, False) for r in rates},
            }
        )
    write_jsonl(project_path("results", "metrics", "smoke_tolerance.jsonl"), tolerance)

    cx_rows = []
    for vid, qs in iter_unique_videos(examples):
        r = score_video(
            qs[0].video_path,
            vid,
            num_frames=int(cfg["complexity"]["num_frames"]),
            use_scene=False,
        )
        cx_rows.append(
            {
                "video_id": r.video_id,
                "complexity": r.complexity,
                "motion": r.motion,
                "scene": r.scene,
            }
        )
    write_jsonl(project_path("results", "metrics", "smoke_complexity.jsonl"), cx_rows)

    cx_map = {r["video_id"]: r["complexity"] for r in cx_rows}
    t1, t2 = float(cfg["complexity"]["t1"]), float(cfg["complexity"]["t2"])
    index = {(r["video_id"], r["question_id"], float(r["pruning_rate"])): r for r in rows}
    budget_rows = []
    oracle_rows = []
    for ex in examples:
        rate = adaptive_pruning_rate(cx_map[ex.video_id], t1, t2)
        budget_rows.append(index[(ex.video_id, ex.question_id, rate)])
        safe = next(t["max_safe_pruning_rate"] for t in tolerance if t["video_id"] == ex.video_id)
        oracle_rows.append(index[(ex.video_id, ex.question_id, float(safe))])

    comparison = build_comparison(
        {
            "No pruning": [r for r in rows if r["pruning_rate"] == 0.0],
            "Fixed 25%": [r for r in rows if r["pruning_rate"] == 0.25],
            "Fixed 50%": [r for r in rows if r["pruning_rate"] == 0.50],
            "Fixed 75%": [r for r in rows if r["pruning_rate"] == 0.75],
            "BudgetVLM": budget_rows,
            "Oracle adaptive": oracle_rows,
        }
    )
    comparison.to_csv(project_path("results", "metrics", "smoke_comparison.csv"), index=False)
    write_json(
        project_path("results", "metrics", "smoke_status.json"),
        {
            "phase": "smoke_complete",
            "pruning_kind": "simulated_token_prune_fixed_frames",
            "n_videos": args.n_videos,
            "backend": "mock",
            "note": "Real runs use vLLM video_pruning_rate + VidCom2 with fixed frames.",
        },
    )
    log.info("\n%s", comparison.to_string(index=False))
    log.info("Smoke pipeline OK.")


if __name__ == "__main__":
    main()
