"""Phase 6: per-video max safe pruning rate (oracle material)."""
from __future__ import annotations

import argparse

from .evaluate import per_video_accuracy
from .policy import (
    aggregate_video_accuracy,
    aggregate_video_correctness,
    max_observed_correct_rate,
    max_safe_pruning_rate,
)
from .utils import get_logger, load_config, project_path, read_jsonl, write_jsonl

log = get_logger("tolerance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pred_path = (
        project_path(args.predictions)
        if args.predictions
        else project_path("results", "predictions", "smoke_fixed.jsonl")
    )
    rows = read_jsonl(pred_path)
    if not rows:
        raise SystemExit(f"No predictions at {pred_path}")

    rates = sorted({float(r["pruning_rate"]) for r in rows})
    by_all = aggregate_video_correctness(rows)
    by_acc = aggregate_video_accuracy(rows)
    out_rows = []
    for vid in sorted(by_all.keys()):
        outcomes = by_all[vid]
        accs = by_acc.get(vid, {})
        out_rows.append(
            {
                "video_id": vid,
                "max_safe_pruning_rate": max_safe_pruning_rate(outcomes, rates),
                "max_observed_correct_rate": max_observed_correct_rate(outcomes, rates),
                **{f"all_correct_p{int(r * 100):02d}": outcomes.get(r, False) for r in rates},
                **{f"acc_p{int(r * 100):02d}": accs.get(r, 0.0) for r in rates},
            }
        )

    out = (
        project_path(args.out)
        if args.out
        else project_path("results", "metrics", "tolerance.jsonl")
    )
    write_jsonl(out, out_rows)
    per_video_accuracy(rows).to_csv(
        project_path("results", "metrics", "per_video_accuracy.csv"), index=False
    )
    n_diff = len({r["max_safe_pruning_rate"] for r in out_rows})
    log.info("Wrote %s (%d videos, %d distinct strict-safe rates)", out, len(out_rows), n_diff)
    for r in out_rows[:10]:
        log.info(
            "%s -> strict_safe=%.2f observed_max=%.2f",
            r["video_id"],
            r["max_safe_pruning_rate"],
            r["max_observed_correct_rate"],
        )


if __name__ == "__main__":
    main()
