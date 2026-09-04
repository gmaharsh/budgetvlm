"""Calibrate BudgetVLM thresholds T1/T2 on a held-out split, then freeze."""
from __future__ import annotations

import argparse
import itertools
from typing import Any

from .policy import adaptive_pruning_rate
from .utils import get_logger, load_config, project_path, read_jsonl, set_seed, write_json

log = get_logger("calibrate")


def _split_video_ids(ids: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    import random

    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n_cal = max(1, int(round(len(ids) * frac)))
    if n_cal >= len(ids):
        n_cal = max(1, len(ids) - 1)
    return ids[:n_cal], ids[n_cal:]


def score_thresholds(
    cal_videos: list[str],
    complexity: dict[str, float],
    index: dict[tuple[str, str, float], dict],
    questions: list[tuple[str, str]],
    t1: float,
    t2: float,
) -> dict[str, float]:
    rows = []
    for vid, qid in questions:
        if vid not in cal_videos:
            continue
        rate = adaptive_pruning_rate(complexity[vid], t1, t2)
        key = (vid, qid, float(rate))
        if key in index:
            rows.append(index[key])
    if not rows:
        return {"accuracy": 0.0, "avg_pruning": 0.0, "avg_latency_sec": 0.0, "n": 0}
    acc = sum(1 for r in rows if r["correct"]) / len(rows)
    return {
        "accuracy": acc,
        "avg_pruning": sum(r["pruning_rate"] for r in rows) / len(rows),
        "avg_latency_sec": sum(r["latency_sec"] for r in rows) / len(rows),
        "n": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--complexity", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    rows = read_jsonl(project_path(args.predictions))
    cx = {r["video_id"]: float(r["complexity"]) for r in read_jsonl(project_path(args.complexity))}
    if not rows or not cx:
        raise SystemExit("Need predictions + complexity")

    video_ids = sorted({r["video_id"] for r in rows if r["video_id"] in cx})
    frac = float(cfg["complexity"].get("calibration_fraction", 0.3))
    cal_ids, test_ids = _split_video_ids(video_ids, frac, cfg["seed"])
    log.info("Calibration videos=%d test videos=%d", len(cal_ids), len(test_ids))

    index = {(r["video_id"], r["question_id"], float(r["pruning_rate"])): r for r in rows}
    questions = sorted({(r["video_id"], r["question_id"]) for r in rows})

    # Grid search on calibration only
    t1_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
    t2_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    best: dict[str, Any] | None = None
    for t1, t2 in itertools.product(t1_grid, t2_grid):
        if not (t1 < t2):
            continue
        s = score_thresholds(cal_ids, cx, index, questions, t1, t2)
        # Prefer high accuracy, then higher avg pruning (more savings)
        key = (s["accuracy"], s["avg_pruning"], -s["avg_latency_sec"])
        if best is None or key > best["key"]:
            best = {"t1": t1, "t2": t2, "cal": s, "key": key}

    assert best is not None
    test_s = score_thresholds(test_ids, cx, index, questions, best["t1"], best["t2"])
    out = {
        "t1": best["t1"],
        "t2": best["t2"],
        "calibration_fraction": frac,
        "calibration_video_ids": cal_ids,
        "test_video_ids": test_ids,
        "calibration_metrics": best["cal"],
        "test_metrics_with_frozen_thresholds": test_s,
        "note": "Freeze t1/t2 into configs/default.yaml before final eval reporting.",
    }
    out_path = project_path(args.out) if args.out else project_path(
        "results", "metrics", "calibrated_thresholds.json"
    )
    write_json(out_path, out)
    log.info("Selected t1=%.3f t2=%.3f | cal_acc=%.3f test_acc=%.3f", best["t1"], best["t2"], best["cal"]["accuracy"], test_s["accuracy"])
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
