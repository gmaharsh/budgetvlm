"""Phase 10: evaluate BudgetVLM (simulate from cached fixed-pruning results)."""
from __future__ import annotations

import argparse

from .compare_policies import build_comparison
from .policy import adaptive_pruning_rate
from .utils import get_logger, load_config, project_path, read_jsonl, write_jsonl

log = get_logger("budgetvlm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--predictions", default=None, help="fixed-pruning jsonl")
    ap.add_argument("--complexity", default=None, help="complexity jsonl")
    ap.add_argument("--tolerance", default=None, help="tolerance jsonl for oracle")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--t2", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    pred_path = (
        project_path(args.predictions)
        if args.predictions
        else project_path("results", "predictions", "smoke_fixed.jsonl")
    )
    cx_path = (
        project_path(args.complexity)
        if args.complexity
        else project_path("results", "metrics", "smoke_complexity.jsonl")
    )
    tol_path = (
        project_path(args.tolerance)
        if args.tolerance
        else project_path("results", "metrics", "smoke_tolerance.jsonl")
    )

    rows = read_jsonl(pred_path)
    cx = {r["video_id"]: float(r["complexity"]) for r in read_jsonl(cx_path)}
    tol = {r["video_id"]: float(r["max_safe_pruning_rate"]) for r in read_jsonl(tol_path)}
    if not rows or not cx:
        raise SystemExit(
            "Need predictions + complexity files. Run smoke or fixed+complexity first."
        )

    t1 = float(args.t1 if args.t1 is not None else cfg["complexity"]["t1"])
    t2 = float(args.t2 if args.t2 is not None else cfg["complexity"]["t2"])
    log.info("Frozen thresholds t1=%.4f t2=%.4f", t1, t2)

    index = {(r["video_id"], r["question_id"], float(r["pruning_rate"])): r for r in rows}
    questions = {(r["video_id"], r["question_id"]) for r in rows}

    budget_rows = []
    oracle_rows = []
    for vid, qid in sorted(questions):
        if vid not in cx:
            continue
        rate = adaptive_pruning_rate(cx[vid], t1, t2)
        key = (vid, qid, float(rate))
        if key not in index:
            log.warning("missing cached result %s", key)
            continue
        budget_rows.append({**index[key], "policy": "BudgetVLM", "complexity": cx[vid]})
        oracle_rate = tol.get(vid, 0.0)
        okey = (vid, qid, float(oracle_rate))
        if okey in index:
            oracle_rows.append({**index[okey], "policy": "Oracle", "complexity": cx[vid]})

    if not budget_rows:
        log.warning(
            "BudgetVLM selected 0 rows — complexity video_ids likely do not match "
            "predictions (e.g. synthetic complexity vs Video-MME). "
            "Re-run: python -m src.compute_complexity --backend vllm "
            "--predictions %s",
            pred_path,
        )

    write_jsonl(project_path("results", "predictions", "budgetvlm.jsonl"), budget_rows)
    write_jsonl(project_path("results", "predictions", "oracle.jsonl"), oracle_rows)

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
    out_csv = project_path("results", "metrics", "comparison.csv")
    comparison.to_csv(out_csv, index=False)
    log.info("\n%s", comparison.to_string(index=False))
    log.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
