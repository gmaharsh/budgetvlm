"""BudgetVLM threshold policy + oracle safe-rate selection."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def adaptive_pruning_rate(complexity: float, t1: float, t2: float) -> float:
    """Map complexity → {0.75, 0.50, 0.25}. Never returns 0% (always prune some)."""
    if complexity < t1:
        return 0.75
    if complexity < t2:
        return 0.50
    return 0.25


def max_safe_pruning_rate(
    outcomes: dict[float, bool],
    rates: Iterable[float] = (0.0, 0.25, 0.5, 0.75),
) -> float:
    """Largest pruning rate that stays correct, given baseline must be correct.

    outcomes: pruning_rate -> whether the (aggregated) prediction was correct.
    If baseline (0%) is wrong, safe rate is defined as 0.0 (no headroom claimed).
    """
    rates = sorted(rates)
    if not outcomes.get(0.0, False):
        return 0.0
    best = 0.0
    for r in rates:
        if outcomes.get(r, False):
            best = r
        else:
            break
    return best


def aggregate_video_correctness(rows: list[dict]) -> dict[str, dict[float, bool]]:
    """video_id -> {rate: all questions correct at that rate}."""
    # rows need: video_id, pruning_rate, correct
    bucket: dict[str, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        bucket[r["video_id"]][float(r["pruning_rate"])].append(bool(r["correct"]))
    out: dict[str, dict[float, bool]] = {}
    for vid, by_rate in bucket.items():
        out[vid] = {rate: all(vals) and len(vals) > 0 for rate, vals in by_rate.items()}
    return out
