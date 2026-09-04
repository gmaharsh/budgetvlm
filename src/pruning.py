"""Pruning-rate → retained frame budget."""
from __future__ import annotations

import math


PRUNING_RATES = (0.0, 0.25, 0.5, 0.75)


def retained_frames(baseline_num_frames: int, pruning_rate: float) -> int:
    """Map pruning rate to number of frames kept.

    pruning_rate=0.0 → keep all baseline frames
    pruning_rate=0.75 → keep 25% of baseline frames
    """
    if not 0.0 <= pruning_rate < 1.0:
        if pruning_rate == 1.0:
            return 1
        raise ValueError(f"pruning_rate must be in [0, 1], got {pruning_rate}")
    n = max(1, int(math.ceil(baseline_num_frames * (1.0 - pruning_rate))))
    return min(n, baseline_num_frames)


def rate_tag(pruning_rate: float) -> str:
    return f"p{int(round(pruning_rate * 100)):02d}"
