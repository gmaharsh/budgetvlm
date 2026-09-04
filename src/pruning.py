"""Pruning helpers.

Real vLLM path: engine-level ``video_pruning_rate`` (token prune after vision encoder).
Frame count stays fixed at ``baseline_num_frames``.
"""
from __future__ import annotations

import math


PRUNING_RATES = (0.0, 0.25, 0.5, 0.75)


def retained_token_estimate(pre_prune_tokens: int, pruning_rate: float) -> int:
    """Estimated retained visual tokens after engine pruning (label as estimate)."""
    if pre_prune_tokens <= 0:
        return 0
    if pruning_rate <= 0:
        return int(pre_prune_tokens)
    return max(1, int(math.floor(pre_prune_tokens * (1.0 - pruning_rate))))


def rate_tag(pruning_rate: float) -> str:
    return f"p{int(round(pruning_rate * 100)):02d}"


# Kept only for mock / legacy smoke paths that still simulate cost via frames.
def retained_frames(baseline_num_frames: int, pruning_rate: float) -> int:
    if not 0.0 <= pruning_rate < 1.0:
        if pruning_rate == 1.0:
            return 1
        raise ValueError(f"pruning_rate must be in [0, 1], got {pruning_rate}")
    n = max(1, int(math.ceil(baseline_num_frames * (1.0 - pruning_rate))))
    return min(n, baseline_num_frames)
