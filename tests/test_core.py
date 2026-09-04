"""Unit tests for pruning, prompts, policy, complexity (no GPU)."""
from __future__ import annotations

import numpy as np

from src.complexity import frame_difference_complexity, scene_change_score
from src.policy import adaptive_pruning_rate, max_safe_pruning_rate
from src.prompts import extract_letter, is_correct
from src.pruning import retained_frames


def test_retained_frames():
    assert retained_frames(32, 0.0) == 32
    assert retained_frames(32, 0.25) == 24
    assert retained_frames(32, 0.50) == 16
    assert retained_frames(32, 0.75) == 8


def test_extract_letter():
    assert extract_letter("B") == "B"
    assert extract_letter("The answer is C.") == "C"
    assert extract_letter("I think A is correct") == "A"


def test_is_correct():
    assert is_correct("A", "A")
    assert is_correct("Answer: B", "B")
    assert not is_correct("C", "A")


def test_max_safe_pruning():
    assert max_safe_pruning_rate({0.0: True, 0.25: True, 0.5: False, 0.75: False}) == 0.25
    assert max_safe_pruning_rate({0.0: True, 0.25: True, 0.5: True, 0.75: True}) == 0.75
    assert max_safe_pruning_rate({0.0: False, 0.25: True}) == 0.0


def test_adaptive_policy():
    assert adaptive_pruning_rate(0.05, 0.15, 0.40) == 0.75
    assert adaptive_pruning_rate(0.20, 0.15, 0.40) == 0.50
    assert adaptive_pruning_rate(0.80, 0.15, 0.40) == 0.25


def test_frame_diff_static_vs_motion():
    static = [np.zeros((32, 32, 3), dtype=np.uint8) + 10 for _ in range(4)]
    motion = []
    for i in range(4):
        f = np.zeros((32, 32, 3), dtype=np.uint8)
        f[:, :, :] = (i * 60) % 255
        motion.append(f)
    assert frame_difference_complexity(static) < frame_difference_complexity(motion)
    assert scene_change_score(motion, threshold=0.1) >= scene_change_score(static, threshold=0.1)
