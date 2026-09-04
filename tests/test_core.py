"""Unit tests for pruning, prompts, policy, complexity (no GPU)."""
from __future__ import annotations

import numpy as np

from src.complexity import frame_difference_complexity, scene_change_score
from src.policy import (
    adaptive_pruning_rate,
    max_observed_correct_rate,
    max_safe_pruning_rate,
)
from src.prompts import extract_letter, is_correct
from src.pruning import retained_frames, retained_token_estimate
from src.s3_sync import parse_s3_uri, resolve_s3_uri


def test_retained_frames_legacy():
    assert retained_frames(32, 0.0) == 32
    assert retained_frames(32, 0.75) == 8


def test_retained_token_estimate():
    assert retained_token_estimate(1000, 0.0) == 1000
    assert retained_token_estimate(1000, 0.25) == 750
    assert retained_token_estimate(1000, 0.75) == 250


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


def test_max_observed_allows_nonmonotonic():
    # 50% wrong but 75% correct → observed max is 75%, strict safe stops at 25%
    outcomes = {0.0: True, 0.25: True, 0.5: False, 0.75: True}
    assert max_safe_pruning_rate(outcomes) == 0.25
    assert max_observed_correct_rate(outcomes) == 0.75


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


def test_parse_s3_uri():
    assert parse_s3_uri("s3://my-bucket/budgetvlm") == ("my-bucket", "budgetvlm")
    assert parse_s3_uri("s3://my-bucket/") == ("my-bucket", "")


def test_resolve_video_path_ignores_empty_and_uses_youtube_id(tmp_path):
    from src.dataset_videomme import resolve_video_path

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    mp4 = video_dir / "ytABC123.mp4"
    mp4.write_bytes(b"fake")
    # Path("") must never win (it is cwd ".")
    assert resolve_video_path(video_dir, video_id="001", youtube_id="ytABC123") == str(mp4)
    assert resolve_video_path(video_dir, video_id="missing", youtube_id="") == ""
    nested = video_dir / "data" / "nestedVid.mp4"
    nested.parent.mkdir()
    nested.write_bytes(b"fake")
    assert resolve_video_path(video_dir, video_id="x", youtube_id="nestedVid").endswith(
        "nestedVid.mp4"
    )


def test_resolve_s3_uri(monkeypatch):
    monkeypatch.delenv("BUDGETVLM_S3_URI", raising=False)
    monkeypatch.delenv("BUDGETVLM_S3_BUCKET", raising=False)
    monkeypatch.delenv("BUDGETVLM_AWS_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("AWS_ACCOUNT_ID", raising=False)
    assert resolve_s3_uri({"s3": {"bucket": "b", "prefix": "results"}}) == "s3://b/results"
    monkeypatch.setenv("BUDGETVLM_S3_URI", "s3://env-bucket/runs")
    assert resolve_s3_uri({"s3": {"bucket": "ignored"}}) == "s3://env-bucket/runs"


def test_default_bucket_name():
    from src.s3_sync import default_bucket_name

    assert default_bucket_name("704052814573") == "budgetvlm-704052814573"
