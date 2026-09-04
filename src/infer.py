"""Single-example and batch inference harness with metrics."""
from __future__ import annotations

from typing import Any

from .backends import Backend, InferResult
from .dataset_videomme import QAExample
from .prompts import is_correct
from .utils import append_jsonl, ensure_dir, get_logger

log = get_logger("infer")


def run_one(
    backend: Backend,
    example: QAExample,
    pruning_rate: float,
    baseline_num_frames: int,
    max_pixels: int | None,
) -> dict[str, Any]:
    if not example.video_path:
        raise FileNotFoundError(f"Missing video for {example.video_id}")
    result: InferResult = backend.infer_mcq(
        video_path=example.video_path,
        question=example.question,
        options=example.options,
        pruning_rate=pruning_rate,
        baseline_num_frames=baseline_num_frames,
        max_pixels=max_pixels,
        ground_truth=example.answer,
    )
    correct = is_correct(result.prediction, example.answer)
    row = {
        "video_id": example.video_id,
        "question_id": example.question_id,
        "question": example.question,
        "prediction": result.prediction,
        "ground_truth": example.answer,
        "correct": correct,
        "latency_sec": result.latency_sec,
        "ttft_sec": result.ttft_sec,
        "visual_tokens": result.visual_tokens,
        "pruning_rate": pruning_rate,
        "n_frames": result.n_frames,
        "video_duration_sec": result.video_duration_sec,
        "duration_category": example.duration,
        "domain": example.domain,
        "task_type": example.task_type,
        **{f"meta_{k}": v for k, v in result.meta.items()},
    }
    return row


def run_matrix(
    backend: Backend,
    examples: list[QAExample],
    rates: list[float],
    baseline_num_frames: int,
    max_pixels: int | None,
    out_jsonl: str,
) -> list[dict[str, Any]]:
    ensure_dir(out_jsonl.rsplit("/", 1)[0] if "/" in out_jsonl else ".")
    # fresh file
    open(out_jsonl, "w").close()
    rows: list[dict[str, Any]] = []
    total = len(examples) * len(rates)
    i = 0
    for ex in examples:
        for rate in rates:
            i += 1
            log.info(
                "[%d/%d] %s q=%s rate=%.2f",
                i,
                total,
                ex.video_id,
                ex.question_id,
                rate,
            )
            row = run_one(backend, ex, rate, baseline_num_frames, max_pixels)
            append_jsonl(out_jsonl, row)
            rows.append(row)
    return rows
