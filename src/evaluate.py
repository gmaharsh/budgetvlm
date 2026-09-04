"""Accuracy + system metric aggregation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


def accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("correct")) / len(rows)


def summarize_by_pruning(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = (
        df.groupby("pruning_rate", as_index=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_latency_sec=("latency_sec", "mean"),
            avg_ttft_sec=("ttft_sec", "mean"),
            avg_visual_tokens=("visual_tokens", "mean"),
            avg_frames=("n_frames", "mean"),
        )
        .sort_values("pruning_rate")
    )
    return g


def summarize_policy(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy": name,
        "n": len(rows),
        "accuracy": accuracy(rows),
        "avg_pruning": float(sum(r["pruning_rate"] for r in rows) / len(rows)) if rows else 0.0,
        "avg_ttft_sec": float(sum(r["ttft_sec"] for r in rows) / len(rows)) if rows else 0.0,
        "avg_latency_sec": float(sum(r["latency_sec"] for r in rows) / len(rows)) if rows else 0.0,
        "avg_visual_tokens": float(
            sum((r.get("visual_tokens") or 0) for r in rows) / len(rows)
        )
        if rows
        else 0.0,
    }
