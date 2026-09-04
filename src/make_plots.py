"""Phase 12: correlation + ablation plots (no paper prose)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(_ROOT / ".mplconfig")
os.environ["MPLBACKEND"] = "Agg"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, get_logger, project_path, read_jsonl

log = get_logger("plots")


def plot_complexity_vs_safe(complexity_path, tolerance_path, out_path) -> None:
    cx = {r["video_id"]: r for r in read_jsonl(complexity_path)}
    tol = {r["video_id"]: r for r in read_jsonl(tolerance_path)}
    xs, ys = [], []
    for vid in cx:
        if vid not in tol:
            continue
        xs.append(cx[vid]["complexity"])
        ys.append(tol[vid]["max_safe_pruning_rate"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, alpha=0.7)
    ax.set_xlabel("complexity")
    ax.set_ylabel("max safe pruning rate")
    ax.set_title("Complexity vs safe pruning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_accuracy_vs_pruning(by_rate_csv, out_path) -> None:
    df = pd.read_csv(by_rate_csv)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["pruning_rate"], df["accuracy"], marker="o")
    ax.set_xlabel("pruning rate")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs pruning")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_accuracy_vs_cost(by_rate_csv, out_path) -> None:
    df = pd.read_csv(by_rate_csv)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["avg_ttft_sec"], df["accuracy"], marker="o")
    for _, r in df.iterrows():
        ax.annotate(f"{int(r['pruning_rate']*100)}%", (r["avg_ttft_sec"], r["accuracy"]))
    ax.set_xlabel("avg TTFT (s)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs TTFT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="smoke", help="smoke | fixed prefix for metrics files")
    args = ap.parse_args()
    fig_dir = ensure_dir(project_path("results", "figures"))
    metrics = project_path("results", "metrics")

    cx = metrics / f"{args.prefix}_complexity.jsonl"
    tol = metrics / f"{args.prefix}_tolerance.jsonl"
    by_rate = metrics / f"{args.prefix}_by_rate.csv"

    if cx.exists() and tol.exists():
        plot_complexity_vs_safe(cx, tol, fig_dir / f"{args.prefix}_complexity_vs_safe.png")
    if by_rate.exists():
        plot_accuracy_vs_pruning(by_rate, fig_dir / f"{args.prefix}_acc_vs_pruning.png")
        plot_accuracy_vs_cost(by_rate, fig_dir / f"{args.prefix}_acc_vs_ttft.png")


if __name__ == "__main__":
    main()
