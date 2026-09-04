"""Phase 4–5: fixed pruning matrix using one vLLM engine per rate.

For vLLM, each pruning rate runs in a **fresh subprocess** so EngineCore is
fully destroyed between rates. In-process teardown has hung on RunPod when
reloading with video_pruning_rate > 0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .backends import get_backend
from .dataset_videomme import load_videomme_annotations, make_synthetic_dataset
from .evaluate import summarize_by_pruning
from .infer import run_one
from .pruning import rate_tag
from .utils import append_jsonl, ensure_dir, get_logger, load_config, project_path, read_jsonl, set_seed

log = get_logger("fixed")


def _run_rate(
    *,
    backend_name: str,
    cfg: dict,
    examples: list,
    rate: float,
    baseline_frames: int,
    max_pixels,
    out: str,
) -> list[dict]:
    log.info("=== rate=%.2f (%s) | fixed_frames=%d ===", rate, rate_tag(rate), baseline_frames)
    backend = get_backend(backend_name, cfg, pruning_rate=rate)
    rows: list[dict] = []
    try:
        warm = examples[0]
        backend.warmup(
            video_path=warm.video_path,
            question=warm.question,
            options=warm.options,
            pruning_rate=rate,
            baseline_num_frames=baseline_frames,
            max_pixels=max_pixels,
            n=int(cfg.get("serving", {}).get("warmup_generates", 2)),
        )
        for ex in examples:
            log.info("%s q=%s rate=%.2f", ex.video_id, ex.question_id, rate)
            row = run_one(backend, ex, rate, baseline_frames, max_pixels)
            append_jsonl(str(out), row)
            rows.append(row)
    finally:
        backend.close()
        # Give CUDA / EngineCore a moment to release before next rate/process.
        time.sleep(3)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max unique videos")
    ap.add_argument("--rates", default="0,0.25,0.5,0.75")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tag", default="fixed")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing predictions jsonl instead of truncating",
    )
    ap.add_argument(
        "--single-process",
        action="store_true",
        help="Run rates in this process (default for mock; vLLM uses subprocesses)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    backend_name = args.backend or cfg["serving"]["backend"]
    limit = args.limit if args.limit is not None else int(cfg["dataset"]["default_limit"])
    rates = [float(x) for x in args.rates.split(",")]

    if args.synthetic or backend_name == "mock":
        examples = make_synthetic_dataset(n_videos=limit, seed=cfg["seed"])
    else:
        examples = load_videomme_annotations(
            root=project_path(*(cfg["dataset"]["root"].split("/"))),
            limit=limit,
            require_video=True,
        )
        missing = [e for e in examples if not e.video_path]
        if not examples or missing:
            raise SystemExit(
                f"Need local videos under data/videomme/videos (got {len(examples)} examples, "
                f"{len(missing)} missing). "
                "Run: python -m src.download_videomme --with-videos-chunk1"
            )

    out = project_path("results", "predictions", f"{args.tag}_{backend_name}.jsonl")
    ensure_dir(out.parent)
    if not args.append:
        open(out, "w").close()

    baseline_frames = int(cfg["video"]["baseline_num_frames"])
    max_pixels = cfg["video"].get("max_pixels")

    # vLLM: one OS process per rate (avoids hung EngineCore on reload)
    use_subprocess = backend_name == "vllm" and not args.single_process and len(rates) > 1
    if use_subprocess:
        log.info("Using one subprocess per pruning rate (clean EngineCore teardown)")
        for rate in rates:
            cmd = [
                sys.executable,
                "-m",
                "src.run_fixed_pruning",
                "--backend",
                "vllm",
                "--limit",
                str(limit),
                "--tag",
                args.tag,
                "--rates",
                str(rate),
                "--append",
                "--single-process",
            ]
            if args.config:
                cmd.extend(["--config", args.config])
            if args.synthetic:
                cmd.append("--synthetic")
            log.info("Launch subprocess: rate=%s", rate)
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise SystemExit(
                    f"Subprocess failed for rate={rate} with exit code {proc.returncode}. "
                    "Kill leftover EngineCore (`pkill -f EngineCore`) and retry that rate with "
                    f"`--rates {rate} --append --single-process`."
                )
            time.sleep(2)
        rows = read_jsonl(out)
    else:
        rows = []
        for rate in rates:
            rows.extend(
                _run_rate(
                    backend_name=backend_name,
                    cfg=cfg,
                    examples=examples,
                    rate=rate,
                    baseline_frames=baseline_frames,
                    max_pixels=max_pixels,
                    out=str(out),
                )
            )

    summary = summarize_by_pruning(rows)
    ensure_dir(project_path("results", "metrics"))
    summary_path = project_path("results", "metrics", f"{args.tag}_{backend_name}_by_rate.csv")
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s (%d rows)", out, len(rows))
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
