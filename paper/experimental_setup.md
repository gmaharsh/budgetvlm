# Experimental Setup

## Model and serving

We evaluate [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) served with [vLLM](https://github.com/vllm-project/vllm). Visual-token pruning uses engine configuration `video_pruning_rate` with method **VidCom2** (`video_pruning_method=vidcom2`). When a build lacks VidCom2, we fall back to EVS and record the method used; the primary paper claims target VidCom2.

Generation uses greedy decoding (`temperature=0`) with a short max new-token budget appropriate for letter answers (`max_tokens=16` in our default config). Video preprocessing keeps `do_resize=false` where applicable so spatial token grids are not crushed by an extra host-side resize before the engine.

## Dataset

We use **Video-MME** ([Fu et al.](https://github.com/BradyFU/Video-MME); Hugging Face `lmms-lab/Video-MME`). Primary runs disable subtitles. We report results on a progressive subset schedule:

| Stage | Videos (approx.) | Purpose |
|-------|------------------|---------|
| Sanity | 1–5 | Engine + path correctness |
| Pilot | ~20 | Full rate matrix + policy wiring |
| Main | `[TODO: N after scale-up, e.g. 100–200]` | Paper numbers |
| Full | 900 videos / 2700 QA (optional) | Completeness, if compute allows |

Exact video IDs and split membership for calibration vs.\ test are recorded in run artifacts (`calibrated_thresholds.json`) for reproducibility.

## Video input (matched across rates)

For every pruning rate we use the **same** sampling settings:

| Knob | Default |
|------|---------|
| Frames $N$ | 32 (`baseline_num_frames`) |
| Max pixels $P$ | 360360 |
| Pruning rates | 0%, 25%, 50%, 75% |
| Pruning kind | vLLM video token prune (not frame drop) |

Only the engine pruning rate changes across conditions.

## Complexity and policy hyperparameters

| Knob | Default | Notes |
|------|---------|--------|
| Complexity frames | 8 | Uniform sample, resize for cheap scoring |
| Motion / scene mix | motion-only primary; $\alpha{=}0.7$, $\beta{=}0.3$ if scene enabled | |
| Scene threshold | 0.35 | Only if scene term enabled |
| Policy thresholds $T_1, T_2$ | Calibrated then frozen | Defaults `0.15` / `0.40` are smoke placeholders only |
| Calibration fraction | 0.3 | Video-level random split, seed 42 |
| $T_1$ grid | $\{0.05,0.10,0.15,0.20,0.25\}$ | Calibration only |
| $T_2$ grid | $\{0.30,\ldots,0.55\}$ | Require $T_1 < T_2$ |

Reported BudgetVLM numbers always use **frozen** $(T_1,T_2)$ chosen on the calibration split.

## Hardware and software

| Item | Value |
|------|-------|
| GPU | `[TODO: e.g. NVIDIA A40 48GB]` |
| CUDA / driver | `[TODO]` |
| PyTorch | `[TODO: e.g. 2.x + cu129]` |
| vLLM | `[TODO: e.g. 0.28.0+cu129]` |
| Host notes | `[TODO: RunPod / local; disk; concurrent engines or sequential reload]` |

Fill this table from the machine that produces the paper’s result files; do not copy smoke/Mac mock settings.

## Evaluation protocol

1. **Fixed matrix.** For each rate $r$, load one vLLM engine with that `video_pruning_rate`, run all (video, question) pairs, and log correctness, end-to-end latency, and token estimates.
2. **Tolerance / oracle.** Aggregate per-video all-question correctness; compute strict safe rate and max observed correct rate.
3. **Complexity.** Score each video with the estimator above.
4. **Calibrate.** Split videos; grid-search $(T_1,T_2)$ on calibration; freeze.
5. **Compare.** On the test split (and/or full set with frozen thresholds, as specified in the Results section), compare fixed rates, BudgetVLM, and oracle on accuracy vs.\ average pruning / average latency.

All primary metrics and plots are generated from logged prediction JSONL files so that adaptive policies can be re-simulated without re-running the VLM when only thresholds change.

## What we will not claim from this setup alone

- Streaming TTFT improvements (not instrumented).
- Exact retained-token counts (estimated unless engine-instrumented).
- Online multi-rate serving cost without a Stage B router measurement.
