# BudgetVLM — Frozen Scope

**Do not change these mid-project.**

## Research question

> Can content-adaptive video token pruning reduce VLM inference cost while
> preserving accuracy better than fixed pruning rates?

## Locked stack

| Component | Choice |
|-----------|--------|
| Model | `Qwen/Qwen3-VL-4B-Instruct` |
| Serving | vLLM (`>=0.11.0`) |
| Dataset | Video-MME (`lmms-lab/Video-MME`) |
| Fixed pruning rates | 0%, 25%, 50%, 75% |
| Adaptive policy | choose 25% / 50% / 75% from video complexity |

## Pruning definition (Phase 2–5)

Visual-token budget is controlled by **temporal frame retention** relative to a
fixed baseline frame budget (`baseline_num_frames`):

```text
retained_frames = ceil(baseline_num_frames * (1 - pruning_rate))
```

This maps cleanly to measurable visual-token counts via the Qwen3-VL processor
(`grid_thw`). Spatial resolution (`max_pixels`) stays fixed across rates so the
only variable is the token budget.

## Critical path

```text
1. Qwen3-VL + vLLM works
2. Video-MME evaluation works
3. Run 0/25/50/75% pruning
4. Per-video pruning tolerance (oracle)
5. Video complexity scores
6. Complexity ↔ tolerance correlation
7. BudgetVLM threshold policy
8. Evaluate + compare
```

**Paper writing starts only after tasks 1–8 work.**
