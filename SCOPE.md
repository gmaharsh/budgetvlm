# BudgetVLM — Frozen Scope

**Do not change these mid-project.**

## Research question

> Can content-adaptive video token pruning reduce VLM inference cost while
> preserving accuracy better than fixed pruning rates?

## Locked stack

| Component | Choice |
|-----------|--------|
| Model | `Qwen/Qwen3-VL-4B-Instruct` |
| Serving | vLLM (`video_pruning_rate` engine config) |
| Pruning method | **VidCom2** (primary); EVS optional ablation |
| Dataset | Video-MME (`lmms-lab/Video-MME`) |
| Fixed pruning rates | 0%, 25%, 50%, 75% |
| Adaptive policy | choose 25% / 50% / 75% from video complexity |

## Pruning definition (LOCKED — vLLM token pruning)

**Not** frame dropping. Frames and resolution stay fixed.

```text
Same video
Same N frames (baseline_num_frames)
Same max_pixels
Same prompt
        ↓
Vision encoder (full frames)
        ↓
VidCom2 / EVS token pruning @ video_pruning_rate
        ↓
LLM prefill on retained visual tokens
```

Engine config (rate is **engine-level**, not per-request):

```python
LLM(
  model="Qwen/Qwen3-VL-4B-Instruct",
  video_pruning_rate=0.50,      # None/0 → no pruning
  video_pruning_method="vidcom2",
)
```

Stage A (this paper): one engine per rate → cache results → simulate BudgetVLM.
Stage B (optional later): multi-engine router or true per-request budgets.

## Metrics (honest)

| Metric | Status |
|--------|--------|
| End-to-end generation latency | **Primary systems metric** |
| TTFT | **Not reported** until streaming serve is instrumented |
| `visual_tokens_pre` | From processor `grid_thw` (pre-prune) |
| `visual_tokens_retained_est` | `pre * (1 - rate)` — **estimated** unless engine-instrumented |

## Critical path

```text
1. Qwen3-VL + vLLM + VidCom2 works
2. Video-MME evaluation works
3. Run 0/25/50/75% token pruning (fixed frames)
4. Per-video pruning tolerance (oracle)
5. Video complexity scores
6. Complexity ↔ tolerance correlation
7. Calibrate T1/T2 on held-out split; freeze
8. Evaluate BudgetVLM on test split
```

**Paper writing starts only after tasks 1–8 work.**
**Do not launch large GPU runs until this pruning definition is implemented.**
