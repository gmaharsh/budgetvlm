# Pipeline status

Last updated by the agent during Phase 1–3 setup.

## Frozen scope

See `SCOPE.md`. Research question and stack are locked.

## What works locally (this Mac, no CUDA)

| Step | Status |
|------|--------|
| Unit tests | PASS (`pytest` 6/6) |
| Synthetic video load | PASS |
| Mock inference 0/25/50/75% | PASS |
| Accuracy + TTFT + visual tokens | PASS |
| Per-video max safe pruning | PASS |
| Complexity estimator | PASS |
| BudgetVLM + Oracle from cache | PASS |
| Comparison table + plots | PASS (`results/figures/smoke_*.png`) |

Smoke on 20 synthetic videos (mock backend):

| Policy | Accuracy | Avg pruning | Avg visual tokens |
|--------|---------:|------------:|------------------:|
| No pruning | 1.00 | 0.00 | 2048 |
| Fixed 25% | 1.00 | 0.25 | 1536 |
| Fixed 50% | 1.00 | 0.50 | 1024 |
| Fixed 75% | 0.45 | 0.75 | 512 |
| **BudgetVLM** | **1.00** | **0.61** | **794** |
| Oracle adaptive | 1.00 | 0.61 | 794 |

This validates the *pipeline logic*. It is **not** a paper result.

## Blocked: real Qwen3-VL + vLLM

This machine is Apple M5 Pro (Metal only). **vLLM requires NVIDIA CUDA.**

To unblock Phase 2 for real:

1. Spin up a CUDA host (RunPod A100/L40S, AWS `g5`/`p4`, etc.)
2. On that host:

```bash
git clone <this-repo> && cd budgetvlm
bash scripts/run_gpu_pipeline.sh
```

Or manually:

```bash
pip install -r requirements-gpu.txt
python -m src.download_videomme --with-videos-chunk1
python -m src.run_one_example --backend vllm --pruning-rate 0.0
python -m src.run_fixed_pruning --backend vllm --limit 20
```

## Next actions (in order)

1. Get a CUDA GPU (user decision: RunPod / AWS / campus cluster)
2. Run one real Video-MME example through Qwen3-VL + vLLM
3. Fixed 0/25/50/75% on 20 videos
4. Tolerance → complexity correlation on real data
5. Freeze BudgetVLM thresholds on a calibration split
6. Only then expand N and write the paper
