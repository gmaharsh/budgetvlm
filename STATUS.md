# Pipeline status

## STOP — do not burn GPU on the old frame-drop definition

Pruning is now **vLLM `video_pruning_rate` + VidCom2** with **fixed frames**.
On RunPod: stop the old job if it was started before this fix, then:

```bash
cd /workspace/budgetvlm
git pull
bash scripts/run_gpu_pipeline.sh
```

## What changed (scientific)

| Before (wrong for paper title) | After |
|--------------------------------|-------|
| `nframes = 32 * (1-rate)` | `nframes = 32` always |
| Frame reduction | Engine `video_pruning_rate` + `vidcom2` |
| Fake TTFT = full latency | Report **e2e latency** only; `ttft_sec=null` |
| Fuzzy visual tokens | `visual_tokens_pre` + `visual_tokens_retained_est` |
| Oracle: monotonic only | + `max_observed_correct_rate` |
| Video tolerance: all-correct only | + per-video accuracy CSV |
| Fixed T1/T2 | `calibrate_thresholds.py` on held-out split |

## Local smoke

```bash
python -m pytest -q
python -m src.run_smoke --n-videos 20
```
