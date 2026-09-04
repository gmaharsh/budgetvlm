# BudgetVLM

**Content-adaptive video token pruning for efficient VLM inference.**

> Can content-adaptive video token pruning reduce VLM inference cost while
> preserving accuracy better than fixed pruning rates?

This repository contains the experiment pipeline for BudgetVLM: fixed visual-token
pruning baselines (0% / 25% / 50% / 75%), a frame-difference video-complexity
estimator, an adaptive BudgetVLM policy, and an oracle upper bound on Video-MME
with Qwen3-VL + vLLM.

See [`SCOPE.md`](SCOPE.md) for the frozen research question and stack.
See [`STATUS.md`](STATUS.md) for current pipeline status.

## Stack

| Component | Choice |
|-----------|--------|
| Model | [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Serving | [vLLM](https://github.com/vllm-project/vllm) (`>=0.11`) |
| Dataset | [Video-MME](https://github.com/BradyFU/Video-MME) (`lmms-lab/Video-MME`) |
| Fixed rates | 0%, 25%, 50%, 75% |
| Adaptive policy | choose 25% / 50% / 75% from video complexity |

## Quick start (offline smoke, no GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.run_smoke --n-videos 20
```

## Full experiment path (CUDA + vLLM)

```bash
pip install -r requirements-gpu.txt
bash scripts/run_gpu_pipeline.sh
```

Or step by step:

```bash
python -m src.download_videomme --with-videos-chunk1
python -m src.run_one_example --backend vllm --pruning-rate 0.0
python -m src.run_fixed_pruning --backend vllm --limit 20 --tag fixed20
python -m src.analyze_tolerance --predictions results/predictions/fixed20_vllm.jsonl
python -m src.compute_complexity --limit 20
python -m src.run_budgetvlm \
  --predictions results/predictions/fixed20_vllm.jsonl \
  --complexity results/metrics/complexity.jsonl \
  --tolerance results/metrics/tolerance.jsonl
python -m src.make_plots --prefix smoke
```

## Repository layout

```text
SCOPE.md                 # frozen RQ + stack
STATUS.md                # pipeline progress
configs/default.yaml     # experiment knobs
src/                     # inference, eval, complexity, BudgetVLM
scripts/                 # CUDA host helpers
tests/                   # unit tests
results/                 # metrics / predictions / figures (generated)
data/                    # Video-MME cache (not committed)
```

## Citation

If you use this code, please cite the accompanying workshop paper (to appear)
and this repository:

```bibtex
@misc{budgetvlm2026,
  title        = {BudgetVLM: Content-Adaptive Video Token Pruning for Efficient VLM Inference},
  author       = {Gheewala, Maharsh},
  year         = {2026},
  howpublished = {\url{https://github.com/gmaharsh/budgetvlm}},
  note         = {Code repository}
}
```

## License

Apache-2.0 (code). Dataset and model weights are subject to their original licenses.
