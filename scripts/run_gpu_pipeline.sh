#!/usr/bin/env bash
# Run on a CUDA machine (RunPod / AWS GPU / local NVIDIA).
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt

echo "=== GPU check ==="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required for vLLM"
print("cuda:", torch.cuda.get_device_name(0))
PY

echo "=== Download Video-MME annotations + first video chunk ==="
python -m src.download_videomme --with-videos-chunk1

echo "=== One example @ 0% pruning ==="
python -m src.run_one_example --backend vllm --pruning-rate 0.0

echo "=== Fixed matrix on 20 videos ==="
python -m src.run_fixed_pruning --backend vllm --limit 20 --tag fixed20

echo "=== Tolerance + complexity + BudgetVLM ==="
python -m src.analyze_tolerance --predictions results/predictions/fixed20_vllm.jsonl \
  --out results/metrics/tolerance.jsonl
python -m src.compute_complexity --limit 20 --out results/metrics/complexity.jsonl
python -m src.run_budgetvlm \
  --predictions results/predictions/fixed20_vllm.jsonl \
  --complexity results/metrics/complexity.jsonl \
  --tolerance results/metrics/tolerance.jsonl
python -m src.make_plots --prefix "" 2>/dev/null || true

echo "Done. Inspect results/metrics/comparison.csv"
