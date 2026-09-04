#!/usr/bin/env bash
# Run on a CUDA machine (RunPod / AWS GPU / local NVIDIA).
# Optionally sync results to S3 when BUDGETVLM_S3_URI is set.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt

RUN_ID="${BUDGETVLM_RUN_ID:-runpod_$(date -u +%Y%m%dT%H%M%SZ)}"
echo "=== run_id=${RUN_ID} ==="

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
python -m src.make_plots --prefix smoke 2>/dev/null || true

if [[ -n "${BUDGETVLM_S3_URI:-}" ]]; then
  echo "=== Upload results to ${BUDGETVLM_S3_URI}/${RUN_ID} ==="
  python -m src.s3_sync upload --run-id "${RUN_ID}"
else
  echo "=== Skipping S3 upload (set BUDGETVLM_S3_URI to enable) ==="
fi

echo "Done. Inspect results/metrics/comparison.csv"
echo "run_id=${RUN_ID}"
