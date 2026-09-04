#!/usr/bin/env bash
# Run on a CUDA machine (RunPod / AWS GPU / local NVIDIA).
# At start: ensure S3 bucket exists (create/update).
# At end: upload results/ to that bucket under run_id/.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
source .venv/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt

RUN_ID="${BUDGETVLM_RUN_ID:-runpod_$(date -u +%Y%m%dT%H%M%SZ)}"
echo "=== run_id=${RUN_ID} ==="

# Optional but recommended defaults for account 704052814573:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#   export BUDGETVLM_AWS_ACCOUNT_ID=704052814573
#   # or pin the bucket explicitly:
#   export BUDGETVLM_S3_BUCKET=budgetvlm-704052814573

echo "=== Ensure S3 bucket (create/update) ==="
if S3_URI="$(python -m src.s3_sync ensure 2>/tmp/budgetvlm_s3_ensure.log)"; then
  export BUDGETVLM_S3_URI="${S3_URI}"
  echo "Using ${BUDGETVLM_S3_URI}"
  cat /tmp/budgetvlm_s3_ensure.log || true
else
  echo "WARNING: S3 ensure failed — continuing without remote results save."
  cat /tmp/budgetvlm_s3_ensure.log || true
  unset BUDGETVLM_S3_URI || true
fi

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
  python -m src.s3_sync upload --uri "${BUDGETVLM_S3_URI}" --run-id "${RUN_ID}"
else
  echo "=== Skipping S3 upload (no bucket configured / ensure failed) ==="
fi

echo "Done. Inspect results/metrics/comparison.csv"
echo "run_id=${RUN_ID}"
if [[ -n "${BUDGETVLM_S3_URI:-}" ]]; then
  echo "s3_results=${BUDGETVLM_S3_URI}/${RUN_ID}"
fi
