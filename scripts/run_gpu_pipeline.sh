#!/usr/bin/env bash
# Run on a CUDA machine AFTER pulling the VidCom2 token-pruning fix.
# Do NOT use frame-drop pruning; engines are one-per-rate with fixed frames.
set -euo pipefail

cd "$(dirname "$0")/.."
# Install GPU deps only if CUDA is not already working in the venv
# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
NEED_INSTALL=1
if python - <<'PY' 2>/dev/null
import torch
import vllm
assert torch.cuda.is_available()
assert "+cu12" in torch.__version__ or (torch.version.cuda or "").startswith("12.")
print("reuse_ok", torch.__version__, getattr(vllm, "__version__", "?"))
PY
then
  NEED_INSTALL=0
  echo "=== Reusing existing CUDA venv ==="
fi
if [[ "${NEED_INSTALL}" == "1" || "${FORCE_GPU_INSTALL:-0}" == "1" ]]; then
  bash scripts/install_gpu_deps.sh
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RUN_ID="${BUDGETVLM_RUN_ID:-runpod_$(date -u +%Y%m%dT%H%M%SZ)}"
echo "=== run_id=${RUN_ID} ==="
echo "=== Pruning: vLLM video_pruning_rate + VidCom2 | frames FIXED ==="

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

echo "=== One example @ 0% token pruning (fixed frames) ==="
python -m src.run_one_example --backend vllm --pruning-rate 0.0

echo "=== Fixed matrix on 5 videos first (sanity) ==="
python -m src.run_fixed_pruning --backend vllm --limit 5 --tag fixed5

echo "=== Expand to 20 videos ==="
python -m src.run_fixed_pruning --backend vllm --limit 20 --tag fixed20

echo "=== Tolerance + complexity + calibrate + BudgetVLM ==="
python -m src.analyze_tolerance --predictions results/predictions/fixed20_vllm.jsonl \
  --out results/metrics/tolerance.jsonl
python -m src.compute_complexity --limit 20 --out results/metrics/complexity.jsonl
python -m src.calibrate_thresholds \
  --predictions results/predictions/fixed20_vllm.jsonl \
  --complexity results/metrics/complexity.jsonl \
  --out results/metrics/calibrated_thresholds.json
# Use calibrated thresholds if present
T1=$(python - <<'PY'
import json
print(json.load(open("results/metrics/calibrated_thresholds.json"))["t1"])
PY
)
T2=$(python - <<'PY'
import json
print(json.load(open("results/metrics/calibrated_thresholds.json"))["t2"])
PY
)
python -m src.run_budgetvlm \
  --predictions results/predictions/fixed20_vllm.jsonl \
  --complexity results/metrics/complexity.jsonl \
  --tolerance results/metrics/tolerance.jsonl \
  --t1 "${T1}" --t2 "${T2}"
python -m src.make_plots --prefix smoke 2>/dev/null || true

if [[ -n "${BUDGETVLM_S3_URI:-}" ]]; then
  echo "=== Upload results to ${BUDGETVLM_S3_URI}/${RUN_ID} ==="
  python -m src.s3_sync upload --uri "${BUDGETVLM_S3_URI}" --run-id "${RUN_ID}"
else
  echo "=== Skipping S3 upload ==="
fi

echo "Done. Inspect results/metrics/comparison.csv"
echo "run_id=${RUN_ID}"
