#!/usr/bin/env bash
# RunPod A40 (driver CUDA 12.8): install vLLM 0.28 *cu129* wheel + matching torch.
# Default PyPI vLLM is cu130 and will not work on this driver.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip setuptools wheel

echo "=== Wipe conflicting packages ==="
pip uninstall -y torch torchvision torchaudio torchcodec vllm 2>/dev/null || true
pip freeze | rg -i 'cu13' | cut -d= -f1 | xargs -r pip uninstall -y 2>/dev/null || true

echo "=== Base deps ==="
pip install -r requirements-gpu.txt

CU129="https://download.pytorch.org/whl/cu129"
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

echo "=== torch cu129 ==="
pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url "${CU129}"

echo "=== vLLM 0.28.0+cu129 wheel ==="
pip install --no-cache-dir "${VLLM_WHL}" --extra-index-url "${CU129}"

echo "=== Ensure cuDNN (exact pin for torch 2.13+cu129) ==="
pip install --force-reinstall --no-cache-dir "nvidia-cudnn-cu12==9.20.0.48"

echo "=== Sanity check ==="
if ! python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable with cu129 stack"
print("device", torch.cuda.get_device_name(0))
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
print("OK")
PY
then
  echo "cu129 failed — falling back to cu128 torch 2.10 + vLLM 0.11.2 (EVS)"
  pip uninstall -y torch torchvision torchaudio vllm 2>/dev/null || true
  pip freeze | rg -i 'cu13' | cut -d= -f1 | xargs -r pip uninstall -y 2>/dev/null || true
  CU128="https://download.pytorch.org/whl/cu128"
  pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url "${CU128}"
  pip install --force-reinstall --no-cache-dir nvidia-cudnn-cu12
  pip install --no-cache-dir "vllm==0.11.2" --extra-index-url "${CU128}"
  pip install --force-reinstall --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --index-url "${CU128}"
  pip install --force-reinstall --no-cache-dir nvidia-cudnn-cu12
  python - <<'PY'
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
PY
  mkdir -p results/metrics
  echo evs > results/metrics/pruning_method_hint.txt
else
  mkdir -p results/metrics
  echo vidcom2 > results/metrics/pruning_method_hint.txt
fi

echo "method hint: $(cat results/metrics/pruning_method_hint.txt)"
echo "GPU deps OK."
