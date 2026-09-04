#!/usr/bin/env bash
# Install torch + vLLM for CUDA 12.8 drivers (RunPod A40 / similar).
# Avoids PyPI default wheels that pull CUDA 13 and then fail with:
#   "NVIDIA driver on your system is too old (found version 12080)"
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip

echo "=== Removing CUDA-mismatched packages (if any) ==="
pip uninstall -y torch torchvision torchaudio torchcodec vllm 2>/dev/null || true

echo "=== Base deps ==="
pip install -r requirements-gpu.txt

echo "=== PyTorch cu128 ==="
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== vLLM (cu128 torch index) ==="
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

echo "=== CUDA sanity check ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA still unavailable — check driver vs torch build"
print("device", torch.cuda.get_device_name(0))
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
PY

echo "GPU deps OK."
