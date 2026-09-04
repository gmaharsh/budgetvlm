#!/usr/bin/env bash
# Install torch + vLLM for CUDA 12.8 drivers (RunPod A40).
#
# Critical: `pip install vllm` from PyPI pulls torch+cu130 and breaks driver 12.8.
# Strategy: install vLLM, then FORCE torch/vision/audio back to cu128 last.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip setuptools wheel

CU128="https://download.pytorch.org/whl/cu128"

purge_cu13() {
  echo "=== Purge cu13 packages ==="
  # shellcheck disable=SC2046
  pkgs=$(pip freeze | rg -i 'cu13|torch==|torchvision==|torchaudio==|torchcodec==' | cut -d= -f1 || true)
  if [[ -n "${pkgs}" ]]; then
    pip uninstall -y ${pkgs} 2>/dev/null || true
  fi
  pip uninstall -y vllm 2>/dev/null || true
}

purge_cu13

echo "=== Base Python deps (no torch/vllm) ==="
pip install -r requirements-gpu.txt

echo "=== Install PyTorch cu128 first ==="
pip install --no-cache-dir \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.11.0+cu128" \
  --index-url "${CU128}"

echo "=== Install vLLM (may temporarily pull wrong torch — we fix next) ==="
pip install --no-cache-dir "vllm>=0.11.0,<0.29" \
  --extra-index-url "${CU128}" || \
pip install --no-cache-dir "vllm>=0.11.0" --extra-index-url "${CU128}"

echo "=== FORCE torch back to cu128 (must be last) ==="
pip install --force-reinstall --no-cache-dir \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.11.0+cu128" \
  --index-url "${CU128}"

# Remove any cu13 leftovers without touching cu128 torch
pip freeze | rg -i 'cu13' | cut -d= -f1 | xargs -r pip uninstall -y 2>/dev/null || true

echo "=== CUDA sanity check ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
assert "+cu128" in torch.__version__ or (
    torch.version.cuda is not None and torch.version.cuda.startswith("12.")
), f"Expected cu12x torch, got {torch.__version__} cuda={torch.version.cuda}"
assert torch.cuda.is_available(), "CUDA unavailable after install"
print("device", torch.cuda.get_device_name(0))
import vllm
print("vllm", getattr(vllm, "__version__", "?"))
print("OK")
PY

echo "GPU deps OK (torch cu128 pinned after vLLM)."
