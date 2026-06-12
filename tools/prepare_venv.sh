#!/bin/bash
set -exo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    export CONDA_ENV="${CONDA_ENV:-base}"
    conda activate "$CONDA_ENV"
fi

PREFIX="${PREFIX:-$HOME/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PREFIX/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-16}"

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
VENV_DIR="${VENV_DIR:-venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --seed --managed-python
fi

uv pip sync "$ROOT_DIR/tools/requirements.lock" \
    --python "$VENV_DIR/bin/python" \
    --torch-backend "$TORCH_BACKEND" \
    --index-strategy unsafe-best-match \
    --link-mode=hardlink

"$VENV_DIR/bin/python" - <<'PY'
import torch
import diffusers
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

print("python import check ok")
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("diffusers", diffusers.__version__)
print("pipeline", Cosmos3OmniPipeline.__name__)
print("scheduler", UniPCMultistepScheduler.__name__)
PY
