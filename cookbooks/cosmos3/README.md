# Cosmos3 Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook (Reasoner and Generator).
Each cookbook README links back here for the backend(s) it supports — pick the
backend you want to run and follow that one section.

| Backend | Use it for | Used by |
| --- | --- | --- |
| [Cosmos Framework](#cosmos-framework) | Native PyTorch inference, launched with `torchrun` | Reasoner, Generator (Audiovisual, Action) |
| [Diffusers](#diffusers) | Direct generation with `Cosmos3OmniPipeline` | Generator (Audiovisual) |
| [Transformers](#transformers-coming-soon) | Hugging Face Transformers inference | Reasoner |
| [vLLM](#vllm) | OpenAI-compatible reasoning server (image/video understanding) | Reasoner |
| [vLLM-Omni](#vllm-omni) | OpenAI-compatible generation server (image/video/audio/action) | Generator (Audiovisual, Action) |

## Prerequisites

- Linux with NVIDIA GPU access.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos. Authenticate once before the first run:

  ```bash
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

- For the Cosmos Framework and vLLM backends: access to `git@github.com:NVIDIA/cosmos-framework.git`.
- Enough local disk for the venv/image, the uv cache, and the model cache. Nano
  downloads plus CUDA dependencies can take tens of GiB.

### CUDA driver and the `cuXXX` backend

Several backends pin a CUDA build of `torch`/`vllm` that **must match your NVIDIA
driver**. Pick the tag that matches the CUDA version your driver supports:

| Driver CUDA | Backend tag | Notes |
| --- | --- | --- |
| 13.x | `cu130` | Default in the notebooks (e.g. `vllm==0.21.0`). |
| 12.x | `cu128` | Use the `cu128` pair instead (e.g. `vllm==0.19.1`). |

vLLM does not publish a wheel for every CUDA minor version, so
`--torch-backend=auto` is not reliable here — choose the pair that matches your
driver.

## Cosmos Framework

Native PyTorch inference through the Cosmos Framework checkout. Used by the
`run_*_with_cosmos_framework.ipynb` notebooks and the Cosmos Framework
quickstarts.

From the `cosmos` repo root, clone (or reuse) the framework checkout:

```bash
mkdir -p packages
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
cd packages/cosmos3
```

Install the framework dependencies into its venv. The inference path currently
imports modules from the training extras, so use the `*-train` dependency group
that matches your driver (see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
# lerobot tracks test artifacts with git-LFS that this cookbook does not need;
# skipping smudge avoids failures from missing LFS blobs in uv's git mirror.
export GIT_LFS_SKIP_SMUDGE=1

# CUDA 13 driver (default):
uv sync --all-extras --group=cu130-train

# CUDA 12.x driver:
# uv sync --all-extras --group=cu128-train
```

The notebooks honor `COSMOS3_UV_GROUP` (default `cu130-train`); set
`export COSMOS3_UV_GROUP=cu128-train` before launching them on CUDA 12.x systems.

This produces a venv at `packages/cosmos3/.venv`. Run framework commands either
by activating it (`source .venv/bin/activate`) or via its absolute interpreter
(`.venv/bin/python`, `.venv/bin/torchrun`).

## Diffusers

Direct generation with `Cosmos3OmniPipeline` (Generator · Audiovisual). Create a
venv and install the backend, choosing `--torch-backend` to match your driver
(see [CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend)):

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  torch \
  torchvision \
  transformers
```

## Transformers (coming soon)

Support for Transformers-based Reasoner inference is coming soon.

## vLLM

OpenAI-compatible **reasoning** server for the Reasoner cookbook (image/video
understanding). Create a venv and install vLLM plus the `vllm-cosmos3` plugin,
which registers the `Cosmos3ReasonerForConditionalGeneration` architecture:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# CUDA 13 driver:
uv pip install --torch-backend=cu130 "vllm==0.21.0" \
  "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"

# CUDA 12.x driver:
# uv pip install --torch-backend=cu128 "vllm==0.19.1" \
#   "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3"
```

The vLLM version and the torch backend are paired — see
[CUDA driver and the `cuXXX` backend](#cuda-driver-and-the-cuxxx-backend).

If your vLLM build reports that DeepGEMM is unavailable, disable it before
starting the server:

```bash
export VLLM_USE_DEEP_GEMM=0
```

> When launching with `.venv/bin/vllm` instead of activating the venv, make sure
> `.venv/bin` is on `PATH` (e.g. `source .venv/bin/activate`). FlashInfer's
> just-in-time kernel build shells out to `ninja`, which lives in the venv.

## vLLM-Omni

OpenAI-compatible **generation** server (image/video/audio/action) for the
Generator cookbooks. The simplest path is the prebuilt Docker image
`vllm/vllm-omni:cosmos3`:

```bash
docker pull vllm/vllm-omni:cosmos3
```

Start a Nano server (publishes the OpenAI-compatible API on port 8000; mount any
directory whose local media/action files the server should read):

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000
```

For **Cosmos3-Super** (the larger 64B model), split the weights across GPUs and,
if needed, offload layers to reduce peak memory:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Super \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --tensor-parallel-size 4 \
  --enable-layerwise-offload \
  --port 8000
```

If you installed vLLM-Omni from the PR branch instead, run the same
`vllm serve ... --omni ...` command directly, without the
`docker run ... vllm/vllm-omni:cosmos3` wrapper.

## Verify the environment

For the Cosmos Framework / Diffusers / vLLM venvs, check that PyTorch sees the GPU:

```bash
.venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

For a vLLM / vLLM-Omni server, confirm it is serving the model:

```bash
curl http://localhost:8000/v1/models
```
