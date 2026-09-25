# ComfyUI + wrapper API worker image (published as shivanshtalwar0/comfyui).
#
# The same image runs on a local GPU box (docker-compose.yml) and as an
# on-demand RunPod pod started by voxmin-backend's FloStudio scheduler. The
# entrypoint (docker/entrypoint.sh) decides everything from env, because a
# RunPod pod runs the image's own CMD with no compose file around it.
#
# Two published targets:
#   - `comfyui` (the default, :latest): torch + requirements baked in, ~5 GB
#     compressed. The local GPU box pulls it once and keeps it.
#   - `runpod` (:runpod): the same system packages and sources but NO Python
#     dependencies. The entrypoint installs them once into a virtualenv on the
#     pod's network volume (/workspace/envs/<env key>) and every later pod
#     reuses it, so a cold pod pulls a few hundred MB instead of ~5 GB.
#
# Build stages are ordered so Docker caches aggressively:
#   - `system`: apt packages, shared by both targets.
#   - `base`: + torch + pip requirements. Rebuilt ONLY when requirements*.txt
#     or the torch build args change.
#   - `comfyui`: the ComfyUI sources on top of base. Rebuilt ONLY when the
#     source tree changes (models/, output/ etc. are excluded via .dockerignore
#     and mounted or linked at runtime).
#   - `runpod`: system + the uv binary + the sources + the env key.
#
# CUDA only. The PyTorch CUDA wheels bundle the CUDA runtime, so this slim
# image uses the host GPU through the NVIDIA container toolkit. cu128 is the
# oldest CUDA with Blackwell (RTX 5090, sm_120) kernels and needs host driver
# >= 570, which every RTX 5090 host already has. A newer CUDA index can be
# chosen (e.g. --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130),
# but any torch that is not a CUDA build fails the image build (`comfyui`) or
# the env build on the volume (`runpod`).
ARG BASE_IMAGE=python:3.12-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.2

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE} AS system

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /opt/ComfyUI

# The compilers stay in both targets: Triton JIT-compiles kernels at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        git \
        libglib2.0-0 \
        libgl1 \
        make \
        tini && \
    rm -rf /var/lib/apt/lists/*

FROM system AS base

ARG TORCH_INDEX_URL

# torch first, from the CUDA index above, so requirements.txt (which lists
# torch unpinned) finds it already satisfied instead of pulling PyPI's default.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

COPY requirements.txt manager_requirements.txt ./
# The assert runs after requirements.txt so a dependency that swaps torch for
# a CPU wheel fails the build instead of shipping a GPU-less worker.
RUN python -m pip install -r requirements.txt && \
    python -c "import torch; assert torch.version.cuda, 'torch %s is not a CUDA build' % torch.__version__; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"

# --- RunPod target: the Python env lives on the network volume ----------------
FROM system AS runpod

ARG TORCH_INDEX_URL

# Only the static uv binary (not uvx): it builds the env on the first boot.
COPY --from=uv /uv /usr/local/bin/uv

COPY . .

# The env key names the env archive on the volume (<data dir>/envs/<key>.tar).
# It hashes everything the env's contents depend on, so an image whose
# dependencies changed builds a fresh env next to the old one, while an image
# that only changed code reuses the existing one. docker/env-inputs.sh prints
# the hashed text (the entrypoint runs the same script at boot for code that
# docker/runpod-bootstrap.sh fetched, so both keys agree); docker/env-inputs
# keeps it, and the entrypoint reads the torch index from it.
RUN mkdir -p input output temp user models api_server/workflows && \
    chmod +x docker/entrypoint.sh docker/env-inputs.sh docker/runpod-bootstrap.sh && \
    TORCH_INDEX_URL="${TORCH_INDEX_URL}" docker/env-inputs.sh > docker/env-inputs && \
    sha256sum docker/env-inputs | cut -c1-16 > docker/env-key && \
    cat docker/env-inputs docker/env-key

ENV COMFY_IMAGE_VARIANT=runpod \
    COMFYUI_PORT=8188
EXPOSE 8188 8000

# The first boot on a fresh volume builds the env (minutes), hence the long
# start period. The check only needs the stdlib, so the system python runs it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=5 \
    CMD python -c "import os, urllib.request; t = os.environ.get('WRAPPER_AUTH_TOKEN'); urllib.request.urlopen(urllib.request.Request('http://localhost:%s/system_stats' % os.environ.get('COMFYUI_PORT', '8188'), headers={'Authorization': 'Bearer ' + t} if t else {}), timeout=8)"

ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]
CMD []

# --- Full target. Last, so a plain `docker build .` still builds it ------------
FROM base AS comfyui

COPY . .

RUN mkdir -p input output temp user models api_server/workflows && \
    chmod +x docker/entrypoint.sh

# 8188: ComfyUI + /api/wrapper (the RunPod proxy and voxmin-backend use this
# port); 8000: the optional developer-api service.
ENV COMFYUI_PORT=8188
EXPOSE 8188 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import os, urllib.request; t = os.environ.get('WRAPPER_AUTH_TOKEN'); urllib.request.urlopen(urllib.request.Request('http://localhost:%s/system_stats' % os.environ.get('COMFYUI_PORT', '8188'), headers={'Authorization': 'Bearer ' + t} if t else {}), timeout=8)"

ENTRYPOINT ["tini", "--", "/opt/ComfyUI/docker/entrypoint.sh"]
CMD []
