FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1
ARG TORCH_CUDA=cu130

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    COMFYUI_PORT=8188 \
    COMFYUI_ARGS="" \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    && python -m pip freeze | grep -E '^(torch|torchvision)==' > /opt/torch-constraints.txt \
    && python -m pip install -r requirements.txt -c /opt/torch-constraints.txt \
    && python -m pip check

COPY . .
EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['COMFYUI_PORT']+'/system_stats',timeout=5).read()"
CMD ["python", "/app/deploy/easypanel/start.py"]
