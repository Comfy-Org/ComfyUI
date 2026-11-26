FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime@sha256:60f22fb80755fd0b470fb47928dbd55816aa9f847edd95cf43c93253507a9ddf

ARG COMFYUI_REVISION
LABEL org.opencontainers.image.revision=$COMFYUI_REVISION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /comfyui

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /comfyui/user

EXPOSE 8188

ENTRYPOINT ["python", "main.py"]
CMD ["--listen", "0.0.0.0", "--user-directory", "/comfyui/user"]
