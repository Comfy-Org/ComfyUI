# ComfyUI Docker 镜像
# RTX 50 系列（Blackwell）GPU 镜像

ARG PYTHON_BASE_IMAGE=python:3.10-slim-bookworm
FROM ${PYTHON_BASE_IMAGE} AS base

# 构建参数
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG APT_MIRROR=
ARG APT_SECURITY_MIRROR=
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

# 避免 Python 生成 .pyc 文件并开启无缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（包含 OpenCV、多媒体处理等常用库）
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org/debian|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && if [ -n "${APT_SECURITY_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先安装 PyTorch 相关重型依赖（利用构建缓存）
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

# 安装项目 Python 依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 安装随 docker-compose 挂载的 custom_nodes 常用运行依赖
COPY docker-extra-requirements.txt /app/docker-extra-requirements.txt
RUN pip install --no-cache-dir -r /app/docker-extra-requirements.txt

# Triton 首次运行会即时编译 CUDA 驱动辅助模块，需要本地 C 工具链
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc

# 复制项目源码
COPY . /app

# 创建 ComfyUI 所需的默认数据目录
RUN mkdir -p /app/models /app/output /app/input /app/user /app/temp /app/custom_nodes

# 暴露 ComfyUI 默认端口
EXPOSE 8188

# GPU 模式监听所有接口，方便从容器外部访问
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
