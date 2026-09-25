#!/usr/bin/env bash
# ComfyUI Apple Silicon (M5 Max 128GB) Optimized Launch Script

set -e

# Enable PyTorch MPS fallback for ops without direct Metal implementations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Maximize MPS memory allocation efficiency
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "=========================================================="
echo "🚀 Starting ComfyUI on Apple Silicon (M5 Max 128GB Unified Memory)"
echo "✨ Features: High VRAM mode, PyTorch MPS acceleration, Auto Preview"
echo "🌐 URL: http://127.0.0.1:8188"
echo "=========================================================="

exec "$PYTHON_EXEC" main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --highvram \
    --preview-method auto \
    "$@"
