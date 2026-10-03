#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${COMFYUI_VENV:-${ROOT_DIR}/.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: Mac environment not found. Run scripts/setup-macos.sh first." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

LISTEN_ADDRESS="${COMFYUI_LISTEN:-127.0.0.1}"
PORT="${COMFYUI_PORT:-8188}"

cd "${ROOT_DIR}"
exec "${VENV_DIR}/bin/python" main.py --listen "${LISTEN_ADDRESS}" --port "${PORT}" "$@"
