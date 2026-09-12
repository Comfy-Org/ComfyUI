#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${COMFYUI_VENV:-${ROOT_DIR}/.venv}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This installer supports Apple Silicon macOS only." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} was not found. Install Python 3.11 or 3.12 first." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("ERROR: Python 3.11 or 3.12 is required.")
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python "${ROOT_DIR}/scripts/verify-macos.py"

cat <<EOF

Mac setup completed successfully.

Start ComfyUI with:
  bash "${ROOT_DIR}/scripts/run-macos.sh"

Open:
  http://127.0.0.1:8188
EOF
