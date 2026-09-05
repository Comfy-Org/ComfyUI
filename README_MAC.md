# Apple Silicon macOS deployment

This branch targets Apple Silicon Macs, including the 2023 M2 Max MacBook Pro.
Native MPS is the supported inference path. Docker Compose is included only as
a CPU-only compatibility option because Docker Desktop cannot pass Metal/MPS
through to an ordinary Linux container.

## Native MPS setup (recommended)

Requirements:

- Apple Silicon Mac
- A current macOS release
- Python 3.11 or 3.12 (`brew install python@3.12`)
- Git and Xcode command-line tools (`xcode-select --install`)

Install:

```bash
git switch Mac_build
bash scripts/setup-macos.sh
```

Start ComfyUI:

```bash
bash scripts/run-macos.sh
```

Open <http://127.0.0.1:8188>. The verification step fails early if the Python
wheel cannot execute an MPS tensor operation.

To allow another machine on the LAN to connect, use:

```bash
COMFYUI_LISTEN=0.0.0.0 bash scripts/run-macos.sh
```

Only do this on a trusted network. macOS may ask whether Python can accept
incoming connections.

Apple unified memory is shared by macOS, CPU, and GPU. Close memory-intensive
applications before running large workflows. `PYTORCH_ENABLE_MPS_FALLBACK=1`
is set by the launcher so unsupported MPS operators can fall back to CPU.

## Docker Compose (CPU-only)

```bash
docker compose -f docker-compose.mac.yml up --build
```

This builds a native `linux/arm64` image and starts ComfyUI at
<http://127.0.0.1:8188>. It is suitable for API and workflow compatibility
testing, but diffusion inference is much slower than native MPS.

## Environment overrides

- `PYTHON_BIN`: Python executable used during setup
- `COMFYUI_VENV`: virtual environment path (default `.venv`)
- `COMFYUI_LISTEN`: listen address (default `127.0.0.1`)
- `COMFYUI_PORT`: HTTP port (default `8188`)
- `PYTORCH_ENABLE_MPS_FALLBACK`: MPS fallback toggle (default `1`)
