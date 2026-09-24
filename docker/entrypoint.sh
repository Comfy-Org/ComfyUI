#!/usr/bin/env bash
# Container entrypoint for the ComfyUI wrapper worker image.
#
# One image serves every place it runs:
#   - a local GPU box through docker-compose.yml (bind mounts for models/ etc.);
#   - an on-demand RunPod pod started by voxmin-backend, where RunPod mounts the
#     pod's network volume at /workspace and runs the image's own CMD (there is
#     no compose file), so everything a pod needs is decided here from env.
#
# Usage:
#   (no args) or flags only   start ComfyUI; extra flags are appended
#   prefetch [spec ...]       download wrapper checkpoints into the model store
#                             and exit (fill a RunPod network volume once)
#   <any other command>       run it as is (e.g. uvicorn for developer-api, bash)
set -euo pipefail

COMFY_ROOT=/opt/ComfyUI
cd "$COMFY_ROOT"

# --- Persistent storage --------------------------------------------------
# COMFYUI_DATA_DIR holds what must outlive a container: model weights, the
# Hugging Face cache and compiled Triton kernels. On RunPod that is the
# network volume at /workspace, which is also what makes a cold pod cheap:
# weights and kernels are already there instead of being fetched again.
DATA_DIR="${COMFYUI_DATA_DIR:-}"
if [[ -z "$DATA_DIR" && -d /workspace && -w /workspace ]]; then
	DATA_DIR=/workspace
fi

# Point $COMFY_ROOT/<name> at <target> unless something is bind-mounted there
# (a compose volume always wins over the data dir).
link_dir() {
	local name=$1 target=$2
	mkdir -p "$target"
	if [[ -L "$name" ]]; then
		ln -sfn "$target" "$name"
		echo "entrypoint: $name -> $target" >&2
		return
	fi
	if [[ -d "$name" ]] && mountpoint -q "$name"; then
		echo "entrypoint: $name is a mount; leaving it in place" >&2
		return
	fi
	if [[ -d "$name" ]]; then
		cp -an "$name"/. "$target"/ 2>/dev/null || true
		rm -rf "$name"
	fi
	ln -sfn "$target" "$name"
	echo "entrypoint: $name -> $target" >&2
}

MODELS_DIR="${COMFYUI_MODELS_DIR:-${DATA_DIR:+$DATA_DIR/models}}"
if [[ -n "$MODELS_DIR" ]]; then
	link_dir models "$MODELS_DIR"
fi
if [[ -n "$DATA_DIR" ]]; then
	export HF_HOME="${HF_HOME:-$DATA_DIR/huggingface}"
	export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_DIR/.triton}"
	mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
fi

# --- Subcommands ----------------------------------------------------------
if [[ "${1:-}" == "prefetch" ]]; then
	shift
	exec python "$COMFY_ROOT/docker/prefetch_models.py" "$@"
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
	exec "$@"
fi

# --- ComfyUI server ---------------------------------------------------------
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

args=(--listen "${COMFYUI_LISTEN:-0.0.0.0}" --port "${COMFYUI_PORT:-8188}")

# The wrapper fetches the checkpoints a workflow needs on first use.
if [[ "${AUTO_DOWNLOAD_MODELS:-1}" == "1" ]]; then
	args+=(--auto-download-models)
fi

# VRAM/RAM tuning, same defaults as the compose file always used (see
# .env.example for why: MiniMax H3's ~43 GB of checkpoints on a 32 GB box).
args+=(--vram-headroom "${VRAM_HEADROOM_GB:-2}")
read -r -a cache_ram <<<"${CACHE_RAM_GB:-2 8}"
args+=(--cache-ram "${cache_ram[@]}")
streams="${ASYNC_OFFLOAD_STREAMS:-0}"
if is_uint "$streams" && ((streams > 0)); then
	args+=(--async-offload "$streams")
else
	args+=(--disable-async-offload)
fi
if [[ "${FAST_DISK:-1}" == "1" ]]; then
	args+=(--fast-disk)
fi

read -r -a extra <<<"${COMFYUI_ARGS:-}"

if [[ -n "${WRAPPER_AUTH_TOKEN:-}" ]]; then
	echo "entrypoint: bearer auth is ON for every route" >&2
else
	echo "entrypoint: WARNING: WRAPPER_AUTH_TOKEN is unset; the server is open to anyone who can reach it" >&2
fi

exec python main.py "${args[@]}" "${extra[@]}" "$@"
