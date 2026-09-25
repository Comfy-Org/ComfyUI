#!/usr/bin/env bash
# Container entrypoint for the ComfyUI wrapper worker image.
#
# One entrypoint serves both image targets and every place they run:
#   - a local GPU box through docker-compose.yml (bind mounts for models/ etc.);
#   - an on-demand RunPod pod started by voxmin-backend, where RunPod mounts the
#     pod's network volume at /workspace and runs the image's own CMD (there is
#     no compose file), so everything a pod needs is decided here from env.
#
# The runpod variant (COMFY_IMAGE_VARIANT=runpod) ships without torch and the
# pip requirements: the first pod on a network volume builds them into a
# virtualenv archived on the volume and every later pod unpacks it (see
# "Python env on the volume" below). It runs from the slim `runpod` image, or
# from RunPod's own cached PyTorch image via docker/runpod-bootstrap.sh.
#
# Usage:
#   (no args) or flags only   start ComfyUI; extra flags are appended
#   prefetch [spec ...]       download wrapper checkpoints into the model store
#                             and exit (fill a RunPod network volume once). The
#                             runpod image builds/verifies its Python env first.
#   prefetch --env-only       runpod image: only build/verify the Python env
#   <any other command>       run it as is (e.g. uvicorn for developer-api, bash)
set -euo pipefail

COMFY_ROOT=/opt/ComfyUI
cd "$COMFY_ROOT"

log() { echo "entrypoint: $*" >&2; }
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

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
	# One copy of each checkpoint on the (per-GB billed) volume: download next
	# to the models dir and move into place, instead of HF cache + a copy
	# (82 GB for H3 + FLUX.2 klein rather than 164 GB).
	export COMFY_HF_DOWNLOAD_MODE="${COMFY_HF_DOWNLOAD_MODE:-direct}"
	mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
fi

# --- Python env on the volume (runpod variant only) --------------------------
# The full image (target `comfyui`, :latest) bakes torch and requirements.txt
# in, which makes it a ~5 GB pull: a cold RunPod pod spent over 9 minutes just
# pulling it. The runpod variant keeps them on the network volume instead,
# built once by the first pod that needs them:
#
#   <data dir>/envs/<key>.tar      the env, archived once; renamed into place
#                                  only when complete
#   <data dir>/envs/.lock-<key>/   held by the pod building it
#   /opt/comfy-env/<key>/          the env itself, built or unpacked on each
#                                  pod's container disk (venvs are
#                                  path-dependent, so this path is fixed)
#   .../<key>/.comfy-env-complete  written last; no marker = partial
#
# Why one archive and not the env directory on the volume: RunPod volumes are
# MooseFS, which streams a big file fast (~800 MB/s measured) but creates small
# files slowly. An env written in place (tens of thousands of files) was still
# being written after 16 minutes, and every import from it pays a network
# round trip per file. One tar is written once and unpacked in seconds.
#
# <key> hashes docker/env-inputs (printed by docker/env-inputs.sh):
# requirements.txt, the torch index, the base interpreter, the arch and the OS.
# The slim image bakes it; code fetched by docker/runpod-bootstrap.sh computes
# it here at boot. An image whose dependencies changed builds a fresh env next
# to the old one, and one that only changed code reuses it.
#
# The lock is a directory: mkdir is atomic on a network filesystem where flock
# may not work (RunPod volumes are MooseFS). The owner writes its id into it
# and rewrites a heartbeat file every few seconds from a background loop. A
# waiting pod that sees owner + heartbeat unchanged for
# COMFY_ENV_LOCK_STALE_SECONDS, timed by its own clock (so clock skew between
# pods does not matter), treats the builder as dead: it takes the lock over
# (guarded by a second mkdir so two waiters cannot both do it) and rebuilds
# from scratch. A builder checks it still owns the lock before it writes the
# completion marker and before it publishes the archive.
ENV_MARKER_NAME=.comfy-env-complete
HEARTBEAT_PID=
ENV_TOKEN=

env_ready() { [[ -f "$ENV_DIR/$ENV_MARKER_NAME" && -x "$ENV_DIR/bin/python" ]]; }
archive_ready() { [[ -s "$ENV_ARCHIVE" ]]; }
lock_owner() { head -n 1 "$ENV_LOCK/owner" 2>/dev/null || true; }
lock_state() { { cat "$ENV_LOCK/owner" "$ENV_LOCK/heartbeat" 2>/dev/null || true; } | tr '\n' ' '; }

start_heartbeat() {
	(
		beat=0
		while :; do
			owner=$(lock_owner)
			# Stop once someone else owns the lock; an unreadable owner file is
			# treated as a transient network-filesystem hiccup.
			[[ -z "$owner" || "$owner" == "$ENV_TOKEN" ]] || exit 0
			beat=$((beat + 1))
			echo "beat $beat $(date -u +%FT%TZ)" >"$ENV_LOCK/heartbeat" 2>/dev/null || true
			sleep "$ENV_HEARTBEAT_SECONDS"
		done
	) &
	HEARTBEAT_PID=$!
}

release_env_lock() {
	if [[ -n "$HEARTBEAT_PID" ]]; then
		kill "$HEARTBEAT_PID" 2>/dev/null || true
		wait "$HEARTBEAT_PID" 2>/dev/null || true
		HEARTBEAT_PID=
	fi
	if [[ -n "$ENV_TOKEN" && "$(lock_owner)" == "$ENV_TOKEN" ]]; then
		rm -rf "$ENV_LOCK"
	fi
}

# Take over a lock whose state has not changed for the stale window. The
# `.break` dir makes the check-and-remove exclusive between waiting pods.
break_stale_lock() {
	local stale=$1 unchanged_for=$2
	if ! mkdir "$ENV_LOCK.break" 2>/dev/null; then
		# Another pod is taking it over right now (a millisecond job). A break
		# dir that outlives two stale windows belongs to a pod that died there.
		if ((unchanged_for >= 2 * ENV_LOCK_STALE_SECONDS)); then
			rmdir "$ENV_LOCK.break" 2>/dev/null || true
		fi
		return 0
	fi
	if [[ -d "$ENV_LOCK" && "$(lock_state)" == "$stale" ]]; then
		log "env lock unchanged for ${unchanged_for}s (owner: ${stale:-none}); its builder is gone, taking over"
		rm -rf "$ENV_LOCK"
	fi
	rmdir "$ENV_LOCK.break" 2>/dev/null || true
}

# Build the env on the container disk, then publish it to the volume as one
# archive. Called only while holding the lock.
build_env() {
	local t0=$SECONDS t base_python torch_info default_cache=/tmp/uv-cache
	base_python=$(readlink -f "$(command -v python3)")
	# uv's cache goes on the container disk too and is removed once the env is
	# built; packages are copied (not hardlinked) so removing it is safe.
	# Bytecode is compiled once here, so no pod compiles torch's sources on
	# import (PYTHONDONTWRITEBYTECODE is set).
	export UV_CACHE_DIR="${UV_CACHE_DIR:-$default_cache}" UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
		UV_PYTHON_DOWNLOADS=never UV_NO_CONFIG=1
	if [[ -e "$ENV_DIR" ]]; then
		log "removing the incomplete env at $ENV_DIR (no completion marker and no live builder)"
		t=$SECONDS
		rm -rf "$ENV_DIR"
		log "env build: removed it in $((SECONDS - t))s"
	fi
	uv venv --python "$base_python" "$ENV_DIR"
	t=$SECONDS
	uv pip install --python "$ENV_DIR/bin/python" --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
	log "env build: torch from $TORCH_INDEX_URL in $((SECONDS - t))s"
	t=$SECONDS
	uv pip install --python "$ENV_DIR/bin/python" -r "$COMFY_ROOT/requirements.txt"
	log "env build: requirements.txt in $((SECONDS - t))s"
	# CUDA only, like the full image's build-time assert. COMFY_ALLOW_CPU_TORCH=1
	# exists for local tests with a CPU torch index only; it defaults off.
	torch_info=$("$ENV_DIR/bin/python" -c '
import os, torch
cuda = torch.version.cuda
if not cuda and os.environ.get("COMFY_ALLOW_CPU_TORCH") != "1":
    raise SystemExit("torch %s is not a CUDA build; the runpod image is CUDA only" % torch.__version__)
print("torch=%s cuda=%s" % (torch.__version__, cuda))
')
	uv pip freeze --python "$ENV_DIR/bin/python" >"$ENV_DIR/comfy-env-freeze.txt"
	if [[ "$(lock_owner)" != "$ENV_TOKEN" ]]; then
		log "ERROR: lost the env lock during the build (owner is now: $(lock_owner)); not marking $ENV_DIR complete"
		exit 1
	fi
	{
		echo "key=$ENV_KEY"
		echo "$torch_info"
		echo "built_at=$(date -u +%FT%TZ)"
		echo "build_seconds=$((SECONDS - t0))"
		echo "builder=$ENV_TOKEN"
		cat "$COMFY_ROOT/docker/env-inputs"
	} >"$ENV_DIR/$ENV_MARKER_NAME.tmp"
	mv -f "$ENV_DIR/$ENV_MARKER_NAME.tmp" "$ENV_DIR/$ENV_MARKER_NAME"
	log "env build: $torch_info"
	if [[ "$UV_CACHE_DIR" == "$default_cache" ]]; then
		rm -rf "$UV_CACHE_DIR"
	fi
	archive_env
}

# Publish the finished env to the volume as one tar, written under a temporary
# name and renamed into place last, so no pod ever reads a partial archive.
archive_env() {
	local t=$SECONDS tmp="$ENV_ARCHIVE.tmp-$$-$RANDOM"
	tar -cf "$tmp" -C "$ENV_DIR" .
	if [[ "$(lock_owner)" != "$ENV_TOKEN" ]]; then
		rm -f "$tmp"
		log "ERROR: lost the env lock while archiving (owner is now: $(lock_owner)); not publishing $ENV_ARCHIVE"
		exit 1
	fi
	mv -f "$tmp" "$ENV_ARCHIVE"
	log "env build: archived to $ENV_ARCHIVE ($(du -h "$ENV_ARCHIVE" | cut -f1)) in $((SECONDS - t))s"
}

# Unpack the volume's archive beside the final path and rename it into place,
# so a pod killed mid-unpack never leaves a half env that looks complete.
unpack_env() {
	local t=$SECONDS partial="$ENV_DIR.partial"
	rm -rf "$partial" "$ENV_DIR"
	mkdir -p "$partial"
	if ! tar -xf "$ENV_ARCHIVE" -C "$partial" || [[ ! -f "$partial/$ENV_MARKER_NAME" || ! -x "$partial/bin/python" ]]; then
		rm -rf "$partial"
		log "ERROR: $ENV_ARCHIVE is not a usable env for this image (the unpack failed, or it has no completion marker or interpreter). Delete it; the next pod rebuilds it."
		exit 1
	fi
	mv "$partial" "$ENV_DIR"
	UNPACK_SECONDS=$((SECONDS - t))
}

ensure_env() {
	local t0=$SECONDS seen="" seen_at=$SECONDS state announced=""
	# Unpacked already: this container restarted.
	if env_ready; then
		log "python env $ENV_KEY found at $ENV_DIR, reusing it (checked in $((SECONDS - t0))s)"
		return 0
	fi
	mkdir -p "$ENVS_DIR" "$(dirname "$ENV_DIR")"
	while :; do
		if archive_ready; then
			unpack_env
			log "python env $ENV_KEY unpacked from $ENV_ARCHIVE in ${UNPACK_SECONDS}s, reusing it ($((SECONDS - t0))s in all)"
			return 0
		fi
		if mkdir "$ENV_LOCK" 2>/dev/null; then
			ENV_TOKEN="${RUNPOD_POD_ID:-${HOSTNAME:-container}} pid=$$ $(date -u +%FT%TZ) $RANDOM"
			echo "$ENV_TOKEN" >"$ENV_LOCK/owner"
			trap release_env_lock EXIT
			start_heartbeat
			if archive_ready; then # published by another pod since the check above
				release_env_lock
				trap - EXIT
				continue
			fi
			log "python env $ENV_KEY is not on the volume yet: building it at $ENV_DIR and archiving it to $ENV_ARCHIVE (once per volume and dependency set; other pods wait for it)"
			build_env
			release_env_lock
			trap - EXIT
			log "python env $ENV_KEY built in $((SECONDS - t0))s"
			return 0
		fi
		# Someone else holds the lock: wait for it, or take it over when stale.
		state=$(lock_state)
		if [[ "$state" != "$seen" ]]; then
			seen=$state
			seen_at=$SECONDS
		fi
		if [[ -z "$announced" ]]; then
			log "another pod is building python env $ENV_KEY (lock owner: $(lock_owner)); waiting up to ${ENV_WAIT_SECONDS}s"
			announced=1
		fi
		if ((SECONDS - seen_at >= ENV_LOCK_STALE_SECONDS)); then
			break_stale_lock "$seen" "$((SECONDS - seen_at))"
		fi
		if ((SECONDS - t0 >= ENV_WAIT_SECONDS)); then
			log "ERROR: gave up after ${ENV_WAIT_SECONDS}s waiting for python env $ENV_KEY (lock $ENV_LOCK, owner: $(lock_owner)). If no pod is building it, delete that directory."
			exit 1
		fi
		sleep "$ENV_POLL_SECONDS"
	done
}

if [[ "${COMFY_IMAGE_VARIANT:-}" == "runpod" ]]; then
	if [[ -z "$DATA_DIR" ]]; then
		log "ERROR: this is the runpod image. Its Python env (torch + requirements, several GB) lives on the network volume, and no volume is mounted. Mount one at /workspace (or set COMFYUI_DATA_DIR), or use shivanshtalwar0/comfyui:latest, which has everything baked in. For a shell without the env: --entrypoint bash."
		exit 64
	fi
	if [[ ! -s "$COMFY_ROOT/docker/env-key" ]]; then
		# Code fetched by docker/runpod-bootstrap.sh rather than baked into the
		# slim image: compute the key the image would have baked.
		bash "$COMFY_ROOT/docker/env-inputs.sh" >"$COMFY_ROOT/docker/env-inputs"
		sha256sum "$COMFY_ROOT/docker/env-inputs" | cut -c1-16 >"$COMFY_ROOT/docker/env-key"
		log "env key computed at boot: $(cat "$COMFY_ROOT/docker/env-key")"
	fi
	ENV_KEY=$(cat "$COMFY_ROOT/docker/env-key")
	export COMFY_ENV_KEY="$ENV_KEY"
	TORCH_INDEX_URL=$(sed -n 's/^torch_index=//p' "$COMFY_ROOT/docker/env-inputs")
	ENVS_DIR="${COMFY_ENVS_DIR:-$DATA_DIR/envs}"
	ENV_ARCHIVE="$ENVS_DIR/$ENV_KEY.tar"
	ENV_LOCK="$ENVS_DIR/.lock-$ENV_KEY"
	ENV_DIR="${COMFY_ENV_LOCAL_DIR:-/opt/comfy-env}/$ENV_KEY"
	ENV_WAIT_SECONDS="${COMFY_ENV_WAIT_SECONDS:-1800}"
	ENV_LOCK_STALE_SECONDS="${COMFY_ENV_LOCK_STALE_SECONDS:-120}"
	for value in "$ENV_WAIT_SECONDS" "$ENV_LOCK_STALE_SECONDS"; do
		is_uint "$value" || { log "ERROR: COMFY_ENV_WAIT_SECONDS / COMFY_ENV_LOCK_STALE_SECONDS must be whole seconds"; exit 64; }
	done
	ENV_HEARTBEAT_SECONDS=$((ENV_LOCK_STALE_SECONDS / 8 > 0 ? ENV_LOCK_STALE_SECONDS / 8 : 1))
	ENV_POLL_SECONDS=$((ENV_HEARTBEAT_SECONDS < 5 ? ENV_HEARTBEAT_SECONDS : 5))

	ensure_env
	# Everything below (the server, prefetch, passthrough commands) runs in it.
	export VIRTUAL_ENV="$ENV_DIR"
	export PATH="$ENV_DIR/bin:$PATH"
fi

# --- Subcommands ----------------------------------------------------------
if [[ "${1:-}" == "prefetch" ]]; then
	shift
	env_only=0
	prefetch_args=()
	for arg in "$@"; do
		if [[ "$arg" == "--env-only" ]]; then
			env_only=1
		else
			prefetch_args+=("$arg")
		fi
	done
	if ((env_only)); then
		if [[ "${COMFY_IMAGE_VARIANT:-}" == "runpod" ]]; then
			log "prefetch --env-only: python env $ENV_KEY is ready at $ENV_DIR"
		else
			log "prefetch --env-only: this image has its Python env baked in; nothing to do"
		fi
		exit 0
	fi
	exec python "$COMFY_ROOT/docker/prefetch_models.py" "${prefetch_args[@]}"
fi
if [[ $# -gt 0 && "$1" != -* ]]; then
	exec "$@"
fi

# --- ComfyUI server ---------------------------------------------------------
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

log "starting ComfyUI ($(command -v python), ${SECONDS}s after container start)"
exec python main.py "${args[@]}" "${extra[@]}" "$@"
