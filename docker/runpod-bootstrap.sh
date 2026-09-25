#!/usr/bin/env bash
# Start command for a RunPod pod on RunPod's own PyTorch image. RunPod keeps
# that image cached on its machines, so a pod starts in seconds instead of
# pulling ours from Docker Hub (measured: ~10 Mbps, 4-5 min for the 370 MB
# slim image, 9+ min for the full one). Everything else lives on the network
# volume:
#
#   1. this repository at COMFY_CODE_REF, fetched once per commit:
#        <data dir>/code/ComfyUI-<ref>.tar.gz
#   2. unpacked to /opt/ComfyUI on the container disk (seconds);
#   3. docker/entrypoint.sh then unpacks (or, once per volume, builds) the
#      Python env archived on the volume, and starts ComfyUI. The model
#      weights, the Hugging Face cache and Triton kernels are on the volume.
#
# voxmin-backend puts this script in a RunPod template as the pod's start
# command, with COMFY_CODE_REF pinned (its scripts/flo-runpod-template.mjs).
#
# Env:
#   COMFY_CODE_REF    required: a commit sha. A branch name would be cached on
#                     the volume under that name and never updated.
#   COMFY_CODE_REPO   default januscaler/ComfyUI
#   COMFY_CODE_URL    override the tarball URL (tests)
#   COMFYUI_DATA_DIR  the network volume, default /workspace
# Arguments are passed on to the entrypoint.
set -euo pipefail

log() { echo "bootstrap: $*" >&2; }
t0=$SECONDS

ref="${COMFY_CODE_REF:-}"
if [[ ! "$ref" =~ ^[0-9a-f]{7,40}$ ]]; then
	log "ERROR: COMFY_CODE_REF must be a commit sha of the ComfyUI repository (got '$ref')"
	exit 64
fi
data="${COMFYUI_DATA_DIR:-/workspace}"
if [[ ! -d "$data" || ! -w "$data" ]]; then
	log "ERROR: no writable network volume at $data; this start command keeps the code, the Python env and the models there"
	exit 64
fi
url="${COMFY_CODE_URL:-https://codeload.github.com/${COMFY_CODE_REPO:-januscaler/ComfyUI}/tar.gz/$ref}"
store="$data/code"
archive="$store/ComfyUI-$ref.tar.gz"
root=/opt/ComfyUI

mkdir -p "$store"
if [[ -s "$archive" ]]; then
	log "code $ref found on the volume"
else
	t=$SECONDS
	tmp="$archive.tmp-$$-$RANDOM"
	# python3 rather than curl: both RunPod's image and ours have it.
	python3 - "$url" "$tmp" <<'PY'
import sys
import time
import urllib.request

url, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 6):
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        break
    except Exception as error:
        if attempt == 5:
            raise SystemExit(f"bootstrap: could not download {url}: {error}")
        time.sleep(2 * attempt)
PY
	gzip -t "$tmp"
	# Renamed into place last, so no pod ever reads a partial download. Two pods
	# racing here both hold the same commit; the last rename wins harmlessly.
	mv -f "$tmp" "$archive"
	log "code $ref downloaded to $archive in $((SECONDS - t))s"
fi

# Out of $root first: an image whose WORKDIR is /opt/ComfyUI would leave this
# shell in a deleted directory.
cd /
rm -rf "$root"
mkdir -p "$root"
tar -xzf "$archive" -C "$root" --strip-components=1
mkdir -p "$root"/{input,output,temp,user,models,api_server/workflows}
log "code $ref unpacked to $root ($((SECONDS - t0))s since start)"

# What the slim image sets with ENV.
export COMFY_IMAGE_VARIANT=runpod COMFYUI_DATA_DIR="$data" COMFYUI_PORT="${COMFYUI_PORT:-8188}" \
	PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
exec bash "$root/docker/entrypoint.sh" "$@"
