#!/usr/bin/env bash
set -e
#######################################################
# Usage examples:
#
#     ComfyUI:
#       ./setup.sh
#
#     With manager:
#       ./setup.sh --manager
#
#     Choose PyTorch:
#       Default: ./setup.sh --torch=intel
#       Nightly: ./setup.sh --torch=amd --torch-nightly
#       Exact version: TORCH_NVIDIA_STABLE=cu126 ./setup.sh --torch=nvidia
#######################################################

# PyTorch index suffixes (edit here or override via env)
TORCH_AMD_STABLE="${TORCH_AMD_STABLE:-rocm7.1}"
TORCH_AMD_NIGHTLY="${TORCH_AMD_NIGHTLY:-rocm7.2}"

TORCH_INTEL_STABLE="${TORCH_INTEL_STABLE:-xpu}"
TORCH_INTEL_NIGHTLY="${TORCH_INTEL_NIGHTLY:-xpu}"

TORCH_NVIDIA_STABLE="${TORCH_NVIDIA_STABLE:-cu130}"
TORCH_NVIDIA_NIGHTLY="${TORCH_NVIDIA_NIGHTLY:-cu130}"

#######################################################

echo "Start setup ..."

# options defaults
MANAGER=false
NIGHTLY=false
TORCH=""

# parse args
for i in "$@"; do
  case $i in
    --manager) MANAGER=true; shift ;;
    --torch=*) TORCH="${i#*=}"; shift ;;
    --torch-nightly) NIGHTLY=true; shift ;;
    *) ;; # skip unknown
  esac
done

# basics
echo "Create virtual environment ..."
uv sync

echo "Install basic dependencies ..."
uv pip install -r requirements.txt

# manager requirements
if [ "$MANAGER" = true ]; then
  echo "Install manager dependencies ..."
  uv pip install -r manager_requirements.txt
fi

# PyTorch handling
if [ -n "$TORCH" ]; then
  INDEX_OPT="--default-index"
  if [ "$NIGHTLY" = true ]; then
    PRE_FLAG="--pre"
    NIGHTLY_PREFIX="nightly/"
  else
    PRE_FLAG=""
    NIGHTLY_PREFIX=""
  fi

  case $TORCH in
    amd)
      SUFFIX="$([ "$NIGHTLY" = true ] && echo "$TORCH_AMD_NIGHTLY" || echo "$TORCH_AMD_STABLE")"
      ;;
    intel)
      SUFFIX="$([ "$NIGHTLY" = true ] && echo "$TORCH_INTEL_NIGHTLY" || echo "$TORCH_INTEL_STABLE")"
      ;;
    nvidia)
      SUFFIX="$([ "$NIGHTLY" = true ] && echo "$TORCH_NVIDIA_NIGHTLY" || echo "$TORCH_NVIDIA_STABLE")"
      [ "$NIGHTLY" != true ] && INDEX_OPT="--index"
      ;;
    *)
      echo "Error: unsupported torch - $TORCH"
      exit 1
      ;;
  esac

  URL="https://download.pytorch.org/whl/${NIGHTLY_PREFIX}${SUFFIX}"

  echo "Install torch for ${TORCH} [nightly=${NIGHTLY}] from ${URL} ..."
  uv pip install -U $PRE_FLAG torch torchvision torchaudio $INDEX_OPT "$URL"
fi

echo "Setup done"
