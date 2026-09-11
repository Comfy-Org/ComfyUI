#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# AMD measured CK-backed DAO Flash Attention as the fastest Navi31 path.
# Set FLASH_ATTN_BACKEND=triton to benchmark the optional Triton path.
case "${FLASH_ATTN_BACKEND:-ck}" in
  ck)
    unset FLASH_ATTENTION_TRITON_AMD_ENABLE
    unset FLASH_ATTENTION_TRITON_AMD_AUTOTUNE
    ;;
  triton)
    .venv/bin/python -c 'import aiter' >/dev/null 2>&1 || {
      printf 'Triton mode requires Aiter; rerun flash-attn with INSTALL_AITER=1\n' >&2
      exit 2
    }
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="${FLASH_ATTENTION_TRITON_AMD_AUTOTUNE:-TRUE}"
    ;;
  *)
    printf 'FLASH_ATTN_BACKEND must be ck or triton\n' >&2
    exit 2
    ;;
esac

# AMD's benchmark enables MIOpen on RDNA and prefers hipBLASLt.
export COMFYUI_ENABLE_MIOPEN="${COMFYUI_ENABLE_MIOPEN:-1}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-1}"

# Optional one-time cold-cache kernel search. Disable for normal warm runs.
if [[ "${MIOPEN_TUNE:-0}" == 1 ]]; then
  export MIOPEN_FIND_MODE=1
  export MIOPEN_FIND_ENFORCE=4
  export MIOPEN_SEARCH_CUTOFF=ON
else
  unset MIOPEN_FIND_MODE MIOPEN_FIND_ENFORCE MIOPEN_SEARCH_CUTOFF
fi

# xformers has higher priority than Flash Attention, so bypass it explicitly.
.venv/bin/python main.py --disable-xformers --use-flash-attention "$@"