#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  hp8)
    project="${LTX_HP8_PROJECT:-$HOME/Project/opensource/github/ai-video3}"
    ui="$project/iris-xe-9f/native-mixed/full-ui"
    launch=("$project/iris-xe-9f/native-mixed/run-xpu-full.sh")
    ;;
  hp)
    project="${LTX_LOCAL_PROJECT:-$HOME/Project/opensource/github/agent-offline/ltx-video}"
    ui="$project/homdx-fork/ComfyUI"
    launch=("$project/run-xpu-local.sh" --cpu-vae)
    ;;
  *) echo "Usage: bash run-t2v.sh hp8|hp" >&2; exit 2 ;;
esac

for file in "${launch[0]}" "$ui/custom_nodes/ltx_duration_noise.py" "$ui/custom_nodes/ltx_text_continuation.py" "$ui/user/default/workflows/LTX25_Q5_T2V_DurationNoise.json"; do
    if [[ ! -r "$file" ]]; then
        echo "Missing: $file" >&2
        exit 1
    fi
done
shift
exec bash "${launch[@]}" "$@"
