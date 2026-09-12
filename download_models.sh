#!/bin/bash
# Download Wan 2.1 1.3B T2V models from Wan-AI HF repo (full downloads; Range is proxy-walled).
set -u
COMFY="C:/Users/Administrator/comfy/ComfyUI"
mkdir -p "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" "$COMFY/models/vae"
DL="$LOCALAPPDATA/Temp/wan_dl"; mkdir -p "$DL"
BASE="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main"
LOG="$DL/download.log"; : > "$LOG"

dl(){ # $1=url $2=dest $3=label
  echo "[$(date +%H:%M:%S)] START $3" >> "$LOG"
  for i in 1 2 3; do
    code=$(curl -L --retry 2 --retry-delay 3 --max-time 7200 --no-progress-bar \
      -o "$2" -w "%{http_code}" "$1" 2>>"$LOG")
    if [ "$code" = "200" ] && [ -s "$2" ]; then
      echo "[$(date +%H:%M:%S)] OK $3 ($(stat -c%s "$2") bytes)" >> "$LOG"; return 0
    fi
    echo "[$(date +%H:%M:%S)] attempt $i http=$code for $3" >> "$LOG"; sleep 4
  done
  echo "[$(date +%H:%M:%S)] FAILED $3" >> "$LOG"; return 1
}

# 1) Base diffusion model  (5.68GB)  -> wan2.1_t2v_1.3B.safetensors
[ -s "$COMFY/models/diffusion_models/wan2.1_t2v_1.3B.safetensors" ] || \
  dl "$BASE/diffusion_pytorch_model.safetensors" "$DL/base.safetensors" "base-1.3B"

# 2) T5 umt5-xxl text encoder  (11.36GB)  -> umt5_xxl_umt5-xxl-enc-bf16.pth
[ -s "$COMFY/models/text_encoders/umt5_xxl_umt5-xxl-enc-bf16.pth" ] || \
  dl "$BASE/models_t5_umt5-xxl-enc-bf16.pth" "$DL/t5_umt5.pth" "t5-umt5-xxl"

# 3) VAE (484MB)  -> wan_2.1_vae.pth
[ -s "$COMFY/models/vae/wan_2.1_vae.pth" ] || \
  dl "$BASE/Wan2.1_VAE.pth" "$DL/wan_vae.pth" "vae"

# Copy to canonical locations
[ -s "$DL/base.safetensors" ] && cp -f "$DL/base.safetensors" "$COMFY/models/diffusion_models/wan2.1_t2v_1.3B.safetensors" 2>>"$LOG"
[ -s "$DL/t5_umt5.pth" ]      && cp -f "$DL/t5_umt5.pth"      "$COMFY/models/text_encoders/umt5_xxl_umt5-xxl-enc-bf16.pth" 2>>"$LOG"
[ -s "$DL/wan_vae.pth" ]      && cp -f "$DL/wan_vae.pth"      "$COMFY/models/vae/wan_2.1_vae.pth" 2>>"$LOG"
echo "[$(date +%H:%M:%S)] DONE_ALL" >> "$LOG"
ls -la "$COMFY/models/diffusion_models" "$COMFY/models/text_encoders" "$COMFY/models/vae" >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
