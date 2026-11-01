#!/bin/bash
# Resumable T5 download: the egress proxy drops long connections, so loop
# `curl --continue-at -` with a bounded timeout, appending until full size.
set -u
DEST="C:/Users/Administrator/comfy/ComfyUI/models/text_encoders/umt5_xxl_umt5-xxl-enc-bf16.pth"
URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
TARGET=11361920418
LOG="$LOCALAPPDATA/Temp/wan_dl/t5_resume.log"
: > "$LOG"
attempts=0
while :; do
  sz=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  if [ "$sz" -ge $((TARGET-1000000)) ]; then
    echo "[$(date +%H:%M:%S)] DONE size=$sz" >> "$LOG"; break
  fi
  attempts=$((attempts+1))
  # Bounded attempt so a dead socket can't hang forever; --continue-at - resumes offset.
  curl -L --continue-at - --max-time 240 --retry 1 --retry-delay 2 --no-progress-bar \
    -o "$DEST" "$URL" >> "$LOG" 2>&1
  sz2=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] attempt $attempts: $sz -> $sz2 bytes ($((${sz2}*100/TARGET))%)" >> "$LOG"
  if [ "$attempts" -ge 200 ]; then echo "[$(date +%H:%M:%S)] GAVE UP after 200 attempts size=$sz2" >> "$LOG"; break; fi
  sleep 3
done
echo "FINAL_EXIT=$? size=$(stat -c%s "$DEST" 2>/dev/null)" >> "$LOG"
