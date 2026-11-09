#!/usr/bin/env bash
# Pack isolation by hand. (With COMFY_PACK_ISOLATION=true ComfyUI starts each pack itself, when a
# render or one of the pack's endpoints needs it.)
#
#   scripts/pack.sh start <pack dir> <port> [--host 127.0.0.1] [--token-file F]
#   scripts/pack.sh stop <port>
#   scripts/pack.sh status
#   scripts/pack.sh describe <pack dir>      # its nodes, routes and web directory, as JSON
#   scripts/pack.sh manifest <pack dir>      # write a V2 pack's comfy-pack.json (imports the pack)
#   scripts/pack.sh install <pack dir>       # prepare a V2 pack: its environment and its weights
#
# A pack dir with a v2/ conversion is loaded from v2/. Logs and run files are in
# $COMFY_PACK_RUN_DIR (default /tmp/comfy-packs-$UID). PYTHON selects the interpreter.
set -euo pipefail
COMFY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COMFY_ROOT"
exec "${PYTHON:-python3}" -m comfy_api._impl.isolation.server "$@"
