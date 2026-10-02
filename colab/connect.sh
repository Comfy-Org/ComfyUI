#!/usr/bin/env bash
# Connects to ComfyUI on Colab through the relay that colab/ssh_tunnel.py forwards it to, and serves it at
# http://localhost:8188 until Ctrl+C. Runs in Git Bash or any shell with OpenSSH:
#   colab/connect.sh ubuntu@<relay ip> ~/.ssh/lightsail.pem
# The key can be left out when ~/.ssh/config already has one for the host.
set -euo pipefail

host=${1:?usage: connect.sh user@relay [key]}
key=${2:-}

echo "ComfyUI at http://localhost:8188, Ctrl+C to disconnect"
exec ssh -N ${key:+-i "$key"} -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L 8188:127.0.0.1:8188 "$host"
