#!/usr/bin/env bash
# Prints the text the Python env key hashes (docker/env-inputs). The key names
# the env archive on the RunPod volume (see "Python env on the volume" in
# docker/entrypoint.sh), so everything the env's contents depend on goes in.
#
# One script for both places that need it, so they always agree:
#   - the Dockerfile's `runpod` stage bakes docker/env-inputs and docker/env-key;
#   - docker/entrypoint.sh computes them at boot when the code did not come from
#     that image (RunPod's own cached image + docker/runpod-bootstrap.sh).
#
# Bump ENV_RECIPE when entrypoint.sh changes HOW the env is built or stored, so
# every volume gets a fresh env instead of reusing an old recipe.
# TORCH_INDEX_URL (the Dockerfile's build arg) selects the torch wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_RECIPE=2
# The venv links to this interpreter, so its exact path is part of the key:
# the same Python minor version at another path is another env.
python=$(readlink -f "$(command -v python3)")

echo "recipe=$ENV_RECIPE"
echo "python=$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "python_path=$python"
echo "platform=$(uname -m)"
# shellcheck disable=SC1091 # the running system's file, not part of this repo
echo "os=$(. /etc/os-release && echo "${ID}${VERSION_ID}")"
echo "torch_index=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
echo "requirements_sha256=$(sha256sum requirements.txt | cut -d' ' -f1)"
