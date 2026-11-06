"""Container startup; upstream main.py and its CLI remain unchanged."""
import os
from pathlib import Path
import shlex
import sys


def main():
    port = int(os.environ.get("COMFYUI_PORT", "8188"))
    if not 1 <= port <= 65535:
        raise ValueError("COMFYUI_PORT must be between 1 and 65535")
    extra_args = shlex.split(os.environ.get("COMFYUI_ARGS", ""))
    managed = {"--listen", "--port", "--base-directory", "--models-directory",
               "--input-directory", "--output-directory", "--user-directory",
               "--temp-directory", "--database-url"}
    if any(arg.split("=", 1)[0] in managed for arg in extra_args):
        raise ValueError("COMFYUI_ARGS cannot override container paths, listen address or port")

    for directory in ("models", "input", "output", "user", "custom_nodes", "cache"):
        Path("/data", directory).mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "/app/main.py", "--listen", "0.0.0.0", "--port", str(port),
               "--models-directory", "/data/models", "--input-directory", "/data/input",
               "--output-directory", "/data/output", "--user-directory", "/data/user",
               "--temp-directory", "/tmp/comfyui",
               "--database-url", "sqlite:////data/user/comfyui.db",
               "--extra-model-paths-config", "/app/deploy/easypanel/paths.yaml"]
    os.execv(sys.executable, command + extra_args)


if __name__ == "__main__":
    main()
