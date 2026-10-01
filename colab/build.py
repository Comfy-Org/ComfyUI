# Run locally after installing or updating custom nodes with ComfyUI-Manager:
#   python colab/build.py
# Pins every node in custom_nodes/ to its Comfy Registry version in colab/custom_nodes.lock
# and checks that their combined requirements resolve for Colab. Commit the lock.

import os
import subprocess
import tomllib

from install import CUSTOM_NODES, NODES_LOCK, requirements

COLAB_PYTHON_VERSION = "3.12"


def main():
    lock = []
    paths = []
    for name in sorted(os.listdir(CUSTOM_NODES)):
        path = os.path.join(CUSTOM_NODES, name)
        # model-manager is part of this repo, not a registry node.
        if not os.path.isdir(path) or name in ("__pycache__", "model-manager"):
            continue
        if not os.path.isfile(os.path.join(path, ".tracking")):
            raise RuntimeError(f"{name} was not installed from the Comfy Registry, reinstall it with ComfyUI-Manager.")
        with open(os.path.join(path, "pyproject.toml"), "rb") as f:
            project = tomllib.load(f)["project"]
        # The registry stores versions as full semver, "1.23" is published as "1.23.0".
        version = project["version"] + ".0" * (2 - project["version"].count("."))
        lock.append(f"{project['name']} {version}")
        paths.append(path)

    with open(NODES_LOCK, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(l + "\n" for l in lock))
    print(f"Pinned {len(lock)} custom nodes in {NODES_LOCK}")

    subprocess.run(["uv", "pip", "compile", "--quiet", "--python-platform", "x86_64-manylinux_2_28",
                    "--python-version", COLAB_PYTHON_VERSION, "-"],
                   input=requirements(paths), text=True, stdout=subprocess.DEVNULL, check=True)
    print("Requirements resolve for Colab.")


if __name__ == "__main__":
    main()
