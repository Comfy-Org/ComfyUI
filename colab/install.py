# Installs ComfyUI's requirements and custom nodes from the Comfy Registry with a single uv install.
# Regenerate colab/custom_nodes.lock with colab/build.py.
#
# Colab notebooks call this through colab/colab_setup.py. Run directly to install every node in the lock:
#   !git clone https://github.com/trailbat/ComfyUI.git /content/ComfyUI
#   !python /content/ComfyUI/colab/install.py

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(COLAB_DIR)
CUSTOM_NODES = os.path.join(ROOT, "custom_nodes")
NODES_LOCK = os.path.join(COLAB_DIR, "custom_nodes.lock")


def run(*cmd, cwd=None, input=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, input=input, text=True, check=True)


def read_lock():
    with open(NODES_LOCK) as f:
        return dict(line.split() for line in f if line.strip())


# Base requirements plus the requirements.txt of each custom node folder, build.py checks the same list resolves.
def requirements(node_paths):
    files = [os.path.join(ROOT, "requirements.txt"), os.path.join(ROOT, "manager_requirements.txt")]
    files += [os.path.join(path, "requirements.txt") for path in node_paths if os.path.isfile(os.path.join(path, "requirements.txt"))]
    return "".join(f"-r {pathlib.Path(file).as_posix()}\n" for file in files) + "aria2p\n"


def fetch_node(node_id, version=None):
    print(f"+ download {node_id}@{version or 'latest'}", flush=True)
    api = f"https://api.comfy.org/nodes/{node_id}/versions/{version}" if version else f"https://api.comfy.org/nodes/{node_id}/install"
    with urllib.request.urlopen(api) as r:
        download_url = json.load(r)["downloadUrl"]
    with urllib.request.urlopen(download_url) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))

    path = os.path.join(CUSTOM_NODES, node_id)
    shutil.rmtree(path, ignore_errors=True)
    archive.extractall(path)
    # ComfyUI-Manager treats a folder with .tracking as a registry install.
    with open(os.path.join(path, ".tracking"), "w", encoding="utf-8") as f:
        f.write("\n".join(archive.namelist()))
    return path


# nodes are registry ids: "*" for every node in the lock, "id@1.2.3" for a version, otherwise the locked or latest version.
def install(nodes):
    lock = read_lock()
    versions = {}
    for node in nodes:
        if node == "*":
            versions |= lock
        else:
            node_id, _, version = node.partition("@")
            versions[node_id] = version or lock.get(node_id)
    paths = [fetch_node(node_id, version) for node_id, version in versions.items()]

    # aria2 is used by colab/downloader.py.
    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "aria2")
    # Colab ships a CUDA build of torch, the unpinned torch requirement keeps it.
    run("uv", "pip", "install", "--system", "-r", "-", input=requirements(paths))

    for path in paths:
        if os.path.isfile(os.path.join(path, "install.py")):
            run(sys.executable, "install.py", cwd=path)


if __name__ == "__main__":
    install(["*"])
    print(f"Done. Start with: python {os.path.join(ROOT, 'main.py')} --enable-manager")
