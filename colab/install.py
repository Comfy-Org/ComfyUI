# Installs ComfyUI and the custom nodes pinned in colab/custom_nodes.lock.
# Regenerate the lock and colab/requirements.txt with colab/build.py.
#
# In a Colab cell:
#   !git clone https://github.com/trailbat/ComfyUI.git /content/ComfyUI
#   !python /content/ComfyUI/colab/install.py
# downloads the nodes from the Comfy Registry and installs colab/requirements.txt
# with a single uv install.

import io
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(COLAB_DIR)
CUSTOM_NODES = os.path.join(ROOT, "custom_nodes")
REQUIREMENTS = os.path.join(COLAB_DIR, "requirements.txt")
NODES_LOCK = os.path.join(COLAB_DIR, "custom_nodes.lock")


def run(*cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def fetch_node(node_id, version):
    print(f"+ download {node_id}@{version}", flush=True)
    with urllib.request.urlopen(f"https://api.comfy.org/nodes/{node_id}/versions/{version}") as r:
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


def main():
    with open(NODES_LOCK) as f:
        nodes = [fetch_node(*line.split()) for line in f if line.strip()]

    # aria2 is used by colab/downloader.py.
    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "aria2")
    # Colab ships a CUDA build of torch, the unpinned torch requirement keeps it.
    run("uv", "pip", "install", "--system", "-r", REQUIREMENTS)

    for path in nodes:
        if os.path.isfile(os.path.join(path, "install.py")):
            run(sys.executable, "install.py", cwd=path)
    print(f"Done. Start with: python {os.path.join(ROOT, 'main.py')} --enable-manager")


if __name__ == "__main__":
    main()
