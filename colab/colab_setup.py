# Sets up ComfyUI on Colab from a notebook's config cell, see base.ipynb in the ComfyUI_Colab_Notebooks repo:
#   !git clone https://github.com/trailbat/ComfyUI.git /content/ComfyUI
#   import sys; sys.path.append("/content/ComfyUI/colab")
#   import colab_setup
#   colab_setup.setup(CUSTOM_NODES, MODELS_YAML, DOWNLOAD_MODELS, TAILSCALE, GOOGLE_DRIVE)
#   !python /content/ComfyUI/main.py --enable-manager --listen
#
# Reads the GITHUB_TOKEN (for a models.yaml in a private repo), CIVITAI_API_KEY, HUG_TOKEN and TS_AUTHKEY Colab secrets.

import os
import urllib.request

from google.colab import drive, userdata

import install
from catalog import copy_model, parse_catalog
import tailscale

CATALOG = os.path.join(install.COLAB_DIR, "models.yaml")


def secret(name):
    try:
        return userdata.get(name)
    except userdata.SecretNotFoundError:
        return ""


def fetch_catalog(url):
    request = urllib.request.Request(url)
    token = secret("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(request) as r:
        text = r.read().decode("utf-8")
    catalog = parse_catalog(text)
    # The Download Models sidebar tab reads its default catalog from colab/models.yaml.
    with open(CATALOG, "w", encoding="utf-8") as f:
        f.write(text)
    return catalog


def setup(custom_nodes, models_yaml, download_models, tailscale_enabled, google_drive):
    # Colab secrets can't be read from the ComfyUI process, so pass them through the environment.
    for name in ["CIVITAI_API_KEY", "HUG_TOKEN", "TS_AUTHKEY"]:
        os.environ[name] = secret(name)

    # Models with a path in models.yaml are usually copied from Google Drive.
    if google_drive:
        drive.mount("/content/drive")

    catalog = fetch_catalog(models_yaml)
    missing = [name for name in download_models if name not in catalog]
    if missing:
        raise ValueError(f"Not in {models_yaml}: {', '.join(missing)}")

    install.install_aria2()
    # aria2p is installed by install.install_aria2, so downloader can only be imported after it.
    from downloader import Downloader

    models_dir = os.path.join(install.ROOT, "models")
    downloader = Downloader(models_dir)
    copies = []
    for model in (catalog[name] for name in download_models):
        path = os.path.join(models_dir, model["folder"], model["filename"])
        # Skip installed models when the setup cell is run again, aria2 refuses to overwrite a finished download.
        if os.path.isfile(path) and not os.path.exists(path + ".aria2"):
            continue
        if "path" in model:
            copies.append((model["path"], path))
        else:
            downloader.download(model["folder"], model["url"], rename=model["filename"])
    # aria2c runs as its own process, so the downloads continue in the background while setup and ComfyUI run.
    install.install(custom_nodes)

    for source, path in copies:
        print(f"Copying {source}")
        copy_model(source, path)

    if tailscale_enabled:
        tailscale.main()
