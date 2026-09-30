# Sidebar tab for installing the models listed in colab/models.json with colab/downloader.py. Each catalog entry is
#   {"name": "SDXL VAE", "folder": "vae", "url": "https://huggingface.co/...", "filename": "sdxl_vae.safetensors"}
# where folder is relative to ComfyUI's models directory. The catalog is reread on every request.
#
# API keys come from the CIVITAI_API_KEY and HUG_TOKEN environment variables, Colab secrets
# can't be read from the ComfyUI process.
#
# Needs aria2c on PATH and aria2p installed, colab/install.py installs both on Colab.
# Locally on Windows: winget install aria2.aria2 and pip install aria2p.

import asyncio
import json
import os
import sys

from aiohttp import web

import folder_paths
from server import PromptServer

COLAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "colab")
CATALOG = os.path.join(COLAB_DIR, "models.json")
sys.path.append(COLAB_DIR)
from downloader import Downloader  # noqa: E402

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}

downloader = Downloader(folder_paths.models_dir)
downloads = {}  # model name -> aria2p.Download started from the sidebar


def load_catalog():
    with open(CATALOG, encoding="utf-8") as f:
        return {model["name"]: model for model in json.load(f)}


def model_status(model):
    status = {"name": model["name"], "folder": model["folder"], "filename": model["filename"]}
    download = downloads.get(model["name"])
    if download is not None:
        download.update()
        if download.status != "complete":
            return status | {"status": download.status, "progress": download.progress, "speed": download.download_speed_string(),
                             "eta": download.eta_string(), "error": download.error_message}

    path = os.path.join(folder_paths.models_dir, model["folder"], model["filename"])
    # aria2 creates the file up front and keeps a .aria2 control file next to it until the download finishes.
    installed = os.path.isfile(path) and not os.path.exists(path + ".aria2")
    return status | {"status": "installed" if installed else "missing"}


@PromptServer.instance.routes.get("/colab_models")
async def list_models(request):
    models = load_catalog().values()
    return web.json_response(await asyncio.to_thread(lambda: [model_status(model) for model in models]))


@PromptServer.instance.routes.post("/colab_models/install")
async def install_models(request):
    catalog = load_catalog()
    models = [catalog[name] for name in await request.json()]
    # Civitai links are resolved with a blocking curl call, so resolve them in parallel off the event loop.
    started = await asyncio.gather(*(asyncio.to_thread(downloader.download, model["folder"], model["url"], rename=model["filename"])
                                     for model in models))
    for model, new_downloads in zip(models, started):
        if new_downloads:
            downloads[model["name"]] = new_downloads[0]
    return web.json_response({})
