# Sidebar tab for installing the models listed in colab/models.json with colab/downloader.py. Each catalog entry is
#   {"name": "SDXL VAE", "folder": "vae", "url": "https://huggingface.co/...", "filename": "sdxl_vae.safetensors"}
# where folder is relative to ComfyUI's models directory. The catalog is reread on every request.
# A models.json uploaded from the sidebar is saved to user/model-manager/models.json and replaces the default catalog
# until it is reset. Models added by URL from the sidebar are appended to that file, starting from the active catalog.
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
from urllib.parse import urlparse

from aiohttp import web

import folder_paths
from server import PromptServer

COLAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "colab")
CATALOG = os.path.join(COLAB_DIR, "models.json")
USER_CATALOG = os.path.join(folder_paths.get_user_directory(), "model-manager", "models.json")
sys.path.append(COLAB_DIR)
from downloader import Downloader, is_civitai  # noqa: E402

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}

downloader = Downloader(folder_paths.models_dir)
downloads = {}  # model name -> aria2p.Download started from the sidebar


def parse_catalog(models):
    for model in models:
        path = os.path.normpath(os.path.join(folder_paths.models_dir, model["folder"], model["filename"]))
        if os.path.basename(path) != model["filename"] or os.path.commonpath([folder_paths.models_dir, os.path.dirname(path)]) != folder_paths.models_dir:
            raise ValueError(f"{model['name']}: folder and filename must stay inside the models directory")
    return {model["name"]: model for model in models}


def load_catalog():
    with open(USER_CATALOG if os.path.isfile(USER_CATALOG) else CATALOG, encoding="utf-8") as f:
        return parse_catalog(json.load(f))


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


def save_catalog(catalog):
    os.makedirs(os.path.dirname(USER_CATALOG), exist_ok=True)
    with open(USER_CATALOG, "w", encoding="utf-8") as f:
        json.dump(list(catalog.values()), f, indent=2)


async def start_downloads(models):
    # Civitai links are resolved with a blocking curl call, so resolve them in parallel off the event loop.
    started = await asyncio.gather(*(asyncio.to_thread(downloader.download, model["folder"], model["url"], rename=model["filename"])
                                     for model in models))
    for model, new_downloads in zip(models, started):
        if new_downloads:
            downloads[model["name"]] = new_downloads[0]


@PromptServer.instance.routes.get("/model-manager/models")
async def list_models(request):
    models = load_catalog().values()
    return web.json_response(await asyncio.to_thread(lambda: [model_status(model) for model in models]))


@PromptServer.instance.routes.post("/model-manager/install")
async def install_models(request):
    catalog = load_catalog()
    await start_downloads([catalog[name] for name in await request.json()])
    return web.json_response({})


@PromptServer.instance.routes.post("/model-manager/catalog")
async def upload_catalog(request):
    try:
        catalog = parse_catalog(json.loads(await request.text()))
    except (ValueError, KeyError, TypeError) as e:
        return web.Response(status=400, text=f"Invalid models.json: {e!r}")
    save_catalog(catalog)
    return web.json_response({})


@PromptServer.instance.routes.post("/model-manager/add")
async def add_model(request):
    body = await request.json()
    url = body["url"].strip()
    # Hugging Face /blob/ links are the file's web page, /resolve/ is the file itself.
    if "huggingface.co/" in url:
        url = url.replace("/blob/", "/resolve/", 1)
    filename = body.get("filename") or ("" if is_civitai(url) else os.path.basename(urlparse(url).path))
    if not filename:
        return web.Response(status=400, text="Civitai links don't include the filename, enter one.")
    model = {"name": filename, "folder": body["folder"], "url": url, "filename": filename}
    catalog = load_catalog() | {filename: model}
    try:
        parse_catalog(list(catalog.values()))
    except ValueError as e:
        return web.Response(status=400, text=str(e))
    save_catalog(catalog)
    await start_downloads([model])
    return web.json_response({})


@PromptServer.instance.routes.delete("/model-manager/catalog")
async def reset_catalog(request):
    if os.path.isfile(USER_CATALOG):
        os.remove(USER_CATALOG)
    return web.json_response({})
