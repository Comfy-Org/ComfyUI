"""Install on the REMOTE ComfyUI as custom_nodes/remote_companion.py, then restart it.

On Colab, paste this file into a cell under the line
%%writefile /content/ComfyUI/custom_nodes/remote_companion.py

Lets the local editor's Download button for missing models save the file into this server's
model folders, and lists or deletes model files to free space. Downloads start only on an
explicit request and only fetch the requested file.
"""
import asyncio
import os
import shutil
import time
import uuid
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

import folder_paths
from server import PromptServer

NODE_CLASS_MAPPINGS = {}

# Same sources and file types the editor offers downloads for.
ALLOWED_URL_PREFIXES = ("https://huggingface.co/", "https://civitai.com/", "https://civitai.red/")
MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pth", ".pt")
NOT_MODEL_FOLDERS = ("custom_nodes", "configs")
# Used when the editor sends no key, e.g. keys kept in Colab secrets / environment only.
TOKEN_ENV = {"huggingface.co": "HF_TOKEN", "civitai.com": "CIVITAI_API_TOKEN", "civitai.red": "CIVITAI_API_TOKEN"}

# Download records for this server process: small metadata only.
downloads = {}
tasks = {}


def target_path(url, directory, name):
    if not url.startswith(ALLOWED_URL_PREFIXES) or urlsplit(url).username:
        raise ValueError("Only Hugging Face and Civitai https URLs can be downloaded.")
    if directory in NOT_MODEL_FOLDERS or directory not in folder_paths.folder_names_and_paths:
        raise ValueError(f"Unknown model folder {directory!r}.")
    if os.path.basename(name) != name or name.startswith(".") or not name.lower().endswith(MODEL_EXTENSIONS):
        raise ValueError(f"Invalid model file name {name!r}.")
    return os.path.join(folder_paths.get_folder_paths(directory)[0], name)


async def fetch(record, url, path, token):
    part = path + ".part"
    # aiohttp drops Authorization when a redirect leaves the host, so the key never reaches the CDN.
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=None, sock_read=300)) as resp:
                if resp.status in (401, 403):
                    raise PermissionError(f"Access denied (HTTP {resp.status}): {'the key was refused' if token else 'this model needs an access key'}; add it in Remote > Model files and accept the model's terms on its page.")
                resp.raise_for_status()
                record["total"] = resp.content_length
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(part, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 22):
                        f.write(chunk)
                        record["received"] += len(chunk)
        os.replace(part, path)
        record["status"] = "completed"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        record["status"] = "failed"
        record["error"] = str(e) or type(e).__name__
    except asyncio.CancelledError:
        record["status"] = "cancelled"
    finally:
        tasks.pop(record["id"], None)
        if os.path.exists(part):
            os.remove(part)


@PromptServer.instance.routes.post("/remote_execution/models/download")
async def start_download(request):
    body = await request.json()
    try:
        path = target_path(body.get("url", ""), body.get("directory", ""), body.get("name", ""))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    if os.path.exists(path):
        return web.json_response({"error": f"{body['directory']}/{body['name']} already exists on this server."}, status=409)
    for record in downloads.values():
        if record["path"] == path and record["status"] == "downloading":
            return web.json_response(record)
    record = {"id": uuid.uuid4().hex, "url": body["url"], "directory": body["directory"], "name": body["name"], "path": path, "status": "downloading", "received": 0, "total": None, "error": None, "started": time.time()}
    downloads[record["id"]] = record
    # The key is used for this request only; it is not kept or listed.
    token = body.get("token") or os.environ.get(TOKEN_ENV.get(urlsplit(body["url"]).hostname, ""))
    tasks[record["id"]] = asyncio.create_task(fetch(record, body["url"], path, token))
    return web.json_response(record)


@PromptServer.instance.routes.get("/remote_execution/models/downloads")
async def list_downloads(request):
    return web.json_response(list(downloads.values()))


@PromptServer.instance.routes.post("/remote_execution/models/downloads/{id}/cancel")
async def cancel_download(request):
    task = tasks.get(request.match_info["id"])
    if task is not None:
        task.cancel()
    return web.json_response({"cancelled": task is not None})


def model_folders():
    return [d for d in sorted(folder_paths.folder_names_and_paths) if d not in NOT_MODEL_FOLDERS]


@PromptServer.instance.routes.get("/remote_execution/models/files")
async def list_model_files(request):
    files = []
    seen = set()
    for directory in model_folders():
        for name in folder_paths.get_filename_list(directory):
            path = folder_paths.get_full_path(directory, name)
            # Folders can share paths (e.g. models/unet); each file is listed once.
            if path is None or os.path.realpath(path) in seen:
                continue
            seen.add(os.path.realpath(path))
            files.append({"directory": directory, "name": name, "size": os.path.getsize(path)})
    disk = shutil.disk_usage(folder_paths.models_dir)
    return web.json_response({"files": files, "disk": {"total": disk.total, "free": disk.free}})


@PromptServer.instance.routes.post("/remote_execution/models/files/delete")
async def delete_model_file(request):
    body = await request.json()
    directory, name = body.get("directory", ""), body.get("name", "")
    # Only a file the server itself lists as a model can be deleted.
    if directory not in model_folders() or name not in folder_paths.get_filename_list(directory):
        return web.json_response({"error": f"{directory}/{name} is not a model file on this server."}, status=404)
    path = folder_paths.get_full_path(directory, name)
    if any(r["path"] == path and r["status"] == "downloading" for r in downloads.values()):
        return web.json_response({"error": f"{name} is still downloading; cancel it first."}, status=409)
    size = os.path.getsize(path)
    os.remove(path)
    return web.json_response({"deleted": f"{directory}/{name}", "freed": size})
