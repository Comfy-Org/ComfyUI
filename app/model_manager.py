import os
import base64
import json
import time
import logging
import folder_paths
import glob
import comfy.utils
import uuid
import asyncio
import ipaddress
import ntpath
import tempfile
from urllib.parse import urlparse, urljoin
from typing import Awaitable, Callable
import aiohttp
from aiohttp import web
from PIL import Image
from io import BytesIO
from folder_paths import map_legacy, filter_files_extensions, filter_files_content_types


ALLOWED_MODEL_HOSTS = {"civitai.com", "civitai.red", "huggingface.co"}
ALLOWED_MODEL_SUFFIXES = (".safetensors", ".sft")
WHITELISTED_MODEL_URLS = {
    "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt",
    "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth?download=true",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}
MODEL_CDN_DOMAINS = ("huggingface.co", "hf.co", "r2.cloudflarestorage.com")
MODEL_CDN_HOSTS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
# Bounds request validation and result storage, not transfer concurrency.
MAX_BULK_MODEL_DOWNLOADS = 200
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
DownloadProgressCallback = Callable[[int], Awaitable[None]]


class ModelDownloadResolver(aiohttp.ThreadedResolver):
    async def resolve(self, host, port=0, family=0):
        addresses = await super().resolve(host, port, family)
        if any(not ipaddress.ip_address(address["host"]).is_global for address in addresses):
            raise OSError("Model source resolved to a non-public address")
        return addresses


class ModelFileManager:
    def __init__(self, send_event: Callable[[str, dict, str], None]) -> None:
        self.cache: dict[str, tuple[list[dict], dict[str, float], float]] = {}
        self.send_event = send_event
        self._model_downloads: dict[str, tuple[str, str, asyncio.Task]] = {}
        self._download_destinations: set[str] = set()

    def get_cache(self, key: str, default=None) -> tuple[list[dict], dict[str, float], float] | None:
        return self.cache.get(key, default)

    def set_cache(self, key: str, value: tuple[list[dict], dict[str, float], float]):
        self.cache[key] = value

    def clear_cache(self):
        self.cache.clear()

    def add_routes(self, routes):
        # NOTE: This is an experiment to replace `/models`
        @routes.get("/experiment/models")
        async def get_model_folders(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())
            folder_black_list = ["configs", "custom_nodes"]
            output_folders: list[dict] = []
            for folder in model_types:
                if folder in folder_black_list:
                    continue
                output_folders.append({
                    "name": folder,
                    "folders": folder_paths.get_folder_paths(folder),
                    "extensions": sorted(folder_paths.folder_names_and_paths[folder][1]),
                })
            return web.json_response(output_folders)

        # NOTE: This is an experiment to replace `/models/{folder}`
        @routes.get("/experiment/models/{folder}")
        async def get_all_models(request):
            folder = request.match_info.get("folder", None)
            if folder not in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = self.get_model_file_list(folder)
            return web.json_response(files)

        @routes.get("/experiment/models/preview/{folder}/{path_index}/{filename:.*}")
        async def get_model_preview(request):
            folder_name = request.match_info.get("folder", None)
            filename = request.match_info.get("filename", None)

            if folder_name not in folder_paths.folder_names_and_paths:
                return web.Response(status=404)

            # The "{filename:.*}" capture also matches the empty string, which
            # would resolve to the folder itself; reject it explicitly.
            if not filename:
                return web.Response(status=400)

            try:
                path_index = int(request.match_info.get("path_index", None))
            except (TypeError, ValueError):
                return web.Response(status=400)

            folders = folder_paths.folder_names_and_paths[folder_name]
            if path_index < 0 or path_index >= len(folders[0]):
                return web.Response(status=404)
            folder = folders[0][path_index]
            full_filename = os.path.normpath(os.path.join(folder, filename))

            # Prevent path traversal: the requested file must stay within the
            # configured model folder. `filename` is an unrestricted ".*" capture,
            # so values like "../../../../etc/passwd" would otherwise escape it.
            if not folder_paths.is_within_directory(folder, full_filename):
                return web.Response(status=403)

            previews = self.get_model_previews(full_filename)
            default_preview = previews[0] if len(previews) > 0 else None
            if default_preview is None or (isinstance(default_preview, str) and not os.path.isfile(default_preview)):
                return web.Response(status=404)

            # The preview is selected by a glob inside get_model_previews, so a
            # companion file (e.g. "model.preview.png") could itself be a symlink
            # resolving outside the model folder. Re-validate the file actually
            # opened: is_within_directory realpaths it, catching symlink escape.
            if isinstance(default_preview, str) and not folder_paths.is_within_directory(folder, default_preview):
                return web.Response(status=403)

            try:
                with Image.open(default_preview) as img:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="WEBP")
                    img_bytes.seek(0)
                    return web.Response(body=img_bytes.getvalue(), content_type="image/webp")
            except:
                return web.Response(status=404)

        @routes.post("/experiment/models/download_missing")
        async def download_missing_models(request: web.Request) -> web.Response:
            try:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("Body must be an object")
                models = payload.get("models")
                if not isinstance(models, list) or not 0 < len(models) <= MAX_BULK_MODEL_DOWNLOADS:
                    raise ValueError(f"Provide between 1 and {MAX_BULK_MODEL_DOWNLOADS} models")
                client_id = _required_string(payload, "client_id")
                batch_id = _required_string(payload, "batch_id")
                models = [
                    {key: _required_string(model, key) for key in ("name", "directory", "url")}
                    for model in models
                ]
            except (ValueError, TypeError) as exc:
                return web.json_response({"error": str(exc), "message": str(exc)}, status=400)

            models = list({tuple(model.values()): model for model in models}.values())
            results = []
            seen = {}
            connector = aiohttp.TCPConnector(resolver=ModelDownloadResolver(), limit=1)
            async with aiohttp.ClientSession(connector=connector, timeout=DOWNLOAD_TIMEOUT, auto_decompress=False) as session:
                for model in models:
                    task_id = uuid.uuid4().hex
                    destination = None
                    reserved = False
                    bytes_downloaded = 0
                    status = "failed"
                    error = None

                    def emit(status, error=None):
                        self.send_event("missing_model_download", {
                            **model, "task_id": task_id, "batch_id": batch_id,
                            "status": status, "bytes_downloaded": bytes_downloaded,
                            **({"error": error} if error else {}),
                        }, client_id)

                    async def progress(size):
                        nonlocal bytes_downloaded
                        bytes_downloaded = size
                        emit("running")

                    try:
                        allowed, reason = _is_model_download_allowed(model["name"], model["url"])
                        if not allowed:
                            status, error = "blocked", reason
                        else:
                            destination = _resolve_download_destination(model["directory"], model["name"])
                            if destination in seen:
                                status = seen[destination]["status"]
                                error = seen[destination].get("error")
                            elif os.path.isfile(destination):
                                status = "skipped_existing"
                            elif destination in self._download_destinations:
                                error = "This model is already being downloaded"
                            else:
                                self._download_destinations.add(destination)
                                reserved = True
                                task = asyncio.create_task(_download_file(session, model["url"], destination, progress))
                                self._model_downloads[task_id] = (client_id, batch_id, task)
                                try:
                                    await asyncio.shield(task)
                                    status = "downloaded"
                                except asyncio.CancelledError:
                                    if not task.cancelled():
                                        task.cancel()
                                        await asyncio.gather(task, return_exceptions=True)
                                        raise
                                    status = "canceled"
                    except aiohttp.ClientResponseError as exc:
                        error = f"Download failed (HTTP {exc.status})"
                    except asyncio.TimeoutError:
                        error = "Download timed out; try again"
                    except aiohttp.ClientError:
                        error = "Could not download model; check the connection and try again"
                    except (ValueError, OSError) as exc:
                        error = str(exc)
                    finally:
                        self._model_downloads.pop(task_id, None)
                        if reserved:
                            self._download_destinations.discard(destination)

                    result = {**model, "status": status, **({"error": error} if error else {})}
                    results.append(result)
                    if destination is not None:
                        seen[destination] = result
                    emit("completed" if status == "downloaded" else status, error)

            self.clear_cache()
            return web.json_response({
                "downloaded": sum(r["status"] == "downloaded" for r in results),
                "skipped": sum(r["status"] == "skipped_existing" for r in results),
                "canceled": sum(r["status"] == "canceled" for r in results),
                "failed": sum(r["status"] in ("failed", "blocked") for r in results),
                "results": results,
            })

        @routes.post("/experiment/models/download_missing/cancel")
        async def cancel_download_missing_model(request: web.Request) -> web.Response:
            try:
                payload = await request.json()
                task_id = _required_string(payload, "task_id")
                client_id = _required_string(payload, "client_id")
                batch_id = _required_string(payload, "batch_id")
            except (ValueError, TypeError) as exc:
                return web.json_response({"error": str(exc), "message": str(exc)}, status=400)
            download = self._model_downloads.get(task_id)
            if download is None or download[:2] != (client_id, batch_id):
                return web.json_response({"error": "Download task not found", "message": "Download task not found"}, status=404)
            download[2].cancel()
            return web.json_response({"ok": True, "task_id": task_id})

    def get_model_file_list(self, folder_name: str):
        folder_name = map_legacy(folder_name)
        folders = folder_paths.folder_names_and_paths[folder_name]
        output_list: list[dict] = []

        for index, folder in enumerate(folders[0]):
            if not os.path.isdir(folder):
                continue
            out = self.cache_model_file_list_(folder)
            if out is None:
                out = self.recursive_search_models_(folder, index)
                self.set_cache(folder, out)
            output_list.extend(out[0])

        return output_list

    def cache_model_file_list_(self, folder: str):
        model_file_list_cache = self.get_cache(folder)

        if model_file_list_cache is None:
            return None
        if not os.path.isdir(folder):
            return None
        if os.path.getmtime(folder) != model_file_list_cache[1]:
            return None
        for x in model_file_list_cache[1]:
            time_modified = model_file_list_cache[1][x]
            folder = x
            if os.path.getmtime(folder) != time_modified:
                return None

        return model_file_list_cache

    def recursive_search_models_(self, directory: str, pathIndex: int) -> tuple[list[str], dict[str, float], float]:
        if not os.path.isdir(directory):
            return [], {}, time.perf_counter()

        excluded_dir_names = [".git"]
        # TODO use settings
        include_hidden_files = False

        result: list[str] = []
        dirs: dict[str, float] = {}

        for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
            subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
            if not include_hidden_files:
                subdirs[:] = [d for d in subdirs if not d.startswith(".")]
                filenames = [f for f in filenames if not f.startswith(".")]

            filenames = filter_files_extensions(filenames, folder_paths.supported_pt_extensions)

            for file_name in filenames:
                try:
                    full_path = os.path.join(dirpath, file_name)
                    relative_path = os.path.relpath(full_path, directory)

                    # Get file metadata
                    file_info = {
                        "name": relative_path,
                        "pathIndex": pathIndex,
                        "modified": os.path.getmtime(full_path),  # Add modification time
                        "created": os.path.getctime(full_path),   # Add creation time
                        "size": os.path.getsize(full_path)        # Add file size
                    }
                    result.append(file_info)

                except Exception as e:
                    logging.warning(f"Warning: Unable to access {file_name}. Error: {e}. Skipping this file.")
                    continue

            for d in subdirs:
                path: str = os.path.join(dirpath, d)
                try:
                    dirs[path] = os.path.getmtime(path)
                except FileNotFoundError:
                    logging.warning(f"Warning: Unable to access {path}. Skipping this path.")
                    continue

        return result, dirs, time.perf_counter()

    def get_model_previews(self, filepath: str) -> list[str | BytesIO]:
        dirname = os.path.dirname(filepath)

        if not os.path.exists(dirname):
            return []

        basename = os.path.splitext(filepath)[0]
        match_files = glob.glob(f"{basename}.*", recursive=False)
        image_files = filter_files_content_types(match_files, "image")
        safetensors_file = next(filter(lambda x: x.endswith(".safetensors"), match_files), None)
        safetensors_metadata = {}

        result: list[str | BytesIO] = []

        for filename in image_files:
            _basename = os.path.splitext(filename)[0]
            if _basename == basename:
                result.append(filename)
            if _basename == f"{basename}.preview":
                result.append(filename)

        if safetensors_file:
            safetensors_filepath = os.path.join(dirname, safetensors_file)
            header = comfy.utils.safetensors_header(safetensors_filepath, max_size=8*1024*1024)
            if header:
                safetensors_metadata = json.loads(header)
        safetensors_images = safetensors_metadata.get("__metadata__", {}).get("ssmd_cover_images", None)
        if safetensors_images:
            safetensors_images = json.loads(safetensors_images)
            for image in safetensors_images:
                result.append(BytesIO(base64.b64decode(image)))

        return result

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_cache()


def _required_string(payload: object, key: str) -> str:
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{key}' must be a non-empty string")
    return value.strip()


def _validate_model_url(url: str, *, redirect: bool = False) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Model downloads require a public HTTPS URL")
    allowed = host in ALLOWED_MODEL_HOSTS or url in WHITELISTED_MODEL_URLS
    if redirect:
        allowed = allowed or host in MODEL_CDN_HOSTS or any(
            host == domain or host.endswith("." + domain) for domain in MODEL_CDN_DOMAINS
        )
    if not allowed:
        raise ValueError("Model download source is not allowed")


def _is_model_download_allowed(model_name: str, model_url: str) -> tuple[bool, str | None]:
    try:
        _validate_model_url(model_url)
    except ValueError as exc:
        return False, str(exc)
    suffixes = ALLOWED_MODEL_SUFFIXES
    if model_url in WHITELISTED_MODEL_URLS:
        suffixes = (os.path.splitext(urlparse(model_url).path)[1],)
    if not model_name.endswith(suffixes):
        return False, f"Only allowed suffixes are: {', '.join(suffixes)}"
    return True, None


def _resolve_download_destination(directory: str, model_name: str) -> str:
    directory = map_legacy(directory)
    if directory in ("configs", "custom_nodes") or directory not in folder_paths.folder_names_and_paths:
        raise ValueError("Unknown model directory")
    model_paths = folder_paths.get_folder_paths(directory)
    if not model_paths:
        raise ValueError("No paths configured for this model directory")
    if ntpath.isabs(model_name) or ntpath.splitdrive(model_name)[0] or "\x00" in model_name:
        raise ValueError("Model name must be a relative path")
    normalized_name = model_name.replace("\\", "/")
    if ".." in normalized_name.split("/"):
        raise ValueError("Model path escapes configured model directory")
    if not os.path.splitext(normalized_name)[1]:
        raise ValueError("Model name must include a file extension")
    paths = [os.path.join(base, normalized_name) for base in model_paths]
    for base, path in zip(model_paths, paths):
        if not folder_paths.is_within_directory(base, path):
            raise ValueError("Model path escapes configured model directory")
        if os.path.isfile(path):
            return os.path.realpath(path)
    return os.path.realpath(paths[0])


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    destination: str,
    progress_callback: DownloadProgressCallback,
) -> None:
    for hop in range(11):
        _validate_model_url(url, redirect=hop > 0)
        async with session.get(url, allow_redirects=False) as response:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Model redirect has no destination")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            if response.status != 200 or response.content_type in ("text/html", "application/json"):
                raise ValueError("Source did not return a model file")
            parent = os.path.dirname(destination)
            os.makedirs(parent, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=parent, suffix=".temp", delete=False) as file_handle:
                temp_file = file_handle.name
                try:
                    downloaded = 0
                    last_emit = time.monotonic()
                    await progress_callback(0)
                    async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                        file_handle.write(chunk)
                        downloaded += len(chunk)
                        if time.monotonic() - last_emit >= 0.25:
                            await progress_callback(downloaded)
                            last_emit = time.monotonic()
                    if downloaded == 0:
                        raise ValueError("Source returned an empty file")
                    file_handle.close()
                    await progress_callback(downloaded)
                    # Publishing with a hard link fails if another writer installed the file.
                    os.link(temp_file, destination)
                finally:
                    file_handle.close()
                    os.unlink(temp_file)
            return
    raise ValueError("Too many model download redirects")
