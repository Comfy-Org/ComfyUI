import os
import base64
import json
import time
import logging
import folder_paths
import glob
import comfy.utils
import uuid
from urllib.parse import urlparse
from typing import Awaitable, Callable
import aiohttp
from aiohttp import web
from PIL import Image
from io import BytesIO
from folder_paths import map_legacy, filter_files_extensions, filter_files_content_types


ALLOWED_MODEL_SOURCES = (
    "https://civitai.com/",
    "https://huggingface.co/",
    "http://localhost:",
)
ALLOWED_MODEL_SUFFIXES = (".safetensors", ".sft")
WHITELISTED_MODEL_URLS = {
    "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt",
    "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth?download=true",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_BULK_MODEL_DOWNLOADS = 200
DownloadProgressCallback = Callable[[int], Awaitable[None]]
DownloadShouldCancel = Callable[[], bool]
DOWNLOAD_PROGRESS_MIN_INTERVAL = 0.25
DOWNLOAD_PROGRESS_MIN_BYTES = 4 * 1024 * 1024


class DownloadCancelledError(Exception):
    pass


class ModelFileManager:
    def __init__(self, prompt_server) -> None:
        self.cache: dict[str, tuple[list[dict], dict[str, float], float]] = {}
        self.prompt_server = prompt_server
        self._cancelled_missing_model_downloads: set[str] = set()

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
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            models = payload.get("models")
            if not isinstance(models, list):
                return web.json_response({"error": "Field 'models' must be a list"}, status=400)
            if len(models) > MAX_BULK_MODEL_DOWNLOADS:
                return web.json_response(
                    {"error": f"Maximum of {MAX_BULK_MODEL_DOWNLOADS} models allowed per request"},
                    status=400,
                )

            target_client_id = str(payload.get("client_id", "")).strip() or None
            batch_id = str(payload.get("batch_id", "")).strip() or uuid.uuid4().hex

            def emit_download_event(
                *,
                task_id: str,
                model_name: str,
                model_directory: str,
                model_url: str,
                status: str,
                bytes_downloaded: int = 0,
                error: str | None = None,
            ) -> None:
                message = {
                    "batch_id": batch_id,
                    "task_id": task_id,
                    "name": model_name,
                    "directory": model_directory,
                    "url": model_url,
                    "status": status,
                    "bytes_downloaded": bytes_downloaded
                }
                if error:
                    message["error"] = error

                self.prompt_server.send_sync("missing_model_download", message, target_client_id)

            results = []
            downloaded = 0
            skipped = 0
            canceled = 0
            failed = 0

            session = self.prompt_server.client_session
            owns_session = False
            if session is None or session.closed:
                timeout = aiohttp.ClientTimeout(total=None)
                session = aiohttp.ClientSession(timeout=timeout)
                owns_session = True

            try:
                for model_entry in models:
                    model_name, model_directory, model_url = _normalize_model_entry(model_entry)
                    task_id = uuid.uuid4().hex
                    self._cancelled_missing_model_downloads.discard(task_id)

                    if not model_name or not model_directory or not model_url:
                        failed += 1
                        error = "Each model must include non-empty name, directory, and url"
                        results.append(
                            {
                                "name": model_name or "",
                                "directory": model_directory or "",
                                "url": model_url or "",
                                "status": "failed",
                                "error": error,
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name or "",
                            model_directory=model_directory or "",
                            model_url=model_url or "",
                            status="failed",
                            error=error,
                        )
                        continue

                    if not _is_http_url(model_url):
                        failed += 1
                        error = "URL must be an absolute HTTP/HTTPS URL"
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "failed",
                                "error": error,
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="failed",
                            error=error,
                        )
                        continue

                    allowed, reason = _is_model_download_allowed(model_name, model_url)
                    if not allowed:
                        failed += 1
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "blocked",
                                "error": reason,
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="blocked",
                            error=reason,
                        )
                        continue

                    try:
                        destination = _resolve_download_destination(model_directory, model_name)
                    except Exception as exc:
                        failed += 1
                        error = str(exc)
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "failed",
                                "error": error,
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="failed",
                            error=error,
                        )
                        continue

                    if os.path.exists(destination):
                        skipped += 1
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "skipped_existing",
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="skipped_existing",
                            bytes_downloaded=0
                        )
                        continue

                    try:
                        latest_downloaded = 0
                        async def on_progress(bytes_downloaded: int) -> None:
                            nonlocal latest_downloaded
                            latest_downloaded = bytes_downloaded
                            emit_download_event(
                                task_id=task_id,
                                model_name=model_name,
                                model_directory=model_directory,
                                model_url=model_url,
                                status="running",
                                bytes_downloaded=bytes_downloaded
                            )

                        await _download_file(
                            session,
                            model_url,
                            destination,
                            progress_callback=on_progress,
                            should_cancel=lambda: task_id in self._cancelled_missing_model_downloads
                        )
                        downloaded += 1
                        final_size = os.path.getsize(destination) if os.path.exists(destination) else latest_downloaded
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "downloaded",
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="completed",
                            bytes_downloaded=final_size
                        )
                    except DownloadCancelledError:
                        canceled += 1
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "canceled",
                                "error": "Download canceled",
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="canceled",
                            bytes_downloaded=latest_downloaded,
                            error="Download canceled",
                        )
                    except Exception as exc:
                        failed += 1
                        error = str(exc)
                        results.append(
                            {
                                "name": model_name,
                                "directory": model_directory,
                                "url": model_url,
                                "status": "failed",
                                "error": error,
                            }
                        )
                        emit_download_event(
                            task_id=task_id,
                            model_name=model_name,
                            model_directory=model_directory,
                            model_url=model_url,
                            status="failed",
                            error=error
                        )
                    finally:
                        self._cancelled_missing_model_downloads.discard(task_id)
            finally:
                if owns_session:
                    await session.close()

            return web.json_response(
                {
                    "downloaded": downloaded,
                    "skipped": skipped,
                    "canceled": canceled,
                    "failed": failed,
                    "results": results,
                },
                status=200,
            )

        @routes.post("/experiment/models/download_missing/cancel")
        async def cancel_download_missing_model(request: web.Request) -> web.Response:
            try:
                payload = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            task_id = str(payload.get("task_id", "")).strip()
            if not task_id:
                return web.json_response({"error": "Field 'task_id' is required"}, status=400)

            self._cancelled_missing_model_downloads.add(task_id)
            return web.json_response({"ok": True, "task_id": task_id}, status=200)

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

def _is_model_download_allowed(model_name: str, model_url: str) -> tuple[bool, str | None]:
    if model_url in WHITELISTED_MODEL_URLS:
        return True, None

    if not any(model_url.startswith(source) for source in ALLOWED_MODEL_SOURCES):
        return (
            False,
            f"Download not allowed from source '{model_url}'.",
        )

    if not any(model_name.endswith(suffix) for suffix in ALLOWED_MODEL_SUFFIXES):
        return (
            False,
            f"Only allowed suffixes are: {', '.join(ALLOWED_MODEL_SUFFIXES)}",
        )

    return True, None


def _resolve_download_destination(directory: str, model_name: str) -> str:
    if directory not in folder_paths.folder_names_and_paths:
        raise ValueError(f"Unknown model directory '{directory}'")

    model_paths = folder_paths.folder_names_and_paths[directory][0]
    if not model_paths:
        raise ValueError(f"No filesystem paths configured for '{directory}'")

    base_path = os.path.abspath(model_paths[0])
    normalized_name = os.path.normpath(model_name).lstrip("/\\")
    if not normalized_name or normalized_name == ".":
        raise ValueError("Model name cannot be empty")

    destination = os.path.abspath(os.path.join(base_path, normalized_name))
    if os.path.commonpath((base_path, destination)) != base_path:
        raise ValueError("Model path escapes configured model directory")

    destination_parent = os.path.dirname(destination)
    if destination_parent:
        os.makedirs(destination_parent, exist_ok=True)

    return destination


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    destination: str,
    progress_callback: DownloadProgressCallback | None = None,
    should_cancel: DownloadShouldCancel | None = None,
    progress_min_interval: float = DOWNLOAD_PROGRESS_MIN_INTERVAL,
    progress_min_bytes: int = DOWNLOAD_PROGRESS_MIN_BYTES,
) -> None:
    temp_file = f"{destination}.{uuid.uuid4().hex}.temp"
    try:
        if should_cancel is not None and should_cancel():
            raise DownloadCancelledError("Download canceled")
        async with session.get(url, allow_redirects=True) as response:
            response.raise_for_status()
            bytes_downloaded = 0
            if progress_callback is not None:
                await progress_callback(bytes_downloaded)
            last_progress_emit_time = time.monotonic()
            last_progress_emit_bytes = 0
            with open(temp_file, "wb") as file_handle:
                async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                    if should_cancel is not None and should_cancel():
                        raise DownloadCancelledError("Download canceled")
                    if chunk:
                        file_handle.write(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback is not None:
                            now = time.monotonic()
                            should_emit = (
                                bytes_downloaded - last_progress_emit_bytes >= progress_min_bytes
                                or now - last_progress_emit_time >= progress_min_interval
                            )
                            if should_emit:
                                await progress_callback(bytes_downloaded)
                                last_progress_emit_time = now
                                last_progress_emit_bytes = bytes_downloaded
                if progress_callback is not None and bytes_downloaded != last_progress_emit_bytes:
                    await progress_callback(bytes_downloaded)
        os.replace(temp_file, destination)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def _normalize_model_entry(model_entry: object) -> tuple[str | None, str | None, str | None]:
    if not isinstance(model_entry, dict):
        return None, None, None

    model_name = str(model_entry.get("name", "")).strip()
    model_directory = str(model_entry.get("directory", "")).strip()
    model_url = str(model_entry.get("url", "")).strip()
    return model_name, model_directory, model_url


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)
