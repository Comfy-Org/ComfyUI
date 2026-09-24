import os
import logging
import urllib.request
import shutil
import threading
import time
from typing import Optional
import yaml

_model_registry: dict[str, str] = {}
_enabled: bool = False
_active_downloads: dict[str, dict] = {}
_download_progress: dict[str, tuple[int, int]] = {}


def load_registry(config_path: Optional[str] = None) -> None:
    global _model_registry

    if config_path is None:
        search_paths = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model_download_urls.yaml"),
        ]
        config_path = None
        for p in search_paths:
            if os.path.isfile(p):
                config_path = p
                break

    if config_path is None or not os.path.isfile(config_path):
        return

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    for key, url in data.items():
        if isinstance(url, str):
            _model_registry[key.strip()] = url.strip()


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def _download_huggingface_file(repo_and_path: str, dest: str, progress_key: str | None = None) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        if "@" in repo_and_path:
            repo_id, filepath = repo_and_path.split("@", 1)
        else:
            logging.warning(f"[ModelDownloader] Invalid HF path (expected repo_id@filepath): {repo_and_path}")
            return False

        repo_id = repo_id.strip()
        filepath = filepath.strip()
        dest_dir = os.path.dirname(dest)
        os.makedirs(dest_dir, exist_ok=True)

        logging.info(f"[ModelDownloader] Downloading from HuggingFace: {repo_id}  file: {filepath}")

        kwargs = {}
        if progress_key:
            kwargs["tqdm_kwargs"] = {"disable": True}
            kwargs["cache_only"] = False

        downloaded = hf_hub_download(repo_id=repo_id, filename=filepath, **kwargs)
        _copy_to_dest(downloaded, dest)
        if os.path.getsize(downloaded) != os.path.getsize(dest):
            os.remove(dest)
            raise OSError(
                f"Copy to {dest} truncated: got {os.path.getsize(dest)} bytes, "
                f"expected {os.path.getsize(downloaded)}; check free disk space.")

        logging.info(f"[ModelDownloader] Downloaded to: {dest}")
        return True

    except Exception as e:
        logging.warning(f"[ModelDownloader] HF download failed: {e}")
        return False


def _copy_to_dest(src: str, dest: str) -> None:
    if src == dest:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(src):
        shutil.copy2(src, dest)


def _download_direct_url(url: str, dest: str, progress_key: str | None = None) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if "huggingface.co" in url:
        return _download_huggingface_resolve_url(url, dest, progress_key)

    try:
        logging.info(f"[ModelDownloader] Downloading: {url}")

        def _progress(block_num, block_size, total_size):
            if progress_key:
                _download_progress[progress_key] = (block_num * block_size, total_size)

        urllib.request.urlretrieve(url, dest, _progress)

        logging.info(f"[ModelDownloader] Downloaded to: {dest}")
        return True

    except Exception as e:
        logging.warning(f"[ModelDownloader] Direct download failed: {e}")
        return False


def _download_huggingface_resolve_url(url: str, dest: str, progress_key: str | None = None) -> bool:
    try:
        from huggingface_hub import hf_hub_download, get_hf_file_metadata

        parts = url.replace("https://huggingface.co/", "").split("/resolve/")
        if len(parts) != 2:
            logging.warning(f"[ModelDownloader] Unexpected HF resolve URL format: {url}")
            return False

        repo_id = parts[0].rstrip("/")
        full_path = parts[1].split("?")[0]

        path_segments = full_path.split("/", 1)
        if len(path_segments) < 2:
            logging.warning(f"[ModelDownloader] HF resolve URL has no filepath: {url}")
            return False

        filepath = path_segments[1]

        # The expected byte size is needed both for progress reporting and for
        # verifying the download was not truncated (an interrupted connection
        # otherwise silently corrupts the model file).
        meta = None
        try:
            meta = get_hf_file_metadata(url)
        except Exception:
            pass
        if progress_key and meta is not None and meta.size:
            _download_progress[progress_key] = (0, meta.size)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        logging.info(f"[ModelDownloader] Downloading from HuggingFace: {repo_id}  file: {filepath}")

        downloaded = hf_hub_download(repo_id=repo_id, filename=filepath)
        if meta is not None and meta.size is not None and os.path.getsize(downloaded) != meta.size:
            raise OSError(
                f"Cached file size mismatch for {filepath}: got {os.path.getsize(downloaded)} bytes, "
                f"expected {meta.size}; the Hugging Face cache entry is corrupt.")
        _copy_to_dest(downloaded, dest)
        if meta is not None and meta.size is not None and os.path.getsize(dest) != meta.size:
            os.remove(dest)
            raise OSError(
                f"Copy to {dest} truncated: got {os.path.getsize(dest)} bytes, expected {meta.size}; "
                "check free disk space.")

        logging.info(f"[ModelDownloader] Downloaded to: {dest}")
        return True

    except Exception as e:
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower() or "authenticate" in msg.lower():
            logging.warning(f"[ModelDownloader] HF gated model requires login: {e}")
            return False
        logging.warning(f"[ModelDownloader] HF resolve download failed: {e}")
        return False


def _start_progress_monitor(progress_key: str, dest_path: str) -> threading.Thread:
    def _monitor():
        while progress_key in _download_progress:
            if os.path.isfile(dest_path):
                sz = os.path.getsize(dest_path)
                prev = _download_progress.get(progress_key, (0, 0))
                _download_progress[progress_key] = (sz, prev[1])
            time.sleep(1)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


def download_model(folder_name: str, filename: str, direct_url: Optional[str] = None, progress_key: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Download model. Returns (path, error_reason). path is None if failed."""
    if not _enabled:
        return (None, "auto-download is disabled")

    from folder_paths import get_folder_paths

    dirs = get_folder_paths(folder_name)
    if not dirs:
        return (None, f"unknown folder type: {folder_name}")

    dest_dir = dirs[0]
    dest_path = os.path.join(dest_dir, filename)

    if os.path.isfile(dest_path):
        return (dest_path, None)

    if direct_url:
        url = direct_url
    else:
        key = f"{folder_name}/{filename}"
        url = _model_registry.get(key) or _model_registry.get(filename)
        if url is None:
            return (None, f"no download URL configured for {folder_name}/{filename}")

    if progress_key:
        _start_progress_monitor(progress_key, dest_path)

    if "@" in url and not url.startswith(("http://", "https://")):
        ok = _download_huggingface_file(url, dest_path, progress_key)
    else:
        ok = _download_direct_url(url, dest_path, progress_key)

    if ok and os.path.isfile(dest_path):
        return (dest_path, None)

    if "huggingface.co" in url:
        return (None, f"download failed (gated model? set HF_TOKEN): {filename}")
    return (None, f"download failed: could not retrieve {filename}")


MODEL_INPUT_TO_FOLDER: dict[str, str] = {
    "ckpt_name": "checkpoints",
    "lora_name": "loras",
    "vae_name": "vae",
    "control_net_name": "controlnet",
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "clip_name4": "text_encoders",
    "clip_vision_name": "clip_vision",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
    "hypernetwork_name": "hypernetworks",
    "photomaker_model_name": "photomaker",
    "config_name": "configs",
}

_CLASS_TYPE_FOLDER_MAP: dict[str, dict[str, str]] = {
    "UpscaleModelLoader": {"model_name": "upscale_models"},
    "FrameInterpolationModelLoader": {"model_name": "frame_interpolation"},
    "LatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
    "AudioEncoderLoader": {"model_name": "audio_encoders"},
    "ModelPatchLoader": {"name": "model_patches"},
    "MoGeModelLoader": {"model_name": "geometry_estimation"},
    "DepthAnything3Loader": {"model_name": "geometry_estimation"},
    "MediaPipeDetectorLoader": {"model_name": "detection"},
}


def _infer_folder_type(class_type: str, input_name: str) -> str | None:
    if input_name in MODEL_INPUT_TO_FOLDER:
        return MODEL_INPUT_TO_FOLDER[input_name]
    node_map = _CLASS_TYPE_FOLDER_MAP.get(class_type)
    if node_map and input_name in node_map:
        return node_map[input_name]
    return None


def _all_possible_folders(class_type: str, input_name: str) -> list[str]:
    primary = _infer_folder_type(class_type, input_name)
    folders = [primary] if primary else []

    if input_name == "vae_name":
        folders.append("vae_approx")

    return folders


def _lookup_registry(folders: list[str], filename: str) -> str | None:
    for folder in folders:
        url = _model_registry.get(f"{folder}/{filename}")
        if url:
            return url
    return _model_registry.get(filename)


def resolve_prompt_models(prompt: dict) -> dict[str, list[str]]:
    """Scan workflow for model references, download missing ones.

    Returns { 'downloaded': [(folder, filename), ...], 'missing': [(folder, filename), ...] }
    """
    imported: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []

    if not _enabled:
        return {"downloaded": [], "missing": []}

    from folder_paths import get_full_path

    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})

        for input_name, input_value in inputs.items():
            if not isinstance(input_value, str) or not input_value.strip():
                continue

            folders = _all_possible_folders(class_type, input_name)
            if not folders:
                continue

            filename = input_value.strip()

            url = _lookup_registry(folders, filename)
            if url is None:
                continue

            already_exists = False
            for folder in folders:
                if get_full_path(folder, filename) is not None:
                    already_exists = True
                    break

            if already_exists:
                continue

            for folder in folders:
                result, _ = download_model(folder, filename)
                if result is not None:
                    imported.append((folder, filename))
                    break
            else:
                missing.append((folders[0], filename))

    return {"downloaded": imported, "missing": missing}


def add_download_routes(routes) -> None:
    from aiohttp import web

    @routes.post("/api/download-model")
    async def download_model_endpoint(request):
        json_data = await request.json()
        folder_type = json_data.get("folder_type", "").strip()
        filename = json_data.get("filename", "").strip()
        url = json_data.get("url", "").strip() or None

        if not folder_type or not filename:
            return web.json_response({"error": "folder_type and filename required"}, status=400)

        download_key = f"{folder_type}/{filename}"
        if download_key in _active_downloads:
            status = _active_downloads[download_key]
            if status["status"] == "done":
                del _active_downloads[download_key]
                _download_progress.pop(download_key, None)
                return web.json_response({"status": "downloaded"})
            if status["status"] == "failed":
                reason = status.get("reason", "unknown")
                del _active_downloads[download_key]
                _download_progress.pop(download_key, None)
                return web.json_response({"status": "failed", "reason": reason})
            return web.json_response({"status": "downloading", "key": download_key})

        def _run_download():
            try:
                result, error = download_model(folder_type, filename, direct_url=url, progress_key=download_key)
                if result is not None and os.path.isfile(result):
                    _active_downloads[download_key] = {"status": "done"}
                else:
                    _active_downloads[download_key] = {"status": "failed", "reason": error or "unknown error"}
            except Exception as e:
                _active_downloads[download_key] = {"status": "failed", "reason": str(e)}
            _download_progress.pop(download_key, None)

        _active_downloads[download_key] = {"status": "downloading"}
        _download_progress[download_key] = (0, 0)
        t = threading.Thread(target=_run_download, daemon=True)
        t.start()
        return web.json_response({"status": "started", "key": download_key})

    @routes.get("/api/download-status/{key}")
    async def download_status_endpoint(request):
        key = request.match_info["key"]
        status = _active_downloads.get(key)
        progress = _download_progress.get(key)

        if status is None:
            return web.json_response({"status": "unknown"})

        resp = {"status": status["status"]}
        if status["status"] == "failed":
            resp["reason"] = status.get("reason", "unknown")
            del _active_downloads[key]
            _download_progress.pop(key, None)
        elif status["status"] == "done":
            del _active_downloads[key]
            _download_progress.pop(key, None)
        elif progress:
            resp["downloaded_bytes"] = progress[0]
            resp["total_bytes"] = progress[1]

        return web.json_response(resp)
