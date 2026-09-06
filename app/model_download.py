from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

from aiohttp import ClientSession, ClientTimeout

import folder_paths


ALLOWED_DOWNLOAD_HOSTS = {"huggingface.co", "civitai.com", "civitai.red"}
ALLOWED_DOWNLOAD_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pth", ".pt")
BLOCKED_MODEL_FOLDERS = {"configs", "custom_nodes"}
CHUNK_SIZE = 1024 * 1024

# Bound the network call so a hung remote eventually surfaces an error
# instead of blocking the request handler forever. ``sock_read`` is the
# inter-chunk read timeout, which is the right knob for long downloads:
# a slow-but-progressing transfer keeps making progress, while a stalled
# socket fails predictably.
DOWNLOAD_TIMEOUT = ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=300)

# Maximum number of redirects we follow manually. Hugging Face typically
# redirects ``/resolve/main/...`` to a single CDN URL, so a small budget
# is enough while still preventing redirect loops.
MAX_DOWNLOAD_REDIRECTS = 5

WHITE_LISTED_DOWNLOAD_URLS = {
    "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt",
    "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth?download=true",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


@dataclass(frozen=True)
class ModelDownloadRequest:
    name: str
    url: str
    directory: str


@dataclass(frozen=True)
class ModelDownloadDestination:
    directory: str
    relative_path: str
    full_path: str
    already_exists: bool = False


class ModelDownloadError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def parse_model_download_request(data) -> ModelDownloadRequest:
    if not isinstance(data, dict):
        raise ModelDownloadError("Expected a JSON object.")

    name = data.get("name")
    url = data.get("url")
    directory = data.get("directory")
    if not isinstance(name, str) or not isinstance(url, str) or not isinstance(directory, str):
        raise ModelDownloadError("Missing model name, URL, or directory.")

    name = name.strip()
    url = url.strip()
    directory = directory.strip()
    if not name or not url or not directory:
        raise ModelDownloadError("Model name, URL, and directory are required.")

    if not is_allowed_model_download_url(url):
        raise ModelDownloadError("Model download URL is not allowed.")

    relative_path = normalize_model_relative_path(name)
    if not relative_path.lower().endswith(ALLOWED_DOWNLOAD_SUFFIXES):
        raise ModelDownloadError("Model filename extension is not allowed.")

    return ModelDownloadRequest(name=relative_path, url=url, directory=directory)


def is_allowed_model_download_url(url: str) -> bool:
    """Return True for URLs we are willing to fetch on behalf of the user.

    The same predicate is applied to the user-supplied URL and to every
    redirect target, so SSRF via redirects on an allowed host is contained
    to the same allowlist. Subdomains of allowlisted hosts are accepted
    because Hugging Face and Civitai both serve actual file payloads from
    CDN subdomains (e.g. ``cdn-lfs.huggingface.co``).
    """
    if url in WHITE_LISTED_DOWNLOAD_URLS:
        return True

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme != "https":
        return False

    host = (parsed.hostname or "").lower()
    if not host:
        return False

    for allowed in ALLOWED_DOWNLOAD_HOSTS:
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def normalize_model_relative_path(name: str) -> str:
    if "\x00" in name:
        raise ModelDownloadError("Model filename is invalid.")

    candidate = name.replace("\\", "/")
    if candidate.startswith("/"):
        raise ModelDownloadError("Model filename must be relative.")

    parts = PurePosixPath(candidate).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ModelDownloadError("Model filename must stay inside the model folder.")

    normalized = posixpath.normpath(candidate)
    if normalized in ("", ".") or normalized.startswith("../"):
        raise ModelDownloadError("Model filename must stay inside the model folder.")

    return normalized


def resolve_model_download_destination(request: ModelDownloadRequest) -> ModelDownloadDestination:
    directory = folder_paths.map_legacy(request.directory)
    if directory in BLOCKED_MODEL_FOLDERS or directory not in folder_paths.folder_names_and_paths:
        raise ModelDownloadError("Model directory is not allowed.", status=404)

    existing_path = folder_paths.get_full_path(directory, request.name)
    if existing_path is not None:
        return ModelDownloadDestination(
            directory=directory,
            relative_path=request.name,
            full_path=existing_path,
            already_exists=True,
        )

    destination_root = find_writable_model_root(directory)
    full_path = safe_join(destination_root, request.name)
    return ModelDownloadDestination(
        directory=directory,
        relative_path=request.name,
        full_path=full_path,
    )


def find_writable_model_root(directory: str) -> str:
    for root in folder_paths.get_folder_paths(directory):
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            continue

        if is_writable_directory(root):
            return os.path.abspath(root)

    raise ModelDownloadError("No writable model folder is configured.", status=403)


def is_writable_directory(path: str) -> bool:
    probe = os.path.join(path, ".comfy-download-write-test")
    try:
        with open(probe, "xb"):
            pass
        os.remove(probe)
        return True
    except OSError:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except OSError:
            pass
        return False


def safe_join(root: str, relative_path: str) -> str:
    root = os.path.abspath(root)
    full_path = os.path.abspath(os.path.join(root, relative_path))
    if os.path.commonpath((root, full_path)) != root:
        raise ModelDownloadError("Model filename must stay inside the model folder.")
    return full_path


async def open_model_download_response(session: ClientSession, url: str):
    """GET ``url`` with explicit timeout and an allowlist-checked redirect chain.

    aiohttp follows redirects by default, which would let an allowed host
    redirect to an arbitrary internal target (SSRF). We disable automatic
    following and validate every ``Location`` against the same allowlist
    used for the initial URL.
    """
    current_url = url
    for _ in range(MAX_DOWNLOAD_REDIRECTS + 1):
        response = await session.get(
            current_url,
            allow_redirects=False,
            timeout=DOWNLOAD_TIMEOUT,
        )
        if response.status not in (301, 302, 303, 307, 308):
            return response

        location = response.headers.get("Location", "").strip()
        response.release()
        if not location:
            raise ModelDownloadError("Redirect response missing Location header.", status=502)
        next_url = urljoin(current_url, location)
        if not is_allowed_model_download_url(next_url):
            raise ModelDownloadError("Model download redirect target is not allowed.", status=403)
        current_url = next_url

    raise ModelDownloadError("Too many redirects while downloading model.", status=502)


async def download_model_to_destination(
    session: ClientSession,
    request: ModelDownloadRequest,
    destination: ModelDownloadDestination,
) -> dict:
    if destination.already_exists:
        return download_response("already_exists", request, destination)

    os.makedirs(os.path.dirname(destination.full_path), exist_ok=True)
    partial_path = f"{destination.full_path}.part"

    try:
        fd = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as err:
        raise ModelDownloadError("A download for this model is already in progress.", status=409) from err

    bytes_written = 0
    try:
        with os.fdopen(fd, "wb") as output:
            async with await open_model_download_response(session, request.url) as response:
                if response.status >= 400:
                    raise ModelDownloadError(f"Model download failed with HTTP {response.status}.", status=502)

                expected_size = response.content_length
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    output.write(chunk)
                    bytes_written += len(chunk)

                if expected_size is not None and bytes_written != expected_size:
                    raise ModelDownloadError("Model download ended before all bytes were received.", status=502)

        os.replace(partial_path, destination.full_path)
    except Exception:
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
        raise

    return download_response("downloaded", request, destination, bytes_written)


def download_response(
    status: str,
    request: ModelDownloadRequest,
    destination: ModelDownloadDestination,
    size: int | None = None,
) -> dict:
    response = {
        "status": status,
        "name": request.name,
        "directory": destination.directory,
        "path": destination.full_path,
    }
    if size is not None:
        response["size"] = size
    return response
