import asyncio
import contextlib
import uuid
from io import BytesIO
from pathlib import Path
from typing import IO
from urllib.parse import urljoin, urlparse

import aiohttp
import torch
from aiohttp.client_exceptions import ClientError, ContentTypeError

from comfy_api.latest import IO as COMFY_IO
from comfy_api.latest import InputImpl, Types
from folder_paths import get_output_directory

from . import request_logger
from ._helpers import (
    default_base_url,
    get_comfy_api_headers,
    is_processing_interrupted,
    sleep_with_interrupt,
    to_aiohttp_url,
)
from .client import _diagnose_connectivity
from .common_exceptions import ApiServerError, LocalNetworkError, ProcessingInterrupted
from .conversions import bytesio_to_image_tensor

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


async def download_url_to_bytesio(
    url: str,
    dest: BytesIO | IO[bytes] | str | Path | None,
    *,
    timeout: float | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    cls: type[COMFY_IO.ComfyNode] = None,
) -> None:
    """Stream-download a URL to `dest`.

    `dest` must be one of:
      - a BytesIO (rewound to 0 after write),
      - a file-like object opened in binary write mode (must implement .write()),
      - a filesystem path (str | pathlib.Path), which will be opened with 'wb'.

    If `url` starts with `/proxy/`, `cls` must be provided so the URL can be expanded
    to an absolute URL and authentication headers can be applied.

    Raises:
        ProcessingInterrupted, LocalNetworkError, ApiServerError, Exception (HTTP and other errors)
    """
    if not isinstance(dest, (str, Path)) and not hasattr(dest, "write"):
        raise ValueError("dest must be a path (str|Path) or a binary-writable object providing .write().")

    attempt = 0
    delay = retry_delay
    headers: dict[str, str] = {}

    parsed_url = urlparse(url)
    if not parsed_url.scheme and not parsed_url.netloc:  # is URL relative?
        if cls is None:
            raise ValueError("For relative 'cloud' paths, the `cls` parameter is required.")
        url = urljoin(default_base_url().rstrip("/") + "/", url.lstrip("/"))
        headers = get_comfy_api_headers(cls)

    while True:
        attempt += 1
        op_id = _generate_operation_id("GET", url, attempt)
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)

        is_path_sink = isinstance(dest, (str, Path))
        fhandle = None
        session: aiohttp.ClientSession | None = None
        stop_evt: asyncio.Event | None = None
        monitor_task: asyncio.Task | None = None
        req_task: asyncio.Task | None = None

        try:
            with contextlib.suppress(Exception):
                request_logger.log_request_response(operation_id=op_id, request_method="GET", request_url=url)

            session = aiohttp.ClientSession(timeout=timeout_cfg)
            stop_evt = asyncio.Event()

            async def _monitor():
                try:
                    while not stop_evt.is_set():
                        if is_processing_interrupted():
                            return
                        await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    return

            monitor_task = asyncio.create_task(_monitor())

            req_task = asyncio.create_task(session.get(to_aiohttp_url(url), headers=headers))
            done, pending = await asyncio.wait({req_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            if monitor_task in done and req_task in pending:
                req_task.cancel()
                with contextlib.suppress(Exception):
                    await req_task
                raise ProcessingInterrupted("Task cancelled")

            try:
                resp = await req_task
            except asyncio.CancelledError:
                raise ProcessingInterrupted("Task cancelled") from None

            async with resp:
                if resp.status >= 400:
                    with contextlib.suppress(Exception):
                        try:
                            body = await resp.json()
                        except (ContentTypeError, ValueError):
                            text = await resp.text()
                            body = text if len(text) <= 4096 else f"[text {len(text)} bytes]"
                        request_logger.log_request_response(
                            operation_id=op_id,
                            request_method="GET",
                            request_url=url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content=body,
                            error_message=f"HTTP {resp.status}",
                        )

                    if resp.status in _RETRY_STATUS and attempt <= max_retries:
                        await sleep_with_interrupt(delay, cls, None, None, None)
                        delay *= retry_backoff
                        continue
                    raise Exception(f"Failed to download (HTTP {resp.status}).")

                if is_path_sink:
                    p = Path(str(dest))
                    with contextlib.suppress(Exception):
                        p.parent.mkdir(parents=True, exist_ok=True)
                    fhandle = open(p, "wb")
                    sink = fhandle
                else:
                    sink = dest  # BytesIO or file-like

                written = 0
                while True:
                    try:
                        chunk = await asyncio.wait_for(resp.content.read(1024 * 1024), timeout=1.0)
                    except asyncio.TimeoutError:
                        chunk = b""
                    except asyncio.CancelledError:
                        raise ProcessingInterrupted("Task cancelled") from None

                    if is_processing_interrupted():
                        raise ProcessingInterrupted("Task cancelled")

                    if not chunk:
                        if resp.content.at_eof():
                            break
                        continue

                    sink.write(chunk)
                    written += len(chunk)

                if isinstance(dest, BytesIO):
                    with contextlib.suppress(Exception):
                        dest.seek(0)

                request_logger.log_request_response(
                    operation_id=op_id,
                    request_method="GET",
                    request_url=url,
                    response_status_code=resp.status,
                    response_headers=dict(resp.headers),
                    response_content=f"[streamed {written} bytes to dest]",
                )
                return
        except asyncio.CancelledError:
            raise ProcessingInterrupted("Task cancelled") from None
        except (ClientError, OSError) as e:
            if attempt <= max_retries:
                request_logger.log_request_response(
                    operation_id=op_id,
                    request_method="GET",
                    request_url=url,
                    error_message=f"{type(e).__name__}: {str(e)} (will retry)",
                )
                await sleep_with_interrupt(delay, cls, None, None, None)
                delay *= retry_backoff
                continue

            diag = await _diagnose_connectivity()
            if not diag["internet_accessible"]:
                raise LocalNetworkError(
                    "Unable to connect to the network. Please check your internet connection and try again."
                ) from e
            raise ApiServerError("The remote service appears unreachable at this time.") from e
        finally:
            if stop_evt is not None:
                stop_evt.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await monitor_task
            if req_task and not req_task.done():
                req_task.cancel()
                with contextlib.suppress(Exception):
                    await req_task
            if session:
                with contextlib.suppress(Exception):
                    await session.close()
            if fhandle:
                with contextlib.suppress(Exception):
                    fhandle.flush()
                    fhandle.close()


async def download_url_to_image_tensor(
    url: str,
    *,
    timeout: float = None,
    cls: type[COMFY_IO.ComfyNode] = None,
) -> torch.Tensor:
    """Downloads an image from a URL and returns a [B, H, W, C] tensor."""
    result = BytesIO()
    await download_url_to_bytesio(url, result, timeout=timeout, cls=cls)
    return bytesio_to_image_tensor(result)


async def download_url_to_video_output(
    video_url: str,
    *,
    timeout: float = None,
    max_retries: int = 5,
    cls: type[COMFY_IO.ComfyNode] = None,
) -> InputImpl.VideoFromFile:
    """Downloads a video from a URL and returns a `VIDEO` output."""
    result = BytesIO()
    await download_url_to_bytesio(video_url, result, timeout=timeout, max_retries=max_retries, cls=cls)
    return InputImpl.VideoFromFile(result)


async def download_url_as_bytesio(
    url: str,
    *,
    timeout: float = None,
    cls: type[COMFY_IO.ComfyNode] = None,
) -> BytesIO:
    """Downloads content from a URL and returns a new BytesIO (rewound to 0)."""
    result = BytesIO()
    await download_url_to_bytesio(url, result, timeout=timeout, cls=cls)
    return result


def _generate_operation_id(method: str, url: str, attempt: int) -> str:
    try:
        parsed = urlparse(url)
        slug = (parsed.path.rsplit("/", 1)[-1] or parsed.netloc or "download").strip("/").replace("/", "_")
    except Exception:
        slug = "download"
    return f"{method}_{slug}_try{attempt}_{uuid.uuid4().hex[:8]}"


async def download_url_to_file_3d(
    url: str,
    file_format: str,
    *,
    task_id: str | None = None,
    timeout: float | None = None,
    max_retries: int = 5,
    cls: type[COMFY_IO.ComfyNode] = None,
) -> Types.File3D:
    """Downloads a 3D model file from a URL into memory as BytesIO.

    If task_id is provided, also writes the file to disk in the output directory
    for backward compatibility with the old save-to-disk behavior.
    """
    file_format = file_format.lstrip(".").lower()
    data = BytesIO()
    await download_url_to_bytesio(
        url,
        data,
        timeout=timeout,
        max_retries=max_retries,
        cls=cls,
    )

    if task_id is not None:
        # This is only for backward compatability with current behavior when every 3D node is output node
        # All new API nodes should not use "task_id" and instead users should use "SaveGLB" node to save results
        output_dir = Path(get_output_directory())
        output_path = output_dir / f"{task_id}.{file_format}"
        output_path.write_bytes(data.getvalue())
        data.seek(0)

    return Types.File3D(source=data, file_format=file_format)
