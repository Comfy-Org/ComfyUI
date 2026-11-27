import asyncio
import contextlib
import logging
import time
import uuid
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
import torch
from pydantic import BaseModel, Field

from comfy_api.latest import IO, Input, Types
from comfy_execution.utils import get_executing_context

from . import request_logger
from ._helpers import diagnose_connectivity, is_processing_interrupted, sleep_with_interrupt
from .client import (
    ApiEndpoint,
    display_text,
    sync_op,
)
from .common_exceptions import ApiServerError, LocalNetworkError, ProcessingInterrupted
from .conversions import (
    audio_ndarray_to_bytesio,
    audio_tensor_to_contiguous_ndarray,
    tensor_to_bytesio,
)


class UploadRequest(BaseModel):
    file_name: str = Field(..., description="Filename to upload")
    content_type: str | None = Field(
        None,
        description="Mime type of the file. For example: image/png, image/jpeg, video/mp4, etc.",
    )


class UploadResponse(BaseModel):
    download_url: str = Field(..., description="URL to GET uploaded file")
    upload_url: str = Field(..., description="URL to PUT file to upload")


_MAX_PARALLEL_UPLOADS = 4


class _NodeUploads:
    def __init__(self, cls: type[IO.ComfyNode], wait_label: str | None):
        self.slots = asyncio.Semaphore(_MAX_PARALLEL_UPLOADS)
        self.registered = 0
        self.unfinished = 0
        self.sending = 0
        self.done = 0
        self.error: Exception | None = None
        self.status_task = asyncio.create_task(self._show_status(cls, wait_label, time.monotonic()))

    async def _show_status(self, cls: type[IO.ComfyNode], wait_label: str | None, start_ts: float) -> None:
        await asyncio.sleep(1.0)
        while self.error is None and not is_processing_interrupted():
            elapsed = int(time.monotonic() - start_ts)
            if self.registered > 1:
                display_text(
                    cls,
                    f"Done: {self.done}, sending: {self.sending}\nTime elapsed: {elapsed}s",
                    status=f"Uploading {self.registered} files",
                )
            elif wait_label:
                display_text(cls, f"Time elapsed: {elapsed}s", status=wait_label)
            await asyncio.sleep(1.0)


_node_uploads: dict[tuple[asyncio.AbstractEventLoop, str, str], _NodeUploads] = {}


@contextlib.asynccontextmanager
async def _upload_slot(cls: type[IO.ComfyNode], wait_label: str | None):
    context = get_executing_context()
    key = (asyncio.get_running_loop(), context.prompt_id, context.node_id)
    uploads = _node_uploads.get(key)
    if uploads is None:
        uploads = _node_uploads[key] = _NodeUploads(cls, wait_label)
    uploads.registered += 1
    uploads.unfinished += 1
    try:
        async with uploads.slots:
            if uploads.error is None and is_processing_interrupted():
                uploads.error = ProcessingInterrupted("Task cancelled")
            if uploads.error is not None:
                raise uploads.error
            uploads.sending += 1
            try:
                yield
            except Exception as e:
                if uploads.error is None:
                    uploads.error = e
                raise
            finally:
                uploads.sending -= 1
            uploads.done += 1
    finally:
        uploads.unfinished -= 1
        if not uploads.unfinished:
            uploads.status_task.cancel()
            if uploads.error is None:
                del _node_uploads[key]
            else:
                asyncio.get_running_loop().call_soon(_drop_failed_uploads, key, uploads)


def _drop_failed_uploads(key: tuple[asyncio.AbstractEventLoop, str, str], uploads: _NodeUploads) -> None:
    if _node_uploads.get(key) is uploads and not uploads.unfinished:
        del _node_uploads[key]


async def upload_images_to_comfyapi(
    cls: type[IO.ComfyNode],
    image: torch.Tensor | list[torch.Tensor],
    *,
    max_images: int = 8,
    mime_type: str | None = None,
    wait_label: str | None = "Uploading",
    total_pixels: int | None = 2048 * 2048,
) -> list[str]:
    """
    Uploads images to ComfyUI API and returns download URLs.
    To upload multiple images, stack them in the batch dimension first.
    """
    tensors: list[torch.Tensor] = []
    if isinstance(image, list):
        for img in image:
            is_batch = len(img.shape) > 3
            if is_batch:
                tensors.extend(img[i] for i in range(img.shape[0]))
            else:
                tensors.append(img)
    else:
        is_batch = len(image.shape) > 3
        if is_batch:
            tensors.extend(image[i] for i in range(image.shape[0]))
        else:
            tensors.append(image)

    return await asyncio.gather(
        *(
            upload_image_to_comfyapi(
                cls, tensor, mime_type=mime_type, wait_label=wait_label, total_pixels=total_pixels
            )
            for tensor in tensors[:max_images]
        )
    )


async def upload_image_to_comfyapi(
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    *,
    mime_type: str | None = None,
    wait_label: str | None = "Uploading",
    total_pixels: int | None = 2048 * 2048,
) -> str:
    """Uploads a single image to ComfyUI API and returns its download URL."""
    async with _upload_slot(cls, wait_label):
        img_io = await asyncio.to_thread(tensor_to_bytesio, image, total_pixels=total_pixels, mime_type=mime_type)
        return await _create_and_put(cls, img_io, img_io.name, mime_type)


async def upload_audio_to_comfyapi(
    cls: type[IO.ComfyNode],
    audio: Input.Audio,
    *,
    container_format: str = "mp4",
    codec_name: str = "aac",
    mime_type: str = "audio/mp4",
) -> str:
    """
    Uploads a single audio input to ComfyUI API and returns its download URL.
    Encodes the raw waveform into the specified format before uploading.
    """
    async with _upload_slot(cls, "Uploading"):
        audio_bytes_io = await asyncio.to_thread(
            audio_ndarray_to_bytesio,
            audio_tensor_to_contiguous_ndarray(audio["waveform"]),
            audio["sample_rate"],
            container_format,
            codec_name,
        )
        return await _create_and_put(cls, audio_bytes_io, f"{uuid.uuid4()}.{container_format}", mime_type)


async def upload_video_to_comfyapi(
    cls: type[IO.ComfyNode],
    video: Input.Video,
    *,
    container: Types.VideoContainer = Types.VideoContainer.MP4,
    codec: Types.VideoCodec = Types.VideoCodec.H264,
    max_duration: int | None = None,
    wait_label: str | None = "Uploading",
) -> str:
    """
    Uploads a single video to ComfyUI API and returns its download URL.
    Uses the specified container and codec for saving the video before upload.
    """
    async with _upload_slot(cls, wait_label):
        if max_duration is not None:
            try:
                actual_duration = video.get_duration()
                if actual_duration > max_duration:
                    raise ValueError(
                        f"Video duration ({actual_duration:.2f}s) exceeds the maximum allowed ({max_duration}s)."
                    )
            except Exception as e:
                logging.error("Error getting video duration: %s", str(e))
                raise ValueError(f"Could not verify video duration from source: {e}") from e

        upload_mime_type = f"video/{container.value.lower()}"
        filename = f"{uuid.uuid4()}.{container.value.lower()}"

        # Convert VideoInput to BytesIO using specified container/codec
        video_bytes_io = BytesIO()
        try:
            video.save_to(video_bytes_io, format=container, codec=codec)
        except Exception as e:
            raise ValueError(
                f"Could not convert the input video to {container.value.upper()} for upload; "
                f"the file may be corrupted or use an unsupported codec. "
                f"Try re-exporting it as MP4 (H.264). Original error: {e}"
            ) from e
        video_bytes_io.seek(0)

        return await _create_and_put(cls, video_bytes_io, filename, upload_mime_type)


_3D_MIME_TYPES = {
    "glb": "model/gltf-binary",
    "obj": "model/obj",
    "fbx": "application/octet-stream",
}


async def upload_3d_model_to_comfyapi(
    cls: type[IO.ComfyNode],
    model_3d: Types.File3D,
    file_format: str,
) -> str:
    """Uploads a 3D model file to ComfyUI API and returns its download URL."""
    return await upload_file_to_comfyapi(
        cls,
        model_3d.get_data(),
        f"{uuid.uuid4()}.{file_format}",
        _3D_MIME_TYPES.get(file_format, "application/octet-stream"),
    )


async def upload_file_to_comfyapi(
    cls: type[IO.ComfyNode],
    file_bytes_io: BytesIO,
    filename: str,
    upload_mime_type: str | None,
    wait_label: str | None = "Uploading",
) -> str:
    """Uploads a single file to ComfyUI API and returns its download URL."""
    async with _upload_slot(cls, wait_label):
        return await _create_and_put(cls, file_bytes_io, filename, upload_mime_type)


async def _create_and_put(
    cls: type[IO.ComfyNode],
    file_bytes_io: BytesIO,
    filename: str,
    upload_mime_type: str | None,
) -> str:
    if is_processing_interrupted():
        raise ProcessingInterrupted("Task cancelled")
    if upload_mime_type is None:
        request_object = UploadRequest(file_name=filename)
    else:
        request_object = UploadRequest(file_name=filename, content_type=upload_mime_type)
    create_resp = await sync_op(
        cls,
        endpoint=ApiEndpoint(path="/customers/storage", method="POST"),
        data=request_object,
        response_model=UploadResponse,
        final_label_on_success=None,
        monitor_progress=False,
    )
    await upload_file(create_resp.upload_url, file_bytes_io, content_type=upload_mime_type)
    return create_resp.download_url


async def upload_file(
    upload_url: str,
    file: BytesIO | str,
    *,
    content_type: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
) -> None:
    """
    Upload a file to a signed URL (e.g., S3 pre-signed PUT) with retries and interruption.

    Raises:
        ProcessingInterrupted, LocalNetworkError, ApiServerError, Exception
    """
    if isinstance(file, BytesIO):
        with contextlib.suppress(Exception):
            file.seek(0)
        data = file.read()
    elif isinstance(file, str):
        with open(file, "rb") as f:
            data = f.read()
    else:
        raise ValueError("file must be a BytesIO or a filesystem path string")

    headers: dict[str, str] = {}
    skip_auto_headers: set[str] = set()
    if content_type:
        headers["Content-Type"] = content_type
    else:
        skip_auto_headers.add("Content-Type")  # Don't let aiohttp add Content-Type, it can break the signed request

    attempt = 0
    delay = retry_delay
    op_uuid = uuid.uuid4().hex[:8]
    while True:
        attempt += 1
        operation_id = _generate_operation_id("PUT", upload_url, attempt, op_uuid)
        timeout = aiohttp.ClientTimeout(total=None)
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
        sess: aiohttp.ClientSession | None = None
        try:
            request_logger.log_request_response(
                operation_id=operation_id,
                request_method="PUT",
                request_url=upload_url,
                request_headers=headers or None,
                request_params=None,
                request_data=f"[File data {len(data)} bytes]",
            )

            sess = aiohttp.ClientSession(timeout=timeout, trust_env=True)
            req = sess.put(upload_url, data=data, headers=headers, skip_auto_headers=skip_auto_headers)
            req_task = asyncio.create_task(req)

            done, pending = await asyncio.wait({req_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            if monitor_task in done and req_task in pending:
                req_task.cancel()
                raise ProcessingInterrupted("Upload cancelled")

            resp = await req_task
            async with resp:
                if resp.status >= 400:
                    with contextlib.suppress(Exception):
                        try:
                            body = await resp.json()
                        except Exception:
                            body = await resp.text()
                        msg = f"Upload failed with status {resp.status}"
                        request_logger.log_request_response(
                            operation_id=operation_id,
                            request_method="PUT",
                            request_url=upload_url,
                            response_status_code=resp.status,
                            response_headers=dict(resp.headers),
                            response_content=body,
                            error_message=msg,
                        )
                    if resp.status in {408, 429, 500, 502, 503, 504} and attempt <= max_retries:
                        await sleep_with_interrupt(delay, None)
                        delay *= retry_backoff
                        continue
                    raise Exception(f"Failed to upload (HTTP {resp.status}).")
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT",
                    request_url=upload_url,
                    response_status_code=resp.status,
                    response_headers=dict(resp.headers),
                    response_content="File uploaded successfully.",
                )
                return
        except (aiohttp.ClientError, OSError) as e:
            if attempt <= max_retries:
                request_logger.log_request_response(
                    operation_id=operation_id,
                    request_method="PUT",
                    request_url=upload_url,
                    request_headers=headers or None,
                    request_data=f"[File data {len(data)} bytes]",
                    error_message=f"{type(e).__name__}: {str(e)} (will retry)",
                )
                await sleep_with_interrupt(delay, None)
                delay *= retry_backoff
                continue

            diag = await diagnose_connectivity()
            if not diag["internet_accessible"]:
                raise LocalNetworkError(
                    "Unable to connect to the network. Please check your internet connection or proxy settings and try again."
                ) from e
            raise ApiServerError("The API service appears unreachable at this time.") from e
        finally:
            stop_evt.set()
            if monitor_task:
                monitor_task.cancel()
                with contextlib.suppress(Exception):
                    await monitor_task
            if sess:
                with contextlib.suppress(Exception):
                    await sess.close()


def _generate_operation_id(method: str, url: str, attempt: int, op_uuid: str) -> str:
    try:
        parsed = urlparse(url)
        slug = (parsed.path.rsplit("/", 1)[-1] or parsed.netloc or "upload").strip("/").replace("/", "_")
    except Exception:
        slug = "upload"
    return f"{method}_{slug}_{op_uuid}_try{attempt}"
