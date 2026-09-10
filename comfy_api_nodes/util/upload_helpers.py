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

from . import request_logger
from ._helpers import is_processing_interrupted, sleep_with_interrupt
from .client import (
    ApiEndpoint,
    _diagnose_connectivity,
    _display_time_progress,
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


async def upload_images_to_comfyapi(
    cls: type[IO.ComfyNode],
    image: torch.Tensor | list[torch.Tensor],
    *,
    max_images: int = 8,
    mime_type: str | None = None,
    wait_label: str | None = "Uploading",
    show_batch_index: bool = True,
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

    # if batched, try to upload each file if max_images is greater than 0
    download_urls: list[str] = []
    num_to_upload = min(len(tensors), max_images)
    batch_start_ts = time.monotonic()

    for idx in range(num_to_upload):
        tensor = tensors[idx]
        img_io = tensor_to_bytesio(tensor, total_pixels=total_pixels, mime_type=mime_type)

        effective_label = wait_label
        if wait_label and show_batch_index and num_to_upload > 1:
            effective_label = f"{wait_label} ({idx + 1}/{num_to_upload})"

        url = await upload_file_to_comfyapi(cls, img_io, img_io.name, mime_type, effective_label, batch_start_ts)
        download_urls.append(url)
    return download_urls


async def upload_image_to_comfyapi(
    cls: type[IO.ComfyNode],
    image: torch.Tensor,
    *,
    mime_type: str | None = None,
    wait_label: str | None = "Uploading",
    total_pixels: int | None = 2048 * 2048,
) -> str:
    """Uploads a single image to ComfyUI API and returns its download URL."""
    return (
        await upload_images_to_comfyapi(
            cls,
            image,
            max_images=1,
            mime_type=mime_type,
            wait_label=wait_label,
            show_batch_index=False,
            total_pixels=total_pixels,
        )
    )[0]


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
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, sample_rate, container_format, codec_name)
    return await upload_file_to_comfyapi(cls, audio_bytes_io, f"{uuid.uuid4()}.{container_format}", mime_type)


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

    return await upload_file_to_comfyapi(cls, video_bytes_io, filename, upload_mime_type, wait_label)


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
    progress_origin_ts: float | None = None,
) -> str:
    """Uploads a single file to ComfyUI API and returns its download URL."""
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
    await upload_file(
        cls,
        create_resp.upload_url,
        file_bytes_io,
        content_type=upload_mime_type,
        wait_label=wait_label,
        progress_origin_ts=progress_origin_ts,
    )
    return create_resp.download_url


async def upload_file(
    cls: type[IO.ComfyNode],
    upload_url: str,
    file: BytesIO | str,
    *,
    content_type: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff: float = 2.0,
    wait_label: str | None = None,
    progress_origin_ts: float | None = None,
) -> None:
    """
    Upload a file to a signed URL (e.g., S3 pre-signed PUT) with retries, Comfy progress display, and interruption.

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
    start_ts = progress_origin_ts if progress_origin_ts is not None else time.monotonic()
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
                    if wait_label:
                        _display_time_progress(cls, wait_label, int(time.monotonic() - start_ts), None)
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

            sess = aiohttp.ClientSession(timeout=timeout)
            req = sess.put(upload_url, data=data, headers=headers, skip_auto_headers=skip_auto_headers)
            req_task = asyncio.create_task(req)

            done, pending = await asyncio.wait({req_task, monitor_task}, return_when=asyncio.FIRST_COMPLETED)

            if monitor_task in done and req_task in pending:
                req_task.cancel()
                raise ProcessingInterrupted("Upload cancelled")

            try:
                resp = await req_task
            except asyncio.CancelledError:
                raise ProcessingInterrupted("Upload cancelled") from None

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
                        await sleep_with_interrupt(
                            delay,
                            cls,
                            wait_label,
                            start_ts,
                            None,
                            display_callback=_display_time_progress if wait_label else None,
                        )
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
        except asyncio.CancelledError:
            raise ProcessingInterrupted("Task cancelled") from None
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
                await sleep_with_interrupt(
                    delay,
                    cls,
                    wait_label,
                    start_ts,
                    None,
                    display_callback=_display_time_progress if wait_label else None,
                )
                delay *= retry_backoff
                continue

            diag = await _diagnose_connectivity()
            if not diag["internet_accessible"]:
                raise LocalNetworkError(
                    "Unable to connect to the network. Please check your internet connection and try again."
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
