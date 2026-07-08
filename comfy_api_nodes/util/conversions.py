import base64
import logging
import math
import mimetypes
import uuid
from io import BytesIO

import av
import numpy as np
import torch
from PIL import Image

from comfy.utils import common_upscale
from comfy_api.latest import Input, InputImpl, Types

from ._helpers import mimetype_to_extension


def bytesio_to_image_tensor(image_bytesio: BytesIO, mode: str = "RGBA") -> torch.Tensor:
    """Converts image data from BytesIO to a torch.Tensor.

    Args:
        image_bytesio: BytesIO object containing the image data.
        mode: The PIL mode to convert the image to (e.g., "RGB", "RGBA").

    Returns:
        A torch.Tensor representing the image (1, H, W, C).

    Raises:
        PIL.UnidentifiedImageError: If the image data cannot be identified.
        ValueError: If the specified mode is invalid.
    """
    image = Image.open(image_bytesio)
    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def image_tensor_pair_to_batch(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    Converts a pair of image tensors to a batch tensor.
    If the images are not the same size, the smaller image is resized to
    match the larger image.
    """
    if image1.shape[1:] != image2.shape[1:]:
        image2 = common_upscale(
            image2.movedim(-1, 1),
            image1.shape[2],
            image1.shape[1],
            "bilinear",
            "center",
        ).movedim(1, -1)
    return torch.cat((image1, image2), dim=0)


def tensor_to_bytesio(
    image: torch.Tensor,
    *,
    total_pixels: int | None = 2048 * 2048,
    mime_type: str | None = "image/png",
) -> BytesIO:
    """Converts a torch.Tensor image to a named BytesIO object.

    Args:
        image: Input torch.Tensor image.
        total_pixels: Maximum total pixels for downscaling. If None, no downscaling is performed.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp', 'video/mp4').

    Returns:
        Named BytesIO object containing the image data, with pointer set to the start of buffer.
    """
    if not mime_type:
        mime_type = "image/png"

    pil_image = tensor_to_pil(image, total_pixels=total_pixels)
    img_binary = pil_to_bytesio(pil_image, mime_type=mime_type)
    img_binary.name = f"{uuid.uuid4()}.{mimetype_to_extension(mime_type)}"
    return img_binary


def tensor_to_pil(image: torch.Tensor, total_pixels: int | None = 2048 * 2048) -> Image.Image:
    """Converts a single torch.Tensor image [H, W, C] to a PIL Image, optionally downscaling."""
    if len(image.shape) > 3:
        image = image[0]
    # TODO: remove alpha if not allowed and present
    input_tensor = image.cpu()
    if total_pixels is not None:
        input_tensor = downscale_image_tensor(input_tensor.unsqueeze(0), total_pixels=total_pixels).squeeze()
    image_np = (input_tensor.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    return img


def tensor_to_base64_string(
    image_tensor: torch.Tensor,
    total_pixels: int | None = 2048 * 2048,
    mime_type: str = "image/png",
) -> str:
    """Convert [B, H, W, C] or [H, W, C] tensor to a base64 string.

    Args:
        image_tensor: Input torch.Tensor image.
        total_pixels: Maximum total pixels for downscaling. If None, no downscaling is performed.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp', 'video/mp4').

    Returns:
        Base64 encoded string of the image.
    """
    pil_image = tensor_to_pil(image_tensor, total_pixels=total_pixels)
    img_byte_arr = pil_to_bytesio(pil_image, mime_type=mime_type)
    img_bytes = img_byte_arr.getvalue()
    # Encode bytes to base64 string
    base64_encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_encoded_string


def pil_to_bytesio(img: Image.Image, mime_type: str = "image/png") -> BytesIO:
    """Converts a PIL Image to a BytesIO object."""
    if not mime_type:
        mime_type = "image/png"

    img_byte_arr = BytesIO()
    # Derive PIL format from MIME type (e.g., 'image/png' -> 'PNG')
    pil_format = mime_type.split("/")[-1].upper()
    if pil_format == "JPG":
        pil_format = "JPEG"
    img.save(img_byte_arr, format=pil_format)
    img_byte_arr.seek(0)
    return img_byte_arr


def _compute_downscale_dims(src_w: int, src_h: int, total_pixels: int) -> tuple[int, int] | None:
    """Return downscaled (w, h) with even dims fitting ``total_pixels``, or None if already fits.

    Source aspect ratio is preserved; output may drift by a fraction of a percent because both dimensions
    are rounded down to even values (many  codecs require divisible-by-2).
    """
    pixels = src_w * src_h
    if pixels <= total_pixels:
        return None
    scale = math.sqrt(total_pixels / pixels)
    new_w = max(2, int(src_w * scale))
    new_h = max(2, int(src_h * scale))
    new_w -= new_w % 2
    new_h -= new_h % 2
    return new_w, new_h


def downscale_image_tensor(image: torch.Tensor, total_pixels: int = 1536 * 1024) -> torch.Tensor:
    """Downscale input image tensor to roughly the specified total pixels.

    Output dimensions are rounded down to even values so that the result is guaranteed to fit within ``total_pixels``
    and is compatible with codecs that require even dimensions (e.g. yuv420p).
    """
    samples = image.movedim(-1, 1)
    dims = _compute_downscale_dims(samples.shape[3], samples.shape[2], int(total_pixels))
    if dims is None:
        return image
    new_w, new_h = dims
    return common_upscale(samples, new_w, new_h, "lanczos", "disabled").movedim(1, -1)


def downscale_image_tensor_by_max_side(image: torch.Tensor, *, max_side: int) -> torch.Tensor:
    """Downscale input image tensor so the largest dimension is at most max_side pixels."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[2], samples.shape[3]
    max_dim = max(width, height)
    if max_dim <= max_side:
        return image
    scale_by = max_side / max_dim
    new_width = round(width * scale_by)
    new_height = round(height * scale_by)
    s = common_upscale(samples, new_width, new_height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s


def tensor_to_data_uri(
    image_tensor: torch.Tensor,
    total_pixels: int | None = 2048 * 2048,
    mime_type: str = "image/png",
) -> str:
    """Converts a tensor image to a Data URI string.

    Args:
        image_tensor: Input torch.Tensor image.
        total_pixels: Maximum total pixels for downscaling. If None, no downscaling is performed.
        mime_type: Target image MIME type (e.g., 'image/png', 'image/jpeg', 'image/webp').

    Returns:
        Data URI string (e.g., 'data:image/png;base64,...').
    """
    base64_string = tensor_to_base64_string(image_tensor, total_pixels, mime_type)
    return f"data:{mime_type};base64,{base64_string}"


def audio_to_base64_string(audio: Input.Audio, container_format: str = "mp4", codec_name: str = "aac") -> str:
    """Converts an audio input to a base64 string."""
    sample_rate: int = audio["sample_rate"]
    waveform: torch.Tensor = audio["waveform"]
    audio_data_np = audio_tensor_to_contiguous_ndarray(waveform)
    audio_bytes_io = audio_ndarray_to_bytesio(audio_data_np, sample_rate, container_format, codec_name)
    audio_bytes = audio_bytes_io.getvalue()
    return base64.b64encode(audio_bytes).decode("utf-8")


def video_to_base64_string(
    video: Input.Video,
    container_format: Types.VideoContainer | None = None,
    codec: Types.VideoCodec | None = None,
) -> str:
    """
    Converts a video input to a base64 string.

    Args:
        video: The video input to convert
        container_format: Optional container format to use (defaults to video.container if available)
        codec: Optional codec to use (defaults to video.codec if available)
    """
    video_bytes_io = BytesIO()
    video.save_to(
        video_bytes_io,
        format=container_format or getattr(video, "container", Types.VideoContainer.MP4),
        codec=codec or getattr(video, "codec", Types.VideoCodec.H264),
    )
    video_bytes_io.seek(0)
    return base64.b64encode(video_bytes_io.getvalue()).decode("utf-8")


def audio_ndarray_to_bytesio(
    audio_data_np: np.ndarray,
    sample_rate: int,
    container_format: str = "mp4",
    codec_name: str = "aac",
) -> BytesIO:
    """
    Encodes a numpy array of audio data into a BytesIO object.
    """
    audio_bytes_io = BytesIO()
    with av.open(audio_bytes_io, mode="w", format=container_format) as output_container:
        audio_stream = output_container.add_stream(codec_name, rate=sample_rate)
        frame = av.AudioFrame.from_ndarray(
            audio_data_np,
            format="fltp",
            layout="stereo" if audio_data_np.shape[0] > 1 else "mono",
        )
        frame.sample_rate = sample_rate
        frame.pts = 0

        for packet in audio_stream.encode(frame):
            output_container.mux(packet)

        # Flush stream
        for packet in audio_stream.encode(None):
            output_container.mux(packet)

    audio_bytes_io.seek(0)
    return audio_bytes_io


def audio_tensor_to_contiguous_ndarray(waveform: torch.Tensor) -> np.ndarray:
    """
    Prepares audio waveform for av library by converting to a contiguous numpy array.

    Args:
        waveform: a tensor of shape (1, channels, samples) derived from a Comfy `AUDIO` type.

    Returns:
        Contiguous numpy array of the audio waveform. If the audio was batched,
            the first item is taken.
    """
    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError("Expected waveform tensor shape (1, channels, samples)")

    # If batch is > 1, take first item
    if waveform.shape[0] > 1:
        waveform = waveform[0]

    # Prepare for av: remove batch dim, move to CPU, make contiguous, convert to numpy array
    audio_data_np = waveform.squeeze(0).cpu().contiguous().numpy()
    if audio_data_np.dtype != np.float32:
        audio_data_np = audio_data_np.astype(np.float32)

    return audio_data_np


def audio_input_to_mp3(audio: Input.Audio) -> BytesIO:
    waveform = audio["waveform"].cpu()

    output_buffer = BytesIO()
    output_container = av.open(output_buffer, mode="w", format="mp3")

    out_stream = output_container.add_stream("libmp3lame", rate=audio["sample_rate"])
    out_stream.bit_rate = 320000

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
        format="flt",
        layout="mono" if waveform.shape[0] == 1 else "stereo",
    )
    frame.sample_rate = audio["sample_rate"]
    frame.pts = 0
    output_container.mux(out_stream.encode(frame))
    output_container.mux(out_stream.encode(None))
    output_container.close()
    output_buffer.seek(0)
    return output_buffer


def trim_video(video: Input.Video, duration_sec: float) -> Input.Video:
    """
    Returns a new VideoInput object trimmed from the beginning to the specified duration,
    using av to avoid loading entire video into memory.

    Args:
        video: Input video to trim
        duration_sec: Duration in seconds to keep from the beginning

    Returns:
        VideoFromFile object that owns the output buffer
    """
    output_buffer = BytesIO()
    input_container = None
    output_container = None

    try:
        # Get the stream source - this avoids loading entire video into memory
        # when the source is already a file path
        input_source = video.get_stream_source()

        # Open containers
        input_container = av.open(input_source, mode="r")
        output_container = av.open(output_buffer, mode="w", format="mp4")

        # Set up output streams for re-encoding
        video_stream = None
        audio_stream = None

        for stream in input_container.streams:
            logging.info("Found stream: type=%s, class=%s", stream.type, type(stream))
            if isinstance(stream, av.VideoStream):
                # Create output video stream with same parameters
                video_stream = output_container.add_stream("h264", rate=stream.average_rate)
                video_stream.width = stream.width
                video_stream.height = stream.height
                video_stream.pix_fmt = "yuv420p"
                logging.info("Added video stream: %sx%s @ %sfps", stream.width, stream.height, stream.average_rate)
            elif isinstance(stream, av.AudioStream):
                # Create output audio stream with same parameters
                audio_stream = output_container.add_stream("aac", rate=stream.sample_rate)
                audio_stream.sample_rate = stream.sample_rate
                audio_stream.layout = stream.layout
                logging.info("Added audio stream: %sHz, %s channels", stream.sample_rate, stream.channels)

        # Calculate target frame count that's divisible by 16
        fps = input_container.streams.video[0].average_rate
        estimated_frames = int(duration_sec * fps)
        target_frames = (estimated_frames // 16) * 16  # Round down to nearest multiple of 16

        if target_frames == 0:
            raise ValueError("Video too short: need at least 16 frames for Moonvalley")

        frame_count = 0
        audio_frame_count = 0

        # Decode and re-encode video frames
        if video_stream:
            for frame in input_container.decode(video=0):
                if frame_count >= target_frames:
                    break

                # Re-encode frame
                for packet in video_stream.encode(frame):
                    output_container.mux(packet)
                frame_count += 1

            # Flush encoder
            for packet in video_stream.encode():
                output_container.mux(packet)

            logging.info("Encoded %s video frames (target: %s)", frame_count, target_frames)

        # Decode and re-encode audio frames
        if audio_stream:
            input_container.seek(0)  # Reset to beginning for audio
            for frame in input_container.decode(audio=0):
                if frame.time >= duration_sec:
                    break

                # Re-encode frame
                for packet in audio_stream.encode(frame):
                    output_container.mux(packet)
                audio_frame_count += 1

            # Flush encoder
            for packet in audio_stream.encode():
                output_container.mux(packet)

            logging.info("Encoded %s audio frames", audio_frame_count)

        # Close containers
        output_container.close()
        input_container.close()

        # Return as VideoFromFile using the buffer
        output_buffer.seek(0)
        return InputImpl.VideoFromFile(output_buffer)

    except Exception as e:
        # Clean up on error
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        raise RuntimeError(f"Failed to trim video: {str(e)}") from e


def downscale_video_to_max_pixels(video: Input.Video, max_pixels: int) -> Input.Video:
    """Downscale a video to fit within ``max_pixels`` (w * h), preserving aspect ratio.

    Returns the original video object untouched when it already fits. Preserves frame rate, duration, and audio.
    Aspect ratio is preserved up to a fraction of a percent (even-dim rounding).
    """
    src_w, src_h = video.get_dimensions()
    scale_dims = _compute_downscale_dims(src_w, src_h, max_pixels)
    if scale_dims is None:
        return video
    return _apply_video_scale(video, scale_dims)


def _compute_upscale_dims(src_w: int, src_h: int, total_pixels: int) -> tuple[int, int] | None:
    """Return upscaled (w, h) with even dims meeting at least ``total_pixels``, or None if already large enough.

    Source aspect ratio is preserved; output may drift by a fraction of a percent because both dimensions
    are rounded up to even values (many codecs require divisible-by-2). The result is guaranteed to be at
    least ``total_pixels``.
    """
    pixels = src_w * src_h
    if pixels >= total_pixels:
        return None
    scale = math.sqrt(total_pixels / pixels)
    new_w = math.ceil(src_w * scale)
    new_h = math.ceil(src_h * scale)
    if new_w % 2:
        new_w += 1
    if new_h % 2:
        new_h += 1
    return new_w, new_h


def upscale_video_to_min_pixels(video: Input.Video, min_pixels: int) -> Input.Video:
    """Upscale a video to meet at least ``min_pixels`` (w * h), preserving aspect ratio.

    Returns the original video object untouched when it already meets the minimum. Preserves frame rate,
    duration, and audio. Aspect ratio is preserved up to a fraction of a percent (even-dim rounding).
    Note: upscaling a low-resolution source does not add real detail; downstream model quality may suffer.
    """
    src_w, src_h = video.get_dimensions()
    scale_dims = _compute_upscale_dims(src_w, src_h, min_pixels)
    if scale_dims is None:
        return video
    return _apply_video_scale(video, scale_dims)


def _apply_video_scale(video: Input.Video, scale_dims: tuple[int, int]) -> Input.Video:
    """Re-encode ``video`` scaled to ``scale_dims`` with a single decode/encode pass."""
    out_w, out_h = scale_dims
    output_buffer = BytesIO()
    input_container = None
    output_container = None

    # get_stream_source() is untrimmed, so apply the trim window in this same pass.
    # start_time is normalized (>= 0); duration == 0 means "until the end".
    start_time, duration = video.get_active_trim_window()
    trimming = bool(start_time or duration)

    try:
        input_source = video.get_stream_source()
        input_container = av.open(input_source, mode="r")
        output_container = av.open(output_buffer, mode="w", format="mp4")

        video_stream = output_container.add_stream("h264", rate=video.get_frame_rate())
        video_stream.width = out_w
        video_stream.height = out_h
        video_stream.pix_fmt = "yuv420p"

        audio_stream = None
        for stream in input_container.streams:
            if isinstance(stream, av.AudioStream):
                audio_stream = output_container.add_stream("aac", rate=stream.sample_rate)
                audio_stream.sample_rate = stream.sample_rate
                audio_stream.layout = stream.layout
                break

        in_video = input_container.streams.video[0]
        start_pts = int(start_time / in_video.time_base) if trimming else 0
        end_pts = int((start_time + duration) / in_video.time_base) if duration else None
        if start_pts:
            input_container.seek(start_pts, stream=in_video)

        encoded = 0
        for frame in input_container.decode(video=0):
            if trimming:
                if frame.pts is None or frame.pts < start_pts:
                    continue
                if end_pts is not None and frame.pts >= end_pts:
                    break
            frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
            # Re-wrap as a fresh frame: dropping irregular source timestamps (VFR/AVI/GIF/...)
            # lets the encoder assign clean ones and avoids mp4 muxer errors.
            frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="yuv420p"), format="yuv420p")
            for packet in video_stream.encode(frame):
                output_container.mux(packet)
            encoded += 1
        for packet in video_stream.encode():
            output_container.mux(packet)

        if encoded == 0:
            raise ValueError(
                f"resize produced no frames (start_time={start_time}, duration={duration} "
                "selected nothing from the source)"
            )

        if audio_stream is not None:
            input_container.seek(0)
            for audio_frame in input_container.decode(audio=0):
                if trimming:
                    if audio_frame.time is None or audio_frame.time < start_time:
                        continue
                    if duration and audio_frame.time > start_time + duration:
                        break
                # Carry odd audio time bases the mp4 muxer rejects; reset pts, encoder assigns clean ones (MP3-in-AVI)
                audio_frame.pts = None
                for packet in audio_stream.encode(audio_frame):
                    output_container.mux(packet)
            for packet in audio_stream.encode():
                output_container.mux(packet)

        output_container.close()
        input_container.close()
        output_buffer.seek(0)
        return InputImpl.VideoFromFile(output_buffer)

    except Exception as e:
        if input_container is not None:
            input_container.close()
        if output_container is not None:
            output_container.close()
        raise RuntimeError(f"Failed to resize video: {str(e)}") from e


def _f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format. Copy-paste from nodes_audio.py file."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2**15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2**31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def audio_bytes_to_audio_input(audio_bytes: bytes) -> dict:
    """
    Decode any common audio container from bytes using PyAV and return
    a Comfy AUDIO dict: {"waveform": [1, C, T] float32, "sample_rate": int}.
    """
    with av.open(BytesIO(audio_bytes)) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in response.")
        stream = af.streams.audio[0]

        in_sr = int(stream.codec_context.sample_rate)
        out_sr = in_sr

        frames: list[torch.Tensor] = []
        n_channels = stream.channels or 1

        for frame in af.decode(streams=stream.index):
            arr = frame.to_ndarray()  # shape can be [C, T] or [T, C] or [T]
            buf = torch.from_numpy(arr)
            if buf.ndim == 1:
                buf = buf.unsqueeze(0)  # [T] -> [1, T]
            elif buf.shape[0] != n_channels and buf.shape[-1] == n_channels:
                buf = buf.transpose(0, 1).contiguous()  # [T, C] -> [C, T]
            elif buf.shape[0] != n_channels:
                buf = buf.reshape(-1, n_channels).t().contiguous()  # fallback to [C, T]
            frames.append(buf)

    if not frames:
        raise ValueError("Decoded zero audio frames.")

    wav = torch.cat(frames, dim=1)  # [C, T]
    wav = _f32_pcm(wav)
    return {"waveform": wav.unsqueeze(0).contiguous(), "sample_rate": out_sr}


def resize_mask_to_image(
    mask: torch.Tensor,
    image: torch.Tensor,
    upscale_method="nearest-exact",
    crop="disabled",
    allow_gradient=True,
    add_channel_dim=False,
):
    """Resize mask to be the same dimensions as an image, while maintaining proper format for API calls."""
    _, height, width, _ = image.shape
    mask = mask.unsqueeze(-1)
    mask = mask.movedim(-1, 1)
    mask = common_upscale(mask, width=width, height=height, upscale_method=upscale_method, crop=crop)
    mask = mask.movedim(1, -1)
    if not add_channel_dim:
        mask = mask.squeeze(-1)
    if not allow_gradient:
        mask = (mask > 0.5).float()
    return mask


def convert_mask_to_image(mask: Input.Image) -> torch.Tensor:
    """Make mask have the expected amount of dims (4) and channels (3) to be recognized as an image."""
    mask = mask.unsqueeze(-1)
    return torch.cat([mask] * 3, dim=-1)


def text_filepath_to_base64_string(filepath: str) -> str:
    """Converts a text file to a base64 string."""
    with open(filepath, "rb") as f:
        file_content = f.read()
    return base64.b64encode(file_content).decode("utf-8")


def text_filepath_to_data_uri(filepath: str) -> str:
    """Converts a text file to a data URI."""
    base64_string = text_filepath_to_base64_string(filepath)
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{base64_string}"
