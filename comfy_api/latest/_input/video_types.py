from __future__ import annotations
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Iterator, Optional, Union, IO
import io
import av
import torch
from .._util import VideoContainer, VideoCodec, VideoComponents, VideoChunk, normalize_crop_rect
from .basic_types import AudioInput


def check_chunk_args(chunk_frames: int, dtype: torch.dtype) -> None:
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be at least 1, got {chunk_frames}")
    if dtype not in (torch.float32, torch.uint8):
        raise ValueError(f"decode_chunked supports torch.float32 and torch.uint8, got {dtype}")


def frames_as(frames: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if frames.dtype == dtype:
        return frames
    if dtype == torch.uint8:
        return frames.float().mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
    if frames.dtype == torch.uint8:
        return frames.float().div_(255.0)
    return frames.to(dtype)


def owned_chunk(frames: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = frames_as(frames, dtype)
    return out.clone(memory_format=torch.contiguous_format) if out is frames else out.contiguous()


class VideoInput(ABC):
    """
    Abstract base class for video input types.
    """

    @abstractmethod
    def get_components(self) -> VideoComponents:
        """
        Abstract method to get the video components (images, audio, and frame rate).

        Returns:
            VideoComponents containing images, audio, and frame rate
        """
        pass

    def decode_chunked(self, chunk_frames: int = 16, dtype: torch.dtype = torch.float32) -> Iterator[VideoChunk]:
        """
        Yield the video's frames in order, up to chunk_frames at a time, so a consumer can process a long
        video without holding every frame in memory.

        Frames are the ones get_components() returns, with the same trim, crop and rotation. dtype is
        torch.float32 (the same values as get_components) or torch.uint8: those values rounded to 0..255, except
        that 8-bit YUV sources convert straight to 8-bit RGB, which is faster and within one level of them.
        Each call starts a new pass over the video.

        Default implementation slices get_components(); subclasses that can decode incrementally override it.
        """
        check_chunk_args(chunk_frames, dtype)
        components = self.get_components()
        images, alpha = components.images, components.alpha
        if images.shape[-1] == 4:  # RGBA frames, as Create Video accepts: the fourth channel is the alpha
            images, alpha = images[..., :3], (images[..., 3:] if alpha is None else alpha)
        if alpha is not None and alpha.ndim == 3:
            alpha = alpha.unsqueeze(-1)
        for start in range(0, images.shape[0], chunk_frames):
            yield VideoChunk(
                images=owned_chunk(images[start:start + chunk_frames], dtype),
                start_frame=start,
                alpha=owned_chunk(alpha[start:start + chunk_frames], dtype) if alpha is not None else None,
            )

    def get_backing_file(self) -> Optional[str]:
        """
        Return the path of the file this video plays unchanged, or None.

        A path is returned only when the file is the whole video as it plays: no trim, no crop, nothing joined,
        generated or held in memory. A consumer that reads the file itself can use it; otherwise it reads the
        frames through decode_chunked and the sound through get_audio.
        """
        return None

    def get_audio(self) -> Optional[AudioInput]:
        """
        Return the audio get_components() would, or None when the video has none.

        Default implementation uses get_components(); subclasses override it to skip decoding frames.
        """
        return self.get_components().audio

    @abstractmethod
    def save_to(
        self,
        path: Union[str, IO[bytes]],
        format: VideoContainer = VideoContainer.AUTO,
        codec: VideoCodec = VideoCodec.AUTO,
        metadata: Optional[dict] = None,
        bit_depth: int | None = None,
        crf: float | None = None,
        color_space: str | None = None,
        preset: str | None = None,
        chroma_subsampling: str | None = None,
    ):
        """
        Abstract method to save the video input to a file.

        bit_depth requests the encoded bit depth; unsupported depths use the encoder's highest
        supported depth with a warning. None keeps the video's native depth where supported.
        crf selects the H.264 or AV1 constant rate factor; None uses 18 for H.264 or 24 for AV1 when encoding.
        preset selects the H.264 encoder speed/compression trade-off (e.g. "ultrafast");
        None uses the encoder default. Ignored for other codecs.
        chroma_subsampling selects 444, 422, or 420. None keeps the video's requested or native
        subsampling; unspecified tensor inputs use 444 for ProRes and 420 for H.264/AV1.
        An explicit choice forces re-encoding. Unsupported choices use the closest supported mode
        with a warning.
        color_space="sRGB" selects SDR BT.709/sRGB, "HDR" selects BT.2020/HLG, and "HDR PQ"
        selects BT.2020/PQ. Bit depth is selected independently.
        Tensor-created videos default to sRGB when color_space is None. Loaded videos keep matching recognized native color
        properties; other input pixels must already use the selected color space.
        """
        pass

    def get_color_space(self) -> str:
        """Return the video's color space as sRGB, HDR, HDR PQ, or auto when unspecified."""
        return "auto"

    def get_chroma_subsampling(self) -> str | None:
        """Return requested or native chroma subsampling, or None for codec defaults."""
        return None

    @abstractmethod
    def as_trimmed(
        self,
        start_time: float | None = None,
        duration: float | None = None,
        strict_duration: bool = False,
    ) -> VideoInput | None:
        """
        Create a new VideoInput which is trimmed to have the corresponding start_time and duration

        Returns:
            A new VideoInput, or None if the result would have negative duration
        """
        pass

    def as_cropped(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> VideoInput:
        """
        Create a new VideoInput spatially cropped to the given pixel rectangle.

        The rectangle is clamped to the frame and even-aligned for encoder
        compatibility. An empty or full-frame rectangle returns the input
        unchanged.

        Default implementation materializes the video via get_components();
        subclasses should override with lazier strategies when possible.
        """
        components = self.get_components()
        rect = normalize_crop_rect(
            x, y, width, height, components.images.shape[2], components.images.shape[1]
        )
        if rect is None:
            return self
        from .._input_impl.video_types import VideoFromComponents

        cx, cy, cw, ch = rect
        return VideoFromComponents(
            VideoComponents(
                images=components.images[:, cy:cy + ch, cx:cx + cw, :].clone(),
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=components.metadata,
                alpha=components.alpha[:, cy:cy + ch, cx:cx + cw].clone()
                if components.alpha is not None
                else None,
            ),
            bit_depth=self.get_bit_depth(),
            chroma_subsampling=self.get_chroma_subsampling(),
        )

    def get_stream_source(self) -> Union[str, io.BytesIO]:
        """
        Get a streamable source for the video. This allows processing without
        loading the entire video into memory.

        Returns:
            Either a file path (str) or a BytesIO object that can be opened with av.

        Default implementation creates a BytesIO buffer, but subclasses should
        override this for better performance when possible.
        """
        buffer = io.BytesIO()
        self.save_to(buffer)
        buffer.seek(0)
        return buffer

    def get_active_trim_window(self) -> tuple[float, float]:
        """Return the active trim as ``(start_time, duration)`` in seconds (start_time normalized
        to ``>= 0``; ``duration == 0`` means "until the end"). Default: no trim; trimmable subclasses override.
        """
        return 0.0, 0.0

    # Provide a default implementation, but subclasses can provide optimized versions
    # if possible.
    def get_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions of the video input.

        Returns:
            Tuple of (width, height)
        """
        components = self.get_components()
        return components.images.shape[2], components.images.shape[1]

    def get_bit_depth(self) -> int:
        """
        Returns the bit depth of the video (e.g. 8 or 10).

        Default implementation returns 8; subclasses report their real depth.
        """
        return 8

    def get_duration(self) -> float:
        """
        Returns the duration of the video in seconds.

        Returns:
            Duration in seconds
        """
        components = self.get_components()
        frame_count = components.images.shape[0]
        return float(frame_count / components.frame_rate)

    def get_frame_count(self) -> int:
        """
        Returns the number of frames in the video.

        Default implementation uses :meth:`get_components`, which may require
        loading all frames into memory. File-based implementations should
        override this method and use container/stream metadata instead.

        Returns:
            Total number of frames as an integer.
        """
        return int(self.get_components().images.shape[0])

    def get_frame_rate(self) -> Fraction:
        """
        Returns the frame rate of the video.

        Default implementation materializes the video into memory via
        `get_components()`. Subclasses that can inspect the underlying
        container (e.g. `VideoFromFile`) should override this with a more
        efficient implementation.

        Returns:
            Frame rate as a Fraction.
        """
        return self.get_components().frame_rate

    def get_container_format(self) -> str:
        """
        Returns the container format of the video (e.g., 'mp4', 'mov', 'avi').

        Returns:
            Container format as string
        """
        # Default implementation - subclasses should override for better performance
        source = self.get_stream_source()
        with av.open(source, mode="r") as container:
            return container.format.name
