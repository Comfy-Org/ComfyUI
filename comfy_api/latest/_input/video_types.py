from __future__ import annotations
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Optional, Union, IO
import io
import av
from .._util import (
    VideoContainer,
    VideoCodec,
    VideoComponents,
    normalize_crop_rect,
    apply_spatial_ops,
)

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
    ):
        """
        Abstract method to save the video input to a file.

        bit_depth selects the encoded bit depth; None keeps the video's native depth.
        crf selects the H.264 or AV1 constant rate factor; None uses the encoder default.
        preset selects the H.264 encoder speed/compression trade-off (e.g. "ultrafast");
        None uses the encoder default. Ignored for other codecs.
        color_space="sRGB" selects SDR BT.709/sRGB, "HDR" selects BT.2020/HLG, and "HDR PQ"
        selects BT.2020/PQ. Bit depth is selected independently.
        Tensor-created videos default to sRGB when color_space is None. Loaded videos keep matching recognized native color
        properties; other input pixels must already use the selected color space.
        """
        pass

    def get_color_space(self) -> str:
        """Return the video's color space as sRGB, HDR, HDR PQ, or auto when unspecified."""
        return "auto"

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
        )

    def as_resized(
        self,
        width: int,
        height: int,
        upscale_method: str,
        crop: str = "disabled",
        post_crop: tuple[int, int, int, int] | None = None,
    ) -> VideoInput:
        """
        Create a new VideoInput scaled to width x height with comfy.utils.common_upscale.

        upscale_method and crop take the same values as the Resize Image/Mask node, so
        the result matches that node's IMAGE output. post_crop is an exact pixel
        rectangle applied after the scale, without the even alignment as_cropped()
        performs; it exists for the node's "scale to multiple" cover crop.

        Only the spatial geometry changes. Frame timing, frame count, frame rate and
        audio are left untouched.

        Default implementation materializes the video via get_components();
        subclasses should override with lazier strategies when possible.
        """
        ops = [("scale", int(width), int(height), upscale_method, crop)]
        if post_crop is not None:
            ops.append(("crop", *(int(value) for value in post_crop)))
        components = self.get_components()
        from .._input_impl.video_types import VideoFromComponents

        # VideoFromComponents takes None for "do not force a color space", which is what
        # "auto" means here; forwarding "auto" itself would be rejected as unsupported.
        # Anything else goes through so an unsupported value still fails there.
        color_space = self.get_color_space()
        if color_space == "auto":
            color_space = None
        alpha = components.alpha
        if alpha is not None:
            # alpha is a MaskInput, so it may arrive as [B, H, W] or with an explicit
            # trailing channel; give it back in the shape it came in.
            if alpha.ndim == 4:
                alpha = apply_spatial_ops(alpha.squeeze(-1), ops, is_mask=True).unsqueeze(-1)
            else:
                alpha = apply_spatial_ops(alpha, ops, is_mask=True)
        return VideoFromComponents(
            VideoComponents(
                images=apply_spatial_ops(components.images, ops),
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=components.metadata,
                alpha=alpha,
            ),
            bit_depth=self.get_bit_depth(),
            color_space=color_space,
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

    def has_pending_frame_edits(self) -> bool:
        """
        Whether this video holds frame edits that its stream source does not contain yet.

        True means get_stream_source() still returns the unedited stream, so callers that
        would otherwise copy those packets have to re-encode instead. It describes this
        video alone; it says nothing about the container, codec, bit depth or quality a
        caller may additionally be asking for.

        Default: no editing, so the stream source can be reused.
        """
        return False

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

    def get_display_dimensions(self) -> tuple[int, int]:
        """
        Returns the dimensions the video is meant to be displayed at, as (width, height).

        This is the size geometry should be computed against. It differs from
        get_dimensions() only for container-rotated sources, where get_dimensions()
        reports the coded size; that inconsistency is deliberately left alone here.
        """
        return self.get_dimensions()

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
