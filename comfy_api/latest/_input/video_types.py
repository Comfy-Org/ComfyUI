from __future__ import annotations
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Optional, Union, IO
import io
import av
from .._util import VideoContainer, VideoCodec, VideoComponents

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
    ):
        """
        Abstract method to save the video input to a file.

        bit_depth selects the encoded bit depth; None keeps the video's native depth.
        """
        pass

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
