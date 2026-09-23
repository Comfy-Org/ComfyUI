from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional
from .._input import ImageInput, AudioInput, MaskInput

class VideoCodec(str, Enum):
    AUTO = "auto"
    H264 = "h264"
    AV1 = "av1"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of codec names that can be used as node input.
        """
        return [member.value for member in cls]

class VideoContainer(str, Enum):
    AUTO = "auto"
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"

    @classmethod
    def as_input(cls) -> list[str]:
        """
        Returns a list of container names that can be used as node input.
        """
        return [member.value for member in cls]

    @classmethod
    def get_extension(cls, value) -> str:
        """
        Returns the file extension for the container.
        """
        if isinstance(value, str):
            value = cls(value)
        if value == VideoContainer.MP4 or value == VideoContainer.AUTO:
            return "mp4"
        if value == VideoContainer.MKV:
            return "mkv"
        if value == VideoContainer.WEBM:
            return "webm"
        return ""

@dataclass
class VideoComponents:
    """
    Dataclass representing the components of a video.
    """

    images: ImageInput
    frame_rate: Fraction
    audio: Optional[AudioInput] = None
    metadata: Optional[dict] = None
    alpha: Optional[MaskInput] = None


def normalize_crop_rect(
    x: int, y: int, width: int, height: int, source_width: int, source_height: int
) -> Optional[tuple[int, int, int, int]]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return None
    x = max(0, min(int(x), source_width - 1))
    y = max(0, min(int(y), source_height - 1))
    x -= x % 2
    y -= y % 2
    width = min(width, source_width - x)
    height = min(height, source_height - y)
    if x == 0 and y == 0 and width == source_width and height == source_height:
        return None
    width -= width % 2
    height -= height % 2
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height
