from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional

import comfy.utils
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


# Ordered spatial operations applied to video frames. Kept as plain tuples so a
# resize chain stays inspectable and is never collapsed into a single step.
# ("scale", width, height, upscale_method, crop): one comfy.utils.common_upscale call.
# ("crop", x, y, width, height): exact pixel slice, no even alignment.
# ("crop_aligned", x, y, width, height): as_cropped() request, resolved against the
#     frame with normalize_crop_rect() so the existing even-grid contract is kept.


def spatial_ops_dimensions(
    ops: list[tuple], width: int, height: int
) -> tuple[int, int]:
    """Return the logical frame size after applying ops to a width x height frame."""
    for op in ops:
        if op[0] == "scale":
            width, height = op[1], op[2]
        elif op[0] == "crop":
            width, height = op[3], op[4]
        else:
            rect = normalize_crop_rect(op[1], op[2], op[3], op[4], width, height)
            if rect is not None:
                width, height = rect[2], rect[3]
    return width, height


def apply_spatial_ops(tensor, ops: list[tuple], is_mask: bool = False):
    """Apply ops to an NHWC image batch, or an NHW mask batch when is_mask is set.

    Uses comfy.utils.common_upscale so the result matches the Resize Image/Mask
    node exactly; common_upscale is batch independent, so per-frame calls give
    the same values as one batched call.
    """
    work = tensor.unsqueeze(1) if is_mask else tensor.movedim(-1, 1)
    for op in ops:
        if op[0] == "scale":
            work = comfy.utils.common_upscale(work, op[1], op[2], op[3], op[4])
            if work.ndim == 3:
                # lanczos drops the channel axis for single channel input
                work = work.unsqueeze(1)
        else:
            if op[0] == "crop_aligned":
                rect = normalize_crop_rect(
                    op[1], op[2], op[3], op[4], work.shape[-1], work.shape[-2]
                )
                if rect is None:
                    continue
                x, y, w, h = rect
            else:
                _, x, y, w, h = op
            work = work[..., y:y + h, x:x + w]
    work = work.squeeze(1) if is_mask else work.movedim(1, -1)
    return work.contiguous()
