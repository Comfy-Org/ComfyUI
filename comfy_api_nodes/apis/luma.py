from typing import Optional

import torch
from pydantic import BaseModel, Field


class LumaIO:
    LUMA_RAY32_KEYFRAME = "LUMA_RAY32_KEYFRAME"


class LumaImageReference(BaseModel):
    """Used for video gen"""

    type: Optional[str] = Field("image", description="Input type, defaults to image")
    url: str = Field(..., description="The URL of the image")


class Luma2ImageRef(BaseModel):
    url: str | None = None
    data: str | None = None
    media_type: str | None = None
    generation_id: str | None = Field(None, description="reference a prior generation (extend / source reuse)")


class Luma2VideoEdit(BaseModel):
    """Edit controls for Ray 3.2 ``video_edit`` generations."""

    auto_controls: bool | None = Field(None, description="derive a conditioning schedule from the source (recommended)")
    strength: str | None = Field(None, description="'adhere_1' .. 'reimagine_3'; constrained by IO.Combo")


class Luma2VideoOptions(BaseModel):
    """Ray 3.2 ``video`` output settings (text / image / keyframe / edit / extend)."""

    resolution: str | None = Field(None, description="360p | 540p | 720p | 1080p")
    duration: str | None = Field(None, description="5s | 10s")
    loop: bool | None = Field(None)
    start_frame: Luma2ImageRef | None = Field(None)
    end_frame: Luma2ImageRef | None = Field(None)
    keyframes: list[Luma2ImageRef] | None = Field(None)
    keyframe_indexes: list[int] | None = Field(None)
    edit: Luma2VideoEdit | None = Field(None)


class Luma2GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=6000)
    model: str | None = None
    type: str | None = None
    aspect_ratio: str | None = None
    style: str | None = None
    output_format: str | None = None
    web_search: bool | None = None
    image_ref: list[Luma2ImageRef] | None = None
    source: Luma2ImageRef | None = None
    video: Luma2VideoOptions | None = Field(None)


class Luma2Generation(BaseModel):
    id: str | None = None
    type: str | None = None
    state: str | None = None
    model: str | None = None
    created_at: str | None = None
    output: list[LumaImageReference] | None = None
    failure_reason: str | None = None
    failure_code: str | None = None


# --- Ray 3.2 multi-keyframe chain ---

LUMA_KEYFRAME_MODE_FRACTION = "fraction"  # value in [0.0, 1.0] of the output video duration
LUMA_KEYFRAME_MODE_SECONDS = "seconds"  # absolute time, in seconds, from the start of the output


class LumaRay32KeyframeItem:
    """One guide image anchored at a position on the Ray 3.2 output timeline."""

    def __init__(self, image: torch.Tensor, mode: str, value: float):
        self.image = image
        self.mode = mode  # LUMA_KEYFRAME_MODE_FRACTION | LUMA_KEYFRAME_MODE_SECONDS
        self.value = value


class LumaRay32KeyframeChain:
    def __init__(self):
        self.items: list[LumaRay32KeyframeItem] = []

    def add(self, item: LumaRay32KeyframeItem) -> None:
        self.items.append(item)

    def clone(self) -> "LumaRay32KeyframeChain":
        c = LumaRay32KeyframeChain()
        c.items = list(self.items)
        return c
