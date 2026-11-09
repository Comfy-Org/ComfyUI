from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import torch
from pydantic import BaseModel, Field, confloat


class LumaIO:
    LUMA_REF = "LUMA_REF"
    LUMA_CONCEPTS = "LUMA_CONCEPTS"
    LUMA_RAY32_KEYFRAME = "LUMA_RAY32_KEYFRAME"


class LumaReference:
    def __init__(self, image: torch.Tensor, weight: float):
        self.image = image
        self.weight = weight

    def create_api_model(self, download_url: str):
        return LumaImageRef(url=download_url, weight=self.weight)


class LumaReferenceChain:
    def __init__(self, first_ref: LumaReference = None):
        self.refs: list[LumaReference] = []
        if first_ref:
            self.refs.append(first_ref)

    def add(self, luma_ref: LumaReference = None):
        self.refs.append(luma_ref)

    def create_api_model(self, download_urls: list[str], max_refs=4):
        if len(self.refs) == 0:
            return None
        api_refs: list[LumaImageRef] = []
        for ref, url in zip(self.refs, download_urls):
            api_ref = LumaImageRef(url=url, weight=ref.weight)
            api_refs.append(api_ref)
        return api_refs

    def clone(self):
        c = LumaReferenceChain()
        for ref in self.refs:
            c.add(ref)
        return c


class LumaConcept:
    def __init__(self, key: str):
        self.key = key


class LumaConceptChain:
    def __init__(self, str_list: list[str] = None):
        self.concepts: list[LumaConcept] = []
        if str_list is not None:
            for c in str_list:
                if c != "None":
                    self.add(LumaConcept(key=c))

    def add(self, concept: LumaConcept):
        self.concepts.append(concept)

    def create_api_model(self):
        if len(self.concepts) == 0:
            return None
        api_concepts: list[LumaConceptObject] = []
        for concept in self.concepts:
            if concept.key == "None":
                continue
            api_concepts.append(LumaConceptObject(key=concept.key))
        if len(api_concepts) == 0:
            return None
        return api_concepts

    def clone(self):
        c = LumaConceptChain()
        for concept in self.concepts:
            c.add(concept)
        return c

    def clone_and_merge(self, other: LumaConceptChain):
        c = self.clone()
        for concept in other.concepts:
            c.add(concept)
        return c


def get_luma_concepts(include_none=False):
    concepts = []
    if include_none:
        concepts.append("None")
    return concepts + [
        "truck_left",
        "pan_right",
        "pedestal_down",
        "low_angle",
        "pedestal_up",
        "selfie",
        "pan_left",
        "roll_right",
        "zoom_in",
        "over_the_shoulder",
        "orbit_right",
        "orbit_left",
        "static",
        "tiny_planet",
        "high_angle",
        "bolt_cam",
        "dolly_zoom",
        "overhead",
        "zoom_out",
        "handheld",
        "roll_left",
        "pov",
        "aerial_drone",
        "push_in",
        "crane_down",
        "truck_right",
        "tilt_down",
        "elevator_doors",
        "tilt_up",
        "ground_level",
        "pull_out",
        "aerial",
        "crane_up",
        "eye_level",
    ]


class LumaImageModel(str, Enum):
    photon_1 = "photon-1"
    photon_flash_1 = "photon-flash-1"


class LumaVideoModel(str, Enum):
    ray_2 = "ray-2"
    ray_flash_2 = "ray-flash-2"
    ray_1_6 = "ray-1-6"


class LumaAspectRatio(str, Enum):
    ratio_1_1 = "1:1"
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_4_3 = "4:3"
    ratio_3_4 = "3:4"
    ratio_21_9 = "21:9"
    ratio_9_21 = "9:21"


class LumaVideoOutputResolution(str, Enum):
    res_540p = "540p"
    res_720p = "720p"
    res_1080p = "1080p"
    res_4k = "4k"


class LumaVideoModelOutputDuration(str, Enum):
    dur_5s = "5s"
    dur_9s = "9s"


class LumaGenerationType(str, Enum):
    video = "video"
    image = "image"


class LumaState(str, Enum):
    queued = "queued"
    dreaming = "dreaming"
    completed = "completed"
    failed = "failed"


class LumaAssets(BaseModel):
    video: Optional[str] = Field(None, description="The URL of the video")
    image: Optional[str] = Field(None, description="The URL of the image")
    progress_video: Optional[str] = Field(None, description="The URL of the progress video")


class LumaImageRef(BaseModel):
    """Used for image gen"""

    url: str = Field(..., description="The URL of the image reference")
    weight: confloat(ge=0.0, le=1.0) = Field(..., description="The weight of the image reference")


class LumaImageReference(BaseModel):
    """Used for video gen"""

    type: Optional[str] = Field("image", description="Input type, defaults to image")
    url: str = Field(..., description="The URL of the image")


class LumaModifyImageRef(BaseModel):
    url: str = Field(..., description="The URL of the image reference")
    weight: confloat(ge=0.0, le=1.0) = Field(..., description="The weight of the image reference")


class LumaCharacterRef(BaseModel):
    identity0: LumaImageIdentity = Field(..., description="The image identity object")


class LumaImageIdentity(BaseModel):
    images: list[str] = Field(..., description="The URLs of the image identity")


class LumaGenerationReference(BaseModel):
    type: str = Field("generation", description="Input type, defaults to generation")
    id: str = Field(..., description="The ID of the generation")


class LumaKeyframes(BaseModel):
    frame0: Optional[Union[LumaImageReference, LumaGenerationReference]] = Field(None, description="")
    frame1: Optional[Union[LumaImageReference, LumaGenerationReference]] = Field(None, description="")


class LumaConceptObject(BaseModel):
    key: str = Field(..., description="Camera Concept name")


class LumaImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt of the generation")
    model: LumaImageModel = Field(LumaImageModel.photon_1, description="The image model used for the generation")
    aspect_ratio: Optional[LumaAspectRatio] = Field(LumaAspectRatio.ratio_16_9)
    image_ref: Optional[list[LumaImageRef]] = Field(None, description="List of image reference objects")
    style_ref: Optional[list[LumaImageRef]] = Field(None, description="List of style reference objects")
    character_ref: Optional[LumaCharacterRef] = Field(None, description="The image identity object")
    modify_image_ref: Optional[LumaModifyImageRef] = Field(None, description="The modify image reference object")


class LumaGenerationRequest(BaseModel):
    prompt: str = Field(..., description="The prompt of the generation")
    model: LumaVideoModel = Field(LumaVideoModel.ray_2, description="The video model used for the generation")
    duration: Optional[LumaVideoModelOutputDuration] = Field(None, description="The duration of the generation")
    aspect_ratio: Optional[LumaAspectRatio] = Field(None, description="The aspect ratio of the generation")
    resolution: Optional[LumaVideoOutputResolution] = Field(None, description="The resolution of the generation")
    loop: Optional[bool] = Field(None, description="Whether to loop the video")
    keyframes: Optional[LumaKeyframes] = Field(None, description="The keyframes of the generation")
    concepts: Optional[list[LumaConceptObject]] = Field(None, description="Camera Concepts to apply to generation")


class LumaGeneration(BaseModel):
    id: str = Field(..., description="The ID of the generation")
    generation_type: LumaGenerationType = Field(..., description="Generation type, image or video")
    state: LumaState = Field(..., description="The state of the generation")
    failure_reason: Optional[str] = Field(None, description="The reason for the state of the generation")
    created_at: str = Field(..., description="The date and time when the generation was created")
    assets: Optional[LumaAssets] = Field(None, description="The assets of the generation")
    model: str = Field(..., description="The model used for the generation")
    request: Union[LumaGenerationRequest, LumaImageGenerationRequest] = Field(...)


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
