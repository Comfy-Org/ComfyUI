from enum import Enum

from pydantic import BaseModel, Field


pixverse_templates = {
    "Microwave": 324641385496960,
    "Suit Swagger": 328545151283968,
    "Anything, Robot": 313358700761536,
    "Subject 3 Fever": 327828816843648,
    "kiss kiss": 315446315336768,
}


class PixverseIO:
    TEMPLATE = "PIXVERSE_TEMPLATE"


class PixverseStatus(int, Enum):
    successful = 1
    generating = 5
    deleted = 6
    contents_moderation = 7
    failed = 8


class PixverseAspectRatio(str, Enum):
    ratio_16_9 = "16:9"
    ratio_4_3 = "4:3"
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_9_16 = "9:16"


class PixverseQuality(str, Enum):
    res_360p = "360p"
    res_540p = "540p"
    res_720p = "720p"
    res_1080p = "1080p"


class PixverseDuration(int, Enum):
    dur_5 = 5
    dur_8 = 8


class PixverseMotionMode(str, Enum):
    normal = "normal"
    fast = "fast"


class PixverseStyle(str, Enum):
    anime = "anime"
    animation_3d = "3d_animation"
    clay = "clay"
    comic = "comic"
    cyberpunk = "cyberpunk"


class PixverseTextVideoRequest(BaseModel):
    aspect_ratio: PixverseAspectRatio = Field(...)
    quality: PixverseQuality = Field(...)
    duration: PixverseDuration = Field(...)
    model: str | None = Field("v3.5")
    motion_mode: PixverseMotionMode | None = Field(PixverseMotionMode.normal)
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    seed: int | None = Field(None)
    style: str | None = Field(None)
    template_id: int | None = Field(None)
    water_mark: bool | None = Field(None)


class PixverseImageVideoRequest(BaseModel):
    quality: PixverseQuality = Field(...)
    duration: PixverseDuration = Field(...)
    img_id: int = Field(...)
    model: str | None = Field("v3.5")
    motion_mode: PixverseMotionMode | None = Field(PixverseMotionMode.normal)
    prompt: str = Field(...)
    negative_prompt: str | None = Field(None)
    seed: int | None = Field(None)
    style: str | None = Field(None)
    template_id: int | None = Field(None)
    water_mark: bool | None = Field(None)


class PixverseTransitionVideoRequest(BaseModel):
    quality: PixverseQuality = Field(...)
    duration: PixverseDuration = Field(...)
    first_frame_img: int = Field(...)
    last_frame_img: int = Field(...)
    model: str | None = Field("v3.5")
    motion_mode: PixverseMotionMode | None = Field(PixverseMotionMode.normal)
    prompt: str = Field(...)
    seed: int | None = Field(None)


class PixverseImgIdResponseObject(BaseModel):
    img_id: int | None = None


class PixverseImageUploadResponse(BaseModel):
    ErrCode: int | None = None
    ErrMsg: str | None = None
    Resp: PixverseImgIdResponseObject | None = Field(None)


class PixverseVideoIdResponseObject(BaseModel):
    video_id: int = Field(...)
    credits: int | None = Field(None)


class PixverseVideoResponse(BaseModel):
    ErrCode: int | None = Field(None)
    ErrMsg: str | None = Field(None)
    Resp: PixverseVideoIdResponseObject | None = Field(None)


class PixverseGenerationStatusResponseObject(BaseModel):
    create_time: str | None = Field(None)
    id: int | None = Field(None)
    modify_time: str | None = Field(None)
    negative_prompt: str | None = Field(None)
    outputHeight: int | None = Field(None)
    outputWidth: int | None = Field(None)
    prompt: str | None = Field(None)
    resolution_ratio: int | None = Field(None)
    seed: int | None = Field(None)
    size: int | None = Field(None)
    status: int | None = Field(None)
    style: str | None = Field(None)
    has_audio: bool | None = Field(None)
    credits: int | None = Field(None)
    url: str | None = Field(None)


class PixverseGenerationStatusResponse(BaseModel):
    ErrCode: int | None = Field(None)
    ErrMsg: str | None = Field(None)
    Resp: PixverseGenerationStatusResponseObject | None = Field(None)


class PixverseV6AspectRatio(str, Enum):
    ratio_16_9 = "16:9"
    ratio_4_3 = "4:3"
    ratio_1_1 = "1:1"
    ratio_3_4 = "3:4"
    ratio_9_16 = "9:16"
    ratio_2_3 = "2:3"
    ratio_3_2 = "3:2"
    ratio_21_9 = "21:9"


class PixverseV6Style(str, Enum):
    none = "none"
    anime = "anime"
    animation_3d = "3d_animation"
    clay = "clay"
    comic = "comic"
    cyberpunk = "cyberpunk"
    realistic = "realistic"


class PixverseReferenceType(str, Enum):
    subject = "subject"
    background = "background"


class PixverseImageReference(BaseModel):
    img_id: int = Field(...)
    ref_name: str = Field(...)
    type: PixverseReferenceType = Field(...)


class PixverseVideoReference(BaseModel):
    ref_name: str = Field(...)
    video_media_id: int | None = Field(None)
    source_video_id: int | None = Field(None)


class PixverseV6BaseRequest(BaseModel):
    model: str = Field("v6")
    prompt: str = Field(...)
    duration: int = Field(...)
    quality: PixverseQuality = Field(...)
    negative_prompt: str | None = Field(None)
    seed: int | None = Field(None)
    style: str | None = Field(None)
    generate_audio_switch: bool | None = Field(None)


class PixverseV6TextVideoRequest(PixverseV6BaseRequest):
    aspect_ratio: PixverseV6AspectRatio = Field(...)
    generate_multi_clip_switch: bool | None = Field(None)


class PixverseV6ImageVideoRequest(PixverseV6BaseRequest):
    img_id: int = Field(...)
    generate_multi_clip_switch: bool | None = Field(None)


class PixverseV6TransitionVideoRequest(PixverseV6BaseRequest):
    first_frame_img: int = Field(...)
    last_frame_img: int = Field(...)


class PixverseV6ExtendVideoRequest(PixverseV6BaseRequest):
    video_media_id: int = Field(...)


class PixverseV6FusionVideoRequest(PixverseV6BaseRequest):
    aspect_ratio: str = Field(...)
    image_references: list[PixverseImageReference] | None = Field(None)
    video_references: list[PixverseVideoReference] | None = Field(None)
    reference_mode: str | None = Field(None)


class PixverseMediaIdResponseObject(BaseModel):
    media_id: int | None = Field(None)
    media_type: str | None = Field(None)
    url: str | None = Field(None)
    width: int | None = Field(None)
    height: int | None = Field(None)


class PixverseMediaUploadResponse(BaseModel):
    ErrCode: int | None = Field(None)
    ErrMsg: str | None = Field(None)
    Resp: PixverseMediaIdResponseObject | None = Field(None)
