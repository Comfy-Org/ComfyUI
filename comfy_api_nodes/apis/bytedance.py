from typing import Literal

from pydantic import BaseModel, Field


class Text2ImageTaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str | None = Field("url")
    size: str | None = Field(None)
    seed: int | None = Field(0, ge=0, le=2147483647)
    guidance_scale: float | None = Field(..., ge=1.0, le=10.0)
    watermark: bool | None = Field(False)


class Seedream4Options(BaseModel):
    max_images: int = Field(15)


class Seedream4TaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str = Field("url")
    image: list[str] | None = Field(None, description="Image URLs")
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    sequential_image_generation: str = Field("disabled")
    sequential_image_generation_options: Seedream4Options = Field(Seedream4Options(max_images=15))
    watermark: bool = Field(False)
    output_format: str | None = None


class ImageTaskCreationResponse(BaseModel):
    model: str = Field(...)
    created: int = Field(..., description="Unix timestamp (in seconds) indicating time when the request was created.")
    data: list = Field([], description="Contains information about the generated image(s).")
    error: dict = Field({}, description="Contains `code` and `message` fields in case of error.")


class TaskTextContent(BaseModel):
    type: str = Field("text")
    text: str = Field(...)


class TaskImageContentUrl(BaseModel):
    url: str = Field(...)


class TaskImageContent(BaseModel):
    type: str = Field("image_url")
    image_url: TaskImageContentUrl = Field(...)
    role: Literal["first_frame", "last_frame", "reference_image"] | None = Field(None)


class TaskVideoContentUrl(BaseModel):
    url: str = Field(...)


class TaskVideoContent(BaseModel):
    type: str = Field("video_url")
    video_url: TaskVideoContentUrl = Field(...)
    role: str = Field("reference_video")


class TaskAudioContentUrl(BaseModel):
    url: str = Field(...)


class TaskAudioContent(BaseModel):
    type: str = Field("audio_url")
    audio_url: TaskAudioContentUrl = Field(...)
    role: str = Field("reference_audio")


class Text2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    content: list[TaskTextContent] = Field(..., min_length=1)
    generate_audio: bool | None = Field(...)


class Image2VideoTaskCreationRequest(BaseModel):
    model: str = Field(...)
    content: list[TaskTextContent | TaskImageContent] = Field(..., min_length=2)
    generate_audio: bool | None = Field(...)


class Seedance2TaskCreationRequest(BaseModel):
    model: str = Field(...)
    content: list[TaskTextContent | TaskImageContent | TaskVideoContent | TaskAudioContent] = Field(..., min_length=1)
    generate_audio: bool | None = Field(None)
    resolution: str | None = Field(None)
    ratio: str | None = Field(None)
    duration: int | None = Field(None, ge=4, le=15)
    seed: int | None = Field(None, ge=0, le=2147483647)
    watermark: bool | None = Field(None)


class TaskCreationResponse(BaseModel):
    id: str = Field(...)


class TaskStatusError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class TaskStatusResult(BaseModel):
    video_url: str = Field(...)


class TaskStatusUsage(BaseModel):
    completion_tokens: int = Field(0)
    total_tokens: int = Field(0)


class TaskStatusResponse(BaseModel):
    id: str = Field(...)
    model: str = Field(...)
    status: Literal["queued", "running", "cancelled", "succeeded", "failed"] = Field(...)
    error: TaskStatusError | None = Field(None)
    content: TaskStatusResult | None = Field(None)
    usage: TaskStatusUsage | None = Field(None)


class GetAssetResponse(BaseModel):
    id: str = Field(...)
    name: str | None = Field(None)
    url: str | None = Field(None)
    asset_type: str = Field(...)
    group_id: str = Field(...)
    status: str = Field(...)
    error: TaskStatusError | None = Field(None)


class SeedanceCreateVisualValidateSessionResponse(BaseModel):
    session_id: str = Field(...)
    h5_link: str = Field(...)


class SeedanceGetVisualValidateSessionResponse(BaseModel):
    session_id: str = Field(...)
    status: str = Field(...)
    group_id: str | None = Field(None)
    error_code: str | None = Field(None)
    error_message: str | None = Field(None)


class SeedanceCreateAssetRequest(BaseModel):
    group_id: str = Field(...)
    url: str = Field(...)
    asset_type: str = Field(...)
    name: str | None = Field(None, max_length=64)
    project_name: str | None = Field(None)


class SeedanceCreateAssetResponse(BaseModel):
    asset_id: str = Field(...)


class SeedanceVirtualLibraryCreateAssetRequest(BaseModel):
    url: str = Field(..., description="Publicly accessible URL of the asset to upload.")
    hash: str = Field(..., description="Dedup key. Re-submitting the same hash returns the existing asset id.")
    asset_type: str | None = Field(None, description="BytePlus asset type. Defaults to Image server-side when omitted.")


# Dollars per 1K tokens, keyed by (model_id, has_video_input, resolution).
SEEDANCE2_PRICE_PER_1K_TOKENS = {
    ("dreamina-seedance-2-0-260128", False, "480p"): 0.007,
    ("dreamina-seedance-2-0-260128", True, "480p"): 0.0043,
    ("dreamina-seedance-2-0-260128", False, "720p"): 0.007,
    ("dreamina-seedance-2-0-260128", True, "720p"): 0.0043,
    ("dreamina-seedance-2-0-260128", False, "1080p"): 0.0077,
    ("dreamina-seedance-2-0-260128", True, "1080p"): 0.0047,
    ("dreamina-seedance-2-0-260128", False, "4k"): 0.004,
    ("dreamina-seedance-2-0-260128", True, "4k"): 0.0024,
    ("dreamina-seedance-2-0-fast-260128", False, "480p"): 0.0056,
    ("dreamina-seedance-2-0-fast-260128", True, "480p"): 0.0033,
    ("dreamina-seedance-2-0-fast-260128", False, "720p"): 0.0056,
    ("dreamina-seedance-2-0-fast-260128", True, "720p"): 0.0033,
    ("dreamina-seedance-2-0-mini", False, "480p"): 0.0035,
    ("dreamina-seedance-2-0-mini", True, "480p"): 0.0021,
    ("dreamina-seedance-2-0-mini", False, "720p"): 0.0035,
    ("dreamina-seedance-2-0-mini", True, "720p"): 0.0021,
}


def seedance2_price_per_1k_tokens(model_id: str, has_video_input: bool, resolution: str) -> float | None:
    return SEEDANCE2_PRICE_PER_1K_TOKENS.get((model_id, has_video_input, resolution))


RECOMMENDED_PRESETS = [
    ("1024x1024 (1:1)", 1024, 1024),
    ("864x1152 (3:4)", 864, 1152),
    ("1152x864 (4:3)", 1152, 864),
    ("1280x720 (16:9)", 1280, 720),
    ("720x1280 (9:16)", 720, 1280),
    ("832x1248 (2:3)", 832, 1248),
    ("1248x832 (3:2)", 1248, 832),
    ("1512x648 (21:9)", 1512, 648),
    ("2048x2048 (1:1)", 2048, 2048),
    ("Custom", None, None),
]

RECOMMENDED_PRESETS_SEEDREAM_4 = [
    ("2048x2048 (1:1)", 2048, 2048),
    ("2304x1728 (4:3)", 2304, 1728),
    ("1728x2304 (3:4)", 1728, 2304),
    ("2560x1440 (16:9)", 2560, 1440),
    ("1440x2560 (9:16)", 1440, 2560),
    ("2496x1664 (3:2)", 2496, 1664),
    ("1664x2496 (2:3)", 1664, 2496),
    ("3024x1296 (21:9)", 3024, 1296),
    ("3072x3072 (1:1)", 3072, 3072),
    ("4096x4096 (1:1)", 4096, 4096),
    ("Custom", None, None),
]

_PRESETS_SEEDREAM_1K = [
    ("(1K) 1024x1024 (1:1)", 1024, 1024),
    ("(1K) 864x1152 (3:4)", 864, 1152),
    ("(1K) 1152x864 (4:3)", 1152, 864),
    ("(1K) 1312x736 (16:9)", 1312, 736),
    ("(1K) 736x1312 (9:16)", 736, 1312),
    ("(1K) 832x1248 (2:3)", 832, 1248),
    ("(1K) 1248x832 (3:2)", 1248, 832),
    ("(1K) 1568x672 (21:9)", 1568, 672),
]

_PRESETS_SEEDREAM_2K = [
    ("(2K) 2048x2048 (1:1)", 2048, 2048),
    ("(2K) 1728x2304 (3:4)", 1728, 2304),
    ("(2K) 2304x1728 (4:3)", 2304, 1728),
    ("(2K) 2848x1600 (16:9)", 2848, 1600),
    ("(2K) 1600x2848 (9:16)", 1600, 2848),
    ("(2K) 1664x2496 (2:3)", 1664, 2496),
    ("(2K) 2496x1664 (3:2)", 2496, 1664),
    ("(2K) 3136x1344 (21:9)", 3136, 1344),
]

_PRESETS_SEEDREAM_3K = [
    ("(3K) 3072x3072 (1:1)", 3072, 3072),
    ("(3K) 2592x3456 (3:4)", 2592, 3456),
    ("(3K) 3456x2592 (4:3)", 3456, 2592),
    ("(3K) 4096x2304 (16:9)", 4096, 2304),
    ("(3K) 2304x4096 (9:16)", 2304, 4096),
    ("(3K) 2496x3744 (2:3)", 2496, 3744),
    ("(3K) 3744x2496 (3:2)", 3744, 2496),
    ("(3K) 4704x2016 (21:9)", 4704, 2016),
]

_PRESETS_SEEDREAM_4K = [
    ("(4K) 4096x4096 (1:1)", 4096, 4096),
    ("(4K) 3520x4704 (3:4)", 3520, 4704),
    ("(4K) 4704x3520 (4:3)", 4704, 3520),
    ("(4K) 5504x3040 (16:9)", 5504, 3040),
    ("(4K) 3040x5504 (9:16)", 3040, 5504),
    ("(4K) 3328x4992 (2:3)", 3328, 4992),
    ("(4K) 4992x3328 (3:2)", 4992, 3328),
    ("(4K) 6240x2656 (21:9)", 6240, 2656),
]

_CUSTOM_PRESET = [("Custom", None, None)]

RECOMMENDED_PRESETS_SEEDREAM_5_LITE = (
    _PRESETS_SEEDREAM_2K + _PRESETS_SEEDREAM_3K + _PRESETS_SEEDREAM_4K + _CUSTOM_PRESET
)
RECOMMENDED_PRESETS_SEEDREAM_4_5 = (
    _PRESETS_SEEDREAM_2K + _PRESETS_SEEDREAM_4K + _CUSTOM_PRESET
)
RECOMMENDED_PRESETS_SEEDREAM_4_0 = (
    _PRESETS_SEEDREAM_1K + _PRESETS_SEEDREAM_2K + _PRESETS_SEEDREAM_4K + _CUSTOM_PRESET
)

# Seedance 2.0 reference video pixel count limits per model and output resolution.
SEEDANCE2_REF_VIDEO_PIXEL_LIMITS = {
    "dreamina-seedance-2-0-260128": {
        "480p": {"min": 409_600, "max": 927_408},
        "720p": {"min": 409_600, "max": 927_408},
        "1080p": {"min": 409_600, "max": 2_073_600},
    },
    "dreamina-seedance-2-0-fast-260128": {
        "480p": {"min": 409_600, "max": 927_408},
        "720p": {"min": 409_600, "max": 927_408},
    },
    "dreamina-seedance-2-0-mini": {
        "480p": {"min": 409_600, "max": 927_408},
        "720p": {"min": 409_600, "max": 927_408},
    },
}

# The time in this dictionary are given for 10 seconds duration.
VIDEO_TASKS_EXECUTION_TIME = {
    "seedance-1-0-lite-t2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-lite-i2v-250428": {
        "480p": 40,
        "720p": 60,
        "1080p": 90,
    },
    "seedance-1-0-pro-250528": {
        "480p": 70,
        "720p": 85,
        "1080p": 115,
    },
    "seedance-1-0-pro-fast-251015": {
        "480p": 50,
        "720p": 65,
        "1080p": 100,
    },
    "seedance-1-5-pro-251215": {
        "480p": 80,
        "720p": 100,
        "1080p": 150,
    },
}
