from typing import Any, Literal

from pydantic import BaseModel, Field


class Seedream4Options(BaseModel):
    max_images: int = Field(15)


class Seedream5OptimizePromptOptions(BaseModel):
    thinking: Literal["auto", "enabled", "disabled"] | None = Field(None)
    mode: Literal["standard", "fast"] | None = Field(None)


class Seedream4TaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    response_format: str = Field("url")
    image: list[str] | None = Field(None, description="Image URLs")
    size: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    sequential_image_generation: str | None = Field("disabled")
    sequential_image_generation_options: Seedream4Options | None = Field(Seedream4Options(max_images=15))
    watermark: bool = Field(False)
    output_format: str | None = None
    optimize_prompt_options: Seedream5OptimizePromptOptions | None = None


class Seedream5LayerOptimizePromptOptions(BaseModel):
    mode: Literal["standard", "fast"] = Field(...)


class Seedream5LayerSeparationRequest(BaseModel):
    model: str = Field(...)
    prompt: str | None = Field(None)
    image: str = Field(..., description="Single image URL")
    size: str = Field("auto")
    seed: int = Field(..., ge=0, le=2147483647)
    response_format: str = Field("url")
    output_format: str = Field("png")
    layer_decomposition: bool = Field(True)
    watermark: bool = Field(False)
    optimize_prompt_options: Seedream5LayerOptimizePromptOptions | None = Field(None)


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
    duration: int | None = Field(None)
    seed: int | None = Field(None, ge=0, le=2147483647)
    watermark: bool | None = Field(None)
    output_format: str | None = Field(None)
    omni_reference_task_type: str | None = Field(None, description="One of: auto, reference, edit, extend.")


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

_PRESETS_SEEDREAM_2K_PRO = [
    ("(2K) 2048x2048 (1:1)", 2048, 2048),
    ("(2K) 1728x2304 (3:4)", 1728, 2304),
    ("(2K) 2304x1728 (4:3)", 2304, 1728),
    # ("(2K) 2848x1600 (16:9)", 2848, 1600),  # 4,556,800 px - temporarily unavailable
    # ("(2K) 1600x2848 (9:16)", 1600, 2848),  # 4,556,800 px - temporarily unavailable
    ("(2K) 1664x2496 (2:3)", 1664, 2496),
    ("(2K) 2496x1664 (3:2)", 2496, 1664),
    # ("(2K) 3136x1344 (21:9)", 3136, 1344),  # 4,214,784 px - temporarily unavailable
]
RECOMMENDED_PRESETS_SEEDREAM_5_PRO = (
    _PRESETS_SEEDREAM_1K + _PRESETS_SEEDREAM_2K_PRO + _CUSTOM_PRESET
)
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
    "dreamina-seedance-2-5-260628": {
        "480p": {"min": 409_600, "max": 8_295_044},
        "720p": {"min": 409_600, "max": 8_295_044},
        "1080p": {"min": 409_600, "max": 8_295_044},
    },
}

SEEDANCE2_REFERENCE_LIMITS_DEFAULT = {
    "max_images": 9,
    "max_videos": 3,
    "max_audios": 3,
    "max_total_seconds": 15.1,
}
SEEDANCE2_REFERENCE_LIMITS = {
    "dreamina-seedance-2-5-260628": {
        "max_images": 30,
        "max_videos": 10,
        "max_audios": 10,
        "max_total_seconds": 30.1,
    },
}


def seedance2_reference_limits(model_id: str) -> dict:
    return SEEDANCE2_REFERENCE_LIMITS.get(model_id, SEEDANCE2_REFERENCE_LIMITS_DEFAULT)

# The time in this dictionary are given for 10 seconds duration.
VIDEO_TASKS_EXECUTION_TIME = {
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


class SeedAudioConfig(BaseModel):
    format: str = Field(default="mp3")
    sample_rate: int = Field(default=24000)
    speech_rate: int = Field(default=0)
    loudness_rate: int = Field(default=0)
    pitch_rate: int = Field(default=0)


class SeedAudioReference(BaseModel):
    speaker: str | None = Field(default=None)
    audio_data: str | None = Field(default=None)
    audio_url: str | None = Field(default=None)
    image_data: str | None = Field(default=None)
    image_url: str | None = Field(default=None)


class SeedAudioRequest(BaseModel):
    model: str = Field(default="seed-audio-1.0")
    text_prompt: str = Field(...)
    references: list[SeedAudioReference] | None = Field(default=None)
    audio_config: SeedAudioConfig = Field(default_factory=SeedAudioConfig)
    watermark: dict[str, Any] = Field(default_factory=dict)


class SeedAudioResponse(BaseModel):
    audio: str | None = Field(default=None)
    url: str | None = Field(default=None)
    duration: float | None = Field(default=None)
    original_duration: float | None = Field(default=None)
    code: int | None = Field(default=None)
    message: str | None = Field(default=None)


class MediaKitVideoEnhanceRequest(BaseModel):
    video_url: str = Field(...)
    tool_version: str = Field(...)
    scene: str | None = Field(None)
    enhance_style: str | None = Field(None)
    resolution: str | None = Field(None)
    resolution_limit: int | None = Field(None)
    fps: float | None = Field(None)
    bitrate_level: str = Field(...)


class MediaKitError(BaseModel):
    code: str | None = Field(None)
    type: str | None = Field(None)
    message: str | None = Field(None)
    param: str | None = Field(None)


class MediaKitTaskCreateResponse(BaseModel):
    success: bool = Field(...)
    task_id: str | None = Field(None)
    request_id: str | None = Field(None)
    error: MediaKitError | None = Field(None)


class MediaKitTaskResult(BaseModel):
    video_url: str = Field(...)
    duration: float | None = Field(None)
    fps: float | None = Field(None)
    resolution: str | None = Field(None)
    tool_version: str | None = Field(None)


class MediaKitTaskResponse(BaseModel):
    success: bool = Field(...)
    task_id: str | None = Field(None)
    task_type: str | None = Field(None)
    status: str | None = Field(None)
    result: MediaKitTaskResult | None = Field(None)
    error: MediaKitError | None = Field(None)
