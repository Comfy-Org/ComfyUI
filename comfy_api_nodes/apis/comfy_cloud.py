from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Only the workflows this build ships. The backend serves more.
ComfyCloudWorkflow = Literal[
    "z-image-turbo/text-to-image",
    "flux-2/text-to-image",
    "minimax-h3/text-to-video",
    "minimax-h3/first-last-frame-to-video",
    "minimax-h3/reference-to-video",
    "minimax-h3/image-to-video",
    "mage-flow/text-to-image",
    "mage-flow-turbo/text-to-image",
    "minimax-music-3/text-to-audio",
]


# Only the inputs the shipped nodes send. The manifests accept more.
class ComfyCloudWorkflowInputs(BaseModel):
    prompt: str | None = Field(None)
    image_url: str | None = Field(None)
    assets: dict[str, "ComfyCloudAssetInput"] | None = Field(None)
    audio_url: str | None = Field(None)
    first_frame_url: str | None = Field(None)
    last_frame_url: str | None = Field(None)
    instruction: str | None = Field(None)
    prompt_enhance: bool | None = Field(None)
    enhance_prompt: bool | None = Field(None)
    negative_prompt: str | None = Field(None)
    aspect_ratio: str | None = Field(None)
    duration_seconds: float | None = Field(None)
    quality_mode: str | None = Field(None)
    seed: int | None = Field(None, ge=0, le=0xFFFFFFFFFFFFFFFF)
    scale: str | None = Field(None)
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    resolution: str | None = Field(None)
    rendering_speed: str | None = Field(None)
    color_correction: str | None = Field(None)
    turbo: bool | None = Field(None)
    style_lora: bool | None = Field(None)
    model: str | None = Field(None)
    lora: str | None = Field(None)
    steps: int | None = Field(None, ge=1)
    turbo_steps: int | None = Field(None, ge=1)
    fast_steps: int | None = Field(None, ge=1)
    cfg: float | None = Field(None, ge=0)
    guidance: float | None = Field(None, ge=0)
    shift: float | None = Field(None, ge=0)
    turbo_strength: float | None = Field(None, ge=0)
    style_strength: float | None = Field(None, ge=0)
    megapixels: float | None = Field(None, gt=0)
    size_multiple: int | None = Field(None, ge=1)
    sampler: str | None = Field(None)
    scheduler: str | None = Field(None)
    denoise: float | None = Field(None, ge=0, le=1)
    text_encoder: str | None = Field(None)
    ref_image_size: str | None = Field(None)
    lyrics: str | None = Field(None)
    max_duration: float | None = Field(None, gt=0)
    caption_cfg: float | None = Field(None, ge=0)
    top_k: int | None = Field(None, ge=1)
    tiled_decode: bool | None = Field(None)
    tile_size: int | None = Field(None, ge=1)
    tile_overlap: int | None = Field(None, ge=0)
    audio_quality: str | None = Field(None)


class ComfyCloudAssetInput(BaseModel):
    type: Literal["IMAGE", "VIDEO", "AUDIO"] = Field(...)
    url: str = Field(...)


class ComfyCloudGenerateRequest(BaseModel):
    workflow: ComfyCloudWorkflow = Field(...)
    inputs: ComfyCloudWorkflowInputs = Field(...)


class ComfyCloudGenerateResponse(BaseModel):
    task_id: str = Field(..., min_length=1)
    status: str = Field(...)
    polling_url: str | None = Field(None)
    cancel_url: str | None = Field(None)

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be blank")
        return value


class ComfyCloudStatusResponse(BaseModel):
    task_id: str = Field(..., min_length=1)
    status: str = Field(...)
    progress: float | None = Field(None)
    output_url: str | None = Field(None)
    error: str | None = Field(None)

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task_id must not be blank")
        return value
