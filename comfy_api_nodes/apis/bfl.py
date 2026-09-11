from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BFLFluxExpandImageRequest(BaseModel):
    prompt: str = Field(...)
    prompt_upsampling: bool | None = Field(None)
    seed: int | None = Field(None)
    top: int = Field(...)
    bottom: int = Field(...)
    left: int = Field(...)
    right: int = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)
    safety_tolerance: int = Field(6)
    output_format: str = Field("png")
    image: str = Field(None, description="A Base64-encoded string representing the image you wish to expand")


class BFLFluxFillImageRequest(BaseModel):
    prompt: str = Field(...)
    prompt_upsampling: bool | None = Field(None)
    seed: int | None = Field(None)
    steps: int = Field(...)
    guidance: float = Field(...)
    safety_tolerance: int = Field(6)
    output_format: str = Field("png")
    image: str = Field(
        None, description="Base64-encoded string representing the image to modify. Can contain alpha mask if desired.",
    )
    mask: str = Field(
        None, description="Base64-encoded string representing the mask of the areas you wish to modify."
    )


class BFLFluxEraseRequest(BaseModel):
    image: str = Field(..., description="A Base64-encoded string representing the image to erase from.")
    mask: str = Field(
        ...,
        description="A Base64-encoded black/white mask matching the input dimensions; "
        "white (255) marks areas to remove, black (0) marks areas to preserve.",
    )
    dilate_pixels: int = Field(10)
    seed: int | None = Field(None)
    output_format: str = Field("png")


class BFLFluxVTORequest(BaseModel):
    prompt: str = Field(
        ..., description="Natural-language styling instruction. Required field, but may be an empty string."
    )
    person: str = Field(..., description="A Base64-encoded string representing the person image.")
    garment: str = Field(..., description="A Base64-encoded string representing the garment reference image.")
    seed: int | None = Field(None)
    safety_tolerance: int = Field(5)
    output_format: str = Field("png")


class BFLFluxProGenerateRequest(BaseModel):
    prompt: str = Field(...)
    prompt_upsampling: bool | None = Field(None)
    seed: int | None = Field(None)
    width: int = Field(1024, description="Must be a multiple of 32.")
    height: int = Field(768, description="Must be a multiple of 32.")
    safety_tolerance: int = Field(6)
    output_format: str = Field("png")
    image_prompt: str | None = Field(None, description="Optional image to remix in base64 format")


class Flux2ProGenerateRequest(BaseModel):
    prompt: str = Field(...)
    width: int = Field(1024, description="Must be a multiple of 32.")
    height: int = Field(768, description="Must be a multiple of 32.")
    seed: int | None = Field(None)
    prompt_upsampling: bool | None = Field(None)
    input_image: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_2: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_3: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_4: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_5: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_6: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_7: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_8: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    input_image_9: str | None = Field(None, description="Base64 encoded image for image-to-image generation")
    safety_tolerance: int = Field(5)
    output_format: str = Field("png")


class BFLFluxKontextProGenerateRequest(BaseModel):
    prompt: str = Field(...)
    input_image: str | None = Field(None, description="Image to edit in base64 format")
    seed: int | None = Field(None)
    guidance: float = Field(...)
    steps: int = Field(...)
    safety_tolerance: int = Field(2)
    output_format: str = Field("png")
    aspect_ratio: str | None = Field(None)
    prompt_upsampling: bool | None = Field(None)


class BFLFluxProUltraGenerateRequest(BaseModel):
    prompt: str = Field(...)
    prompt_upsampling: bool | None = Field(None)
    seed: int | None = Field(None)
    aspect_ratio: str | None = Field(None)
    safety_tolerance: int = Field(6)
    output_format: str = Field("png")
    raw: bool | None = Field(None)
    image_prompt: str | None = Field(None, description="Optional image to remix in base64 format")
    image_prompt_strength: float | None = Field(None)


class BFLFluxProGenerateResponse(BaseModel):
    id: str = Field(...)
    polling_url: str = Field(...)
    cost: float | None = Field(None, description="Price in cents")


class BFLStatus(str, Enum):
    task_not_found = "Task not found"
    pending = "Pending"
    reasoning = "Reasoning"
    generating = "Generating"
    request_moderated = "Request Moderated"
    content_moderated = "Content Moderated"
    ready = "Ready"
    error = "Error"


class BFLFluxStatusResponse(BaseModel):
    id: str = Field(...)
    status: BFLStatus = Field(...)
    result: dict[str, Any] | None = Field(None)
    progress: float | None = Field(None, ge=0.0, le=1.0)


class Flux3VideoRequest(BaseModel):
    """Fields shared by every generation mode of /v1/flux-3-video."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(...)
    aspect_ratio: str = Field("auto")
    duration: int | str = Field("auto", description="Whole seconds, or 'auto'.")
    resolution: str = Field("hd", description="'hd' is the 720p class, 'fhd' the 1080p class.")
    generate_audio: bool = Field(True)
    safety_tolerance: int = Field(2, description="0 is the strictest; conditioned modes cap at 2.")


class Flux3TextToVideoRequest(Flux3VideoRequest):
    mode: str = Field("t2v")


class Flux3ImageToVideoRequest(Flux3VideoRequest):
    mode: str = Field("i2v")
    keyframes: list[str] | list[tuple[float, str]] = Field(
        ...,
        description="Images (URL or base64), or [seconds, image] pairs pinning each to a time.",
    )


class Flux3VideoContinuationRequest(Flux3VideoRequest):
    mode: str = Field("v2v")
    start_video: str = Field(
        ..., description="MP4 (URL or base64); the new clip carries on from its final frames."
    )


class BFLFluxVideoUpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_video: str = Field(..., description="MP4 (URL or base64), 1 to 20 seconds.")
    upscale_factor: float = Field(2.0, ge=1.5, le=3.0)
    creativity: int = Field(1, description="0 preserves the source precisely, 1 enhances detail.")
    prompt: str | None = Field(None)
    safety_tolerance: int = Field(2, ge=0, le=4)
