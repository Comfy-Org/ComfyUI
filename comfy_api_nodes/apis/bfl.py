from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    request_moderated = "Request Moderated"
    content_moderated = "Content Moderated"
    ready = "Ready"
    error = "Error"


class BFLFluxStatusResponse(BaseModel):
    id: str = Field(...)
    status: BFLStatus = Field(...)
    result: dict[str, Any] | None = Field(None)
    progress: float | None = Field(None, ge=0.0, le=1.0)
