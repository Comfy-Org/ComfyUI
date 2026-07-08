from pydantic import BaseModel, Field


class RevePostprocessingOperation(BaseModel):
    process: str = Field(..., description="The postprocessing operation: upscale or remove_background.")
    upscale_factor: int | None = Field(
        None,
        description="Upscale factor (2, 3, or 4). Only used when process is upscale.",
        ge=2,
        le=4,
    )


class ReveImageCreateRequest(BaseModel):
    prompt: str = Field(...)
    aspect_ratio: str | None = Field(...)
    version: str = Field(...)
    test_time_scaling: int = Field(
        ...,
        description="If included, the model will spend more effort making better images. Values between 1 and 15.",
        ge=1,
        le=15,
    )
    postprocessing: list[RevePostprocessingOperation] | None = Field(
        None, description="Optional postprocessing operations to apply after generation."
    )


class ReveImageEditRequest(BaseModel):
    edit_instruction: str = Field(...)
    reference_image: str = Field(..., description="A base64 encoded image to use as reference for the edit.")
    aspect_ratio: str | None = Field(...)
    version: str = Field(...)
    test_time_scaling: int | None = Field(
        ...,
        description="If included, the model will spend more effort making better images. Values between 1 and 15.",
        ge=1,
        le=15,
    )
    postprocessing: list[RevePostprocessingOperation] | None = Field(
        None, description="Optional postprocessing operations to apply after generation."
    )


class ReveImageRemixRequest(BaseModel):
    prompt: str = Field(...)
    reference_images: list[str] = Field(..., description="A list of 1-6 base64 encoded reference images.")
    aspect_ratio: str | None = Field(...)
    version: str = Field(...)
    test_time_scaling: int | None = Field(
        ...,
        description="If included, the model will spend more effort making better images. Values between 1 and 15.",
        ge=1,
        le=15,
    )
    postprocessing: list[RevePostprocessingOperation] | None = Field(
        None, description="Optional postprocessing operations to apply after generation."
    )


class ReveImageResponse(BaseModel):
    image: str | None = Field(None, description="The base64 encoded image data.")
    request_id: str | None = Field(None, description="A unique id for the request.")
    credits_used: float | None = Field(None, description="The number of credits used for this request.")
    version: str | None = Field(None, description="The specific model version used.")
    content_violation: bool | None = Field(
        None, description="Indicates whether the generated image violates the content policy."
    )
