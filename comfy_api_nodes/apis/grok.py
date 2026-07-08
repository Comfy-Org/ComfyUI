from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    aspect_ratio: str = Field(...)
    n: int = Field(...)
    seed: int = Field(...)
    response_format: str = Field("url")
    resolution: str = Field(...)


class InputUrlObject(BaseModel):
    url: str = Field(...)


class ImageEditRequest(BaseModel):
    model: str = Field(...)
    images: list[InputUrlObject] = Field(...)
    prompt: str = Field(...)
    resolution: str = Field(...)
    n: int = Field(...)
    seed: int = Field(...)
    response_format: str = Field("url")
    aspect_ratio: str | None = Field(...)


class VideoGenerationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    image: InputUrlObject | None = Field(None)
    reference_images: list[InputUrlObject] | None = Field(None)
    duration: int = Field(...)
    aspect_ratio: str | None = Field(...)
    resolution: str = Field(...)
    seed: int = Field(...)


class VideoExtensionRequest(BaseModel):
    prompt: str = Field(...)
    video: InputUrlObject = Field(...)
    duration: int = Field(default=6)
    model: str | None = Field(default=None)


class VideoEditRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    video: InputUrlObject = Field(...)
    seed: int = Field(...)


class ImageResponseObject(BaseModel):
    url: str | None = Field(None)
    b64_json: str | None = Field(None)
    revised_prompt: str | None = Field(None)


class UsageObject(BaseModel):
    cost_in_usd_ticks: int | None = Field(None)


class ImageGenerationResponse(BaseModel):
    data: list[ImageResponseObject] = Field(...)
    usage: UsageObject | None = Field(None)


class VideoGenerationResponse(BaseModel):
    request_id: str = Field(...)


class VideoResponseObject(BaseModel):
    url: str = Field(...)
    upsampled_prompt: str | None = Field(None)
    duration: int = Field(...)


class VideoStatusResponse(BaseModel):
    status: str | None = Field(None)
    video: VideoResponseObject | None = Field(None)
    model: str | None = Field(None)
    usage: UsageObject | None = Field(None)
