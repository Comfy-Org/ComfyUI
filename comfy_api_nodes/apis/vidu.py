from pydantic import BaseModel, Field


class SubjectReference(BaseModel):
    id: str = Field(...)
    images: list[str] = Field(...)


class FrameSetting(BaseModel):
    prompt: str = Field(...)
    key_image: str = Field(...)
    duration: int = Field(...)


class TaskMultiFrameCreationRequest(BaseModel):
    model: str = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    resolution: str = Field(...)
    start_image: str = Field(...)
    image_settings: list[FrameSetting] = Field(...)


class TaskExtendCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(..., max_length=2000)
    duration: int = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    resolution: str = Field(...)
    images: list[str] | None = Field(None, description="Base64 encoded string or image URL")
    video_url: str = Field(..., description="URL of the video to extend")


class TaskCreationRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(..., max_length=5000)
    duration: int = Field(...)
    seed: int = Field(..., ge=0, le=2147483647)
    aspect_ratio: str | None = Field(None)
    resolution: str | None = Field(None)
    movement_amplitude: str | None = Field(None)
    images: list[str] | None = Field(None, description="Base64 encoded string or image URL")
    subjects: list[SubjectReference] | None = Field(None)
    sounds: list[str] | None = Field(None)
    bgm: bool | None = Field(None)
    audio: bool | None = Field(None)


class TaskCreationResponse(BaseModel):
    task_id: str = Field(...)
    state: str = Field(...)
    created_at: str = Field(...)
    code: int | None = Field(None, description="Error code")


class TaskResult(BaseModel):
    id: str = Field(..., description="Creation id")
    url: str = Field(..., description="The URL of the generated results, valid for one hour")
    cover_url: str = Field(..., description="The cover URL of the generated results, valid for one hour")


class BlockedResource(BaseModel):
    content_type: str | None = Field(None)
    index: int | None = Field(None)


class TaskStatusResponse(BaseModel):
    state: str = Field(...)
    err_code: str | None = Field(None)
    blocked_resources: list[BlockedResource] | None = Field(None)
    progress: float | None = Field(None)
    credits: int | None = Field(None)
    creations: list[TaskResult] = Field(..., description="Generated results")
