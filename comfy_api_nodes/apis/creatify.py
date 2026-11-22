from pydantic import BaseModel, Field


class CreatifyBorealRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    image_url: str | None = Field(None)
    end_image_url: str | None = Field(None)
    reference_image_urls: list[str] | None = Field(None)
    reference_video_urls: list[str] | None = Field(None)
    reference_audio_urls: list[str] | None = Field(None)
    resolution: str = Field(...)
    aspect_ratio: str = Field(...)
    duration: int = Field(...)
    prompt_enhancement: str = Field(...)


class CreatifyAuroraRequest(BaseModel):
    image: str = Field(...)
    audio: str = Field(...)
    model_version: str = Field(...)
    resolution: str = Field(...)
    text_prompt: str | None = Field(None)


class CreatifyTask(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    failed_reason: str | None = Field(None)
    video_output: str | None = Field(None)
