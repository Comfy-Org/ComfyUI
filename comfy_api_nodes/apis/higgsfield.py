from pydantic import BaseModel, Field


class MotionTransferRequest(BaseModel):
    prompt: str = Field(...)
    video_url: str = Field(...)
    image_urls: list[str] = Field(...)
    resolution: str = Field(...)


class HiggsfieldMedia(BaseModel):
    url: str = Field(...)


class HiggsfieldRequestStatus(BaseModel):
    request_id: str = Field(...)
    status: str = Field(...)
    error: str | None = Field(None)
    video: HiggsfieldMedia | None = Field(None)
