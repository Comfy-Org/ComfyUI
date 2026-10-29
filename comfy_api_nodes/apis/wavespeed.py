from pydantic import BaseModel, Field


class SeedVR2ImageRequest(BaseModel):
    image: str = Field(...)
    target_resolution: str = Field(...)
    output_format: str = Field("png")
    enable_sync_mode: bool = Field(False)


class FlashVSRRequest(BaseModel):
    target_resolution: str = Field(...)
    video: str = Field(...)
    duration: float = Field(...)


class TaskResultDataResponse(BaseModel):
    status: str = Field(...)
    outputs: list[str] = Field([])


class TaskResultResponse(BaseModel):
    code: int = Field(...)
    message: str = Field(...)
    data: TaskResultDataResponse | None = Field(None)
