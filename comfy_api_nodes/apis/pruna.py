from pydantic import BaseModel, Field


class PrunaVideoInput(BaseModel):
    prompt: str = Field(...)
    duration: int | None = Field(None)
    resolution: str = Field(...)
    aspect_ratio: str | None = Field(None)
    fps: int = Field(...)
    draft: bool = Field(...)
    save_audio: bool = Field(...)
    prompt_upsampling: bool = Field(...)
    seed: int = Field(...)
    image: str | None = Field(None)
    last_frame_image: str | None = Field(None)
    audio: str | None = Field(None)


class PrunaPredictionRequest(BaseModel):
    input: PrunaVideoInput = Field(...)


class PrunaPredictionResponse(BaseModel):
    id: str = Field(...)
    model: str | None = Field(None)
    get_url: str | None = Field(None)


class PrunaPredictionStatusResponse(BaseModel):
    status: str | None = Field(None)
    message: str | None = Field(None)
    generation_url: str | None = Field(None)
    error: str | None = Field(None)
