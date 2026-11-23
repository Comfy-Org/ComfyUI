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


class PrunaReferenceVideoInput(BaseModel):
    prompt: str = Field(...)
    image: list[str] | None = Field(None)
    video: list[str] | None = Field(None)
    audio: list[str] | None = Field(None)
    keep_audio: bool = Field(...)
    first_frame_image: str | None = Field(None)
    last_frame_image: str | None = Field(None)
    duration: int = Field(...)
    resolution: str = Field(...)
    mode: str = Field(...)
    aspect_ratio: str = Field(...)
    prompt_upsampler: str = Field(...)
    seed: int = Field(...)


class PrunaAvatarInput(BaseModel):
    image: str = Field(...)
    audio: str | None = Field(None)
    voice_script: str | None = Field(None)
    voice: str | None = Field(None)
    voice_language: str | None = Field(None)
    voice_prompt: str | None = Field(None)
    prompt: str | None = Field(None)
    prompt_upsampling: bool = Field(...)
    resolution: str = Field(...)
    seed: int = Field(...)


class PrunaPredictionRequest(BaseModel):
    input: PrunaVideoInput | PrunaReferenceVideoInput | PrunaAvatarInput = Field(...)


class PrunaUpscaleInput(BaseModel):
    video: str = Field(...)
    target: str = Field(...)
    fps: int | None = Field(None)
    turbo: bool = Field(...)


class PrunaUpscaleRequest(BaseModel):
    input: PrunaUpscaleInput = Field(...)


class PrunaPredictionResponse(BaseModel):
    id: str = Field(...)
    model: str | None = Field(None)
    get_url: str | None = Field(None)


class PrunaPredictionStatusResponse(BaseModel):
    status: str | None = Field(None)
    message: str | None = Field(None)
    generation_url: str | None = Field(None)
    error: str | None = Field(None)
