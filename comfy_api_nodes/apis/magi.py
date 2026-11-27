from pydantic import BaseModel, Field


class Magi2TextToVideoRequest(BaseModel):
    prompt: str = Field(...)
    duration: int = Field(...)
    aspect_ratio: str = Field(...)
    generate_audio: bool = Field(...)
    seed: int = Field(...)


class Magi2ImageToVideoRequest(BaseModel):
    image: str = Field(...)
    end_image: str | None = Field(None)
    prompt: str = Field(...)
    duration: int = Field(...)
    aspect_ratio: str = Field(...)
    generate_audio: bool = Field(...)
    seed: int = Field(...)


class Magi2ReferenceToVideoRequest(BaseModel):
    prompt: str = Field(...)
    images: list[str] | None = Field(None)
    videos: list[str] | None = Field(None)
    audios: list[str] | None = Field(None)
    duration: int = Field(...)
    aspect_ratio: str = Field(...)
    generate_audio: bool = Field(...)
    seed: int = Field(...)


class SandbaseRunOutput(BaseModel):
    url: str = Field(...)
    content_type: str | None = Field(None)


class SandbaseRunResponse(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    outputs: list[SandbaseRunOutput] = Field(...)
