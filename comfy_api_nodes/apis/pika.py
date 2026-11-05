from pydantic import BaseModel, Field


class PikaH3FlexRequest(BaseModel):
    prompt: str = Field(...)
    duration: int = Field(...)
    aspect_ratio: str = Field(...)
    resolution: str = Field(...)
    seed: int = Field(...)
    rewrite: bool = Field(...)


class PikaH3FlexImageRequest(PikaH3FlexRequest):
    image: str | None = Field(None)
    last_frame_image: str | None = Field(None)


class PikaH3FlexReference(BaseModel):
    type: str = Field(...)
    url: str = Field(...)


class PikaH3FlexReferenceRequest(PikaH3FlexRequest):
    references: list[PikaH3FlexReference] = Field(...)


class PikaSoundtrackRequest(BaseModel):
    video: str = Field(...)
    instruction: str | None = Field(None)
    seed: int = Field(...)


class PikaSFXRequest(BaseModel):
    prompt: str = Field(...)
    duration: float = Field(...)
    seed: int = Field(...)


class PikaMusicRequest(BaseModel):
    mode: str = Field(...)
    prompt: str = Field(...)
    lyrics: str | None = Field(None)
    source_audio: str | None = Field(None)
    rewrite_prompt: bool = Field(...)
    duration: float | None = Field(None)
    seed: int = Field(...)


class PikaMediaFile(BaseModel):
    url: str = Field(...)


class PikaJobOutput(BaseModel):
    video: PikaMediaFile | None = Field(None)
    audio: PikaMediaFile | None = Field(None)


class PikaJob(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    output: PikaJobOutput | None = Field(None)
