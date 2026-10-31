from pydantic import BaseModel, Field


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
