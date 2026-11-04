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


class PikaMediaFile(BaseModel):
    url: str = Field(...)


class PikaJobOutput(BaseModel):
    video: PikaMediaFile | None = Field(None)


class PikaJob(BaseModel):
    id: str = Field(...)
    status: str = Field(...)
    output: PikaJobOutput | None = Field(None)
