from pydantic import BaseModel, Field


class MuseImageToolEnablement(BaseModel):
    enable_image_search: bool = Field(...)
    enable_web_search: bool = Field(...)
    enable_shell: bool = Field(...)


class MuseImageRequest(BaseModel):
    model: str = Field(...)
    prompt: str = Field(...)
    n: int = Field(1, ge=1, le=10)
    size: str | None = Field(None)
    reasoning_strength: str = Field("high")
    output_format: str = Field("png")
    response_format: str = Field("b64_json")
    tool_enablement: MuseImageToolEnablement | None = Field(None)


class MuseImageInput(BaseModel):
    image_url: str = Field(...)


class MuseImageEditRequest(MuseImageRequest):
    images: list[MuseImageInput] = Field(...)


class MuseImageData(BaseModel):
    b64_json: str | None = Field(None)
    url: str | None = Field(None)


class MuseImageUsage(BaseModel):
    input_tokens: int | None = Field(None)
    output_tokens: int | None = Field(None)
    total_tokens: int | None = Field(None)


class MuseImageResponse(BaseModel):
    created: int | None = Field(None)
    data: list[MuseImageData] = Field(default_factory=list)
    output_format: str | None = Field(None)
    background: str | None = Field(None)
    usage: MuseImageUsage | None = Field(None)
