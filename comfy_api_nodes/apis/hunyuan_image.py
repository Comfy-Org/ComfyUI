from pydantic import BaseModel, Field


class HunyuanImageUrl(BaseModel):
    url: str = Field(...)


class HunyuanImageContentItem(BaseModel):
    type: str = Field(...)
    text: str | None = Field(None)
    image_url: HunyuanImageUrl | None = Field(None)


class HunyuanImageMessage(BaseModel):
    role: str = Field("user")
    content: list[HunyuanImageContentItem] = Field(...)


class HunyuanImageRequest(BaseModel):
    model: str = Field(...)
    messages: list[HunyuanImageMessage] = Field(...)
    size: str | None = Field(None)
    generate_max_pixels: int | None = Field(None)
    resize_max_pixels: int | None = Field(None)
    seed: int = Field(...)
    logo_add: int = Field(0)


class HunyuanImageResult(BaseModel):
    url: str | None = Field(None)
    width: int | None = Field(None)
    height: int | None = Field(None)


class HunyuanImageDelta(BaseModel):
    image: HunyuanImageResult | None = Field(None)


class HunyuanImageChoice(BaseModel):
    delta: HunyuanImageDelta | None = Field(None)
    finish_reason: str | None = Field(None)


class HunyuanImageError(BaseModel):
    message: str | None = Field(None)
    code: str | int | None = Field(None)
    type: str | None = Field(None)


class HunyuanImageUsage(BaseModel):
    total_tokens: int | None = Field(None)


class HunyuanImageResponse(BaseModel):
    choices: list[HunyuanImageChoice] = Field(default_factory=list)
    error: HunyuanImageError | None = Field(None)
    tokenhub_usage: HunyuanImageUsage | None = Field(None)
    request_id: str | None = Field(None)
