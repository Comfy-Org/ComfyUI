from typing import Literal

from pydantic import BaseModel, Field


class TencentHyTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(...)


class TencentHyImageUrl(BaseModel):
    url: str = Field(...)


class TencentHyImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: TencentHyImageUrl = Field(...)


class TencentHyMessage(BaseModel):
    role: str = Field(...)
    content: list[TencentHyTextPart | TencentHyImagePart] = Field(...)


class TencentHyThinking(BaseModel):
    type: str = Field(...)


class TencentHyChatRequest(BaseModel):
    model: str = Field(...)
    messages: list[TencentHyMessage] = Field(...)
    thinking: TencentHyThinking = Field(...)
    temperature: float = Field(...)
    max_tokens: int = Field(...)
    seed: int = Field(...)
    stream: bool = Field(...)


class TencentHyResponseMessage(BaseModel):
    content: str | None = Field(None)


class TencentHyChoice(BaseModel):
    message: TencentHyResponseMessage | None = Field(None)
    finish_reason: str | None = Field(None)


class TencentHyChatResponse(BaseModel):
    choices: list[TencentHyChoice] | None = Field(None)
