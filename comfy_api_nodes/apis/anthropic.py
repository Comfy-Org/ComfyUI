from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AnthropicRole(str, Enum):
    user = "user"
    assistant = "assistant"


class AnthropicTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(...)


class AnthropicImageSourceBase64(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str = Field(..., description="MIME type of the image, e.g. image/png, image/jpeg")
    data: str = Field(..., description="Base64-encoded image data")


class AnthropicImageSourceUrl(BaseModel):
    type: Literal["url"] = "url"
    url: str = Field(...)


class AnthropicImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSourceBase64 | AnthropicImageSourceUrl = Field(...)


class AnthropicMessage(BaseModel):
    role: AnthropicRole = Field(...)
    content: list[AnthropicTextContent | AnthropicImageContent] = Field(...)


class AnthropicThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled", "adaptive"] = Field(...)
    budget_tokens: int | None = Field(
        None, ge=1024,
        description="Reasoning budget in tokens. Used when type is 'enabled'. Must be less than max_tokens.",
    )


class AnthropicOutputConfig(BaseModel):
    """Used with `thinking.type='adaptive'` on models like Opus 4.7."""
    effort: Literal["low", "medium", "high"] | None = Field(None)


class AnthropicMessagesRequest(BaseModel):
    model: str = Field(...)
    messages: list[AnthropicMessage] = Field(...)
    max_tokens: int = Field(..., ge=1)
    system: str | None = Field(None, description="Top-level system prompt")
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0)
    stop_sequences: list[str] | None = Field(None)
    thinking: AnthropicThinkingConfig | None = Field(None)
    output_config: AnthropicOutputConfig | None = Field(None)


class AnthropicResponseTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(...)


class AnthropicResponseThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = Field(...)


AnthropicResponseBlock = AnthropicResponseTextBlock | AnthropicResponseThinkingBlock


class AnthropicCacheCreationUsage(BaseModel):
    ephemeral_5m_input_tokens: int | None = Field(None)
    ephemeral_1h_input_tokens: int | None = Field(None)


class AnthropicMessagesUsage(BaseModel):
    input_tokens: int | None = Field(None)
    output_tokens: int | None = Field(None)
    cache_creation_input_tokens: int | None = Field(None)
    cache_read_input_tokens: int | None = Field(None)
    cache_creation: AnthropicCacheCreationUsage | None = Field(None)


class AnthropicMessagesResponse(BaseModel):
    id: str | None = Field(None)
    type: str | None = Field(None)
    role: str | None = Field(None)
    model: str | None = Field(None)
    content: list[AnthropicResponseBlock] | None = Field(None)
    stop_reason: str | None = Field(None)
    stop_sequence: str | None = Field(None)
    usage: AnthropicMessagesUsage | None = Field(None)
