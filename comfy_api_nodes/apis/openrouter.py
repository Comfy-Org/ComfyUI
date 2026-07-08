"""Pydantic models for the OpenRouter chat completions API.

See: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
"""

from typing import Literal

from pydantic import BaseModel, Field


class OpenRouterTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(...)


class OpenRouterImageUrl(BaseModel):
    url: str = Field(...)


class OpenRouterImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenRouterImageUrl = Field(...)


class OpenRouterVideoUrl(BaseModel):
    url: str = Field(...)


class OpenRouterVideoContent(BaseModel):
    type: Literal["video_url"] = "video_url"
    video_url: OpenRouterVideoUrl = Field(...)


OpenRouterContentBlock = OpenRouterTextContent | OpenRouterImageContent | OpenRouterVideoContent


class OpenRouterMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(...)
    content: str | list[OpenRouterContentBlock] = Field(...)


class OpenRouterReasoningConfig(BaseModel):
    effort: str | None = Field(None)
    exclude: bool | None = Field(None, description="If true, model reasons but reasoning is excluded from response.")


class OpenRouterWebSearchOptions(BaseModel):
    search_context_size: str | None = Field(None)


class OpenRouterChatRequest(BaseModel):
    model: str = Field(...)
    messages: list[OpenRouterMessage] = Field(...)
    seed: int | None = Field(None)
    reasoning: OpenRouterReasoningConfig | None = Field(None)
    web_search_options: OpenRouterWebSearchOptions | None = Field(None)
    stream: bool = Field(False)


class OpenRouterUsage(BaseModel):
    prompt_tokens: int | None = Field(None)
    completion_tokens: int | None = Field(None)
    total_tokens: int | None = Field(None)
    cost: float | None = Field(None, description="Server-side authoritative USD cost of the call.")


class OpenRouterResponseMessage(BaseModel):
    role: str | None = Field(None)
    content: str | None = Field(None)
    reasoning: str | None = Field(None)
    refusal: str | None = Field(None)


class OpenRouterChoice(BaseModel):
    index: int | None = Field(None)
    message: OpenRouterResponseMessage | None = Field(None)
    finish_reason: str | None = Field(None)


class OpenRouterError(BaseModel):
    code: int | str | None = Field(None)
    message: str | None = Field(None)
    metadata: dict | None = Field(None)


class OpenRouterChatResponse(BaseModel):
    id: str | None = Field(None)
    model: str | None = Field(None)
    object: str | None = Field(None)
    provider: str | None = Field(None)
    choices: list[OpenRouterChoice] | None = Field(None)
    usage: OpenRouterUsage | None = Field(None)
    error: OpenRouterError | None = Field(None)
