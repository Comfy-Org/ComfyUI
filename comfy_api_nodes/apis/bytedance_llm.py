"""Pydantic models for BytePlus ModelArk Responses API.

See: https://docs.byteplus.com/en/docs/ModelArk/1585128 (request)
     https://docs.byteplus.com/en/docs/ModelArk/1783703 (response)
"""

from typing import Literal

from pydantic import BaseModel, Field


class BytePlusInputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str = Field(...)


class BytePlusInputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str = Field(..., description="Image URL or `data:image/...;base64,...` payload")
    detail: str = Field("auto", description="One of high, low, auto")


class BytePlusInputVideo(BaseModel):
    type: Literal["input_video"] = "input_video"
    video_url: str = Field(..., description="Video URL or `data:video/...;base64,...` payload")
    fps: float | None = Field(None, ge=0.2, le=5.0)


BytePlusMessageContent = BytePlusInputText | BytePlusInputImage | BytePlusInputVideo


class BytePlusInputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: str = Field(..., description="One of user, system, assistant, developer")
    content: list[BytePlusMessageContent] = Field(...)


class BytePlusResponseCreateRequest(BaseModel):
    model: str = Field(...)
    input: list[BytePlusInputMessage] = Field(...)
    instructions: str | None = Field(None)
    max_output_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    store: bool | None = Field(False)
    stream: bool | None = Field(False)


class BytePlusOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str = Field(...)


class BytePlusOutputRefusal(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str = Field(...)


class BytePlusOutputContent(BaseModel):
    type: str = Field(...)
    text: str | None = Field(None)
    refusal: str | None = Field(None)


class BytePlusOutputMessage(BaseModel):
    type: str = Field(...)
    id: str | None = Field(None)
    role: str | None = Field(None)
    status: str | None = Field(None)
    content: list[BytePlusOutputContent] | None = Field(None)


class BytePlusInputTokensDetails(BaseModel):
    cached_tokens: int | None = Field(None)


class BytePlusOutputTokensDetails(BaseModel):
    reasoning_tokens: int | None = Field(None)


class BytePlusResponseUsage(BaseModel):
    input_tokens: int | None = Field(None)
    output_tokens: int | None = Field(None)
    total_tokens: int | None = Field(None)
    input_tokens_details: BytePlusInputTokensDetails | None = Field(None)
    output_tokens_details: BytePlusOutputTokensDetails | None = Field(None)


class BytePlusResponseError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class BytePlusResponseObject(BaseModel):
    id: str | None = Field(None)
    object: str | None = Field(None)
    created_at: int | None = Field(None)
    model: str | None = Field(None)
    status: str | None = Field(None)
    error: BytePlusResponseError | None = Field(None)
    output: list[BytePlusOutputMessage] | None = Field(None)
    usage: BytePlusResponseUsage | None = Field(None)
