from pydantic import BaseModel, Field


class QwenImageContentItem(BaseModel):
    image: str | None = Field(None)
    text: str | None = Field(None)


class QwenImageMessage(BaseModel):
    role: str = Field("user")
    content: list[QwenImageContentItem] = Field(...)


class QwenImageInputField(BaseModel):
    messages: list[QwenImageMessage] = Field(...)


class QwenImageParametersField(BaseModel):
    size: str | None = Field(None, description="Output resolution as 'width*height'; omit for the model default.")
    n: int = Field(1, ge=1, le=6)
    seed: int = Field(..., ge=0, le=2147483647)
    prompt_extend: bool = Field(True)
    watermark: bool = Field(False)
    negative_prompt: str | None = Field(None)


class QwenImageGenerationRequest(BaseModel):
    model: str = Field(...)
    input: QwenImageInputField = Field(...)
    parameters: QwenImageParametersField = Field(...)


class QwenImageChoice(BaseModel):
    finish_reason: str | None = Field(None)
    message: QwenImageMessage | None = Field(None)


class QwenImageOutputField(BaseModel):
    choices: list[QwenImageChoice] = Field(default_factory=list)


class QwenImageGenerationResponse(BaseModel):
    output: QwenImageOutputField | None = Field(None)
    request_id: str = Field(...)
    code: str | None = Field(None, description="Error code for the failed request.")
    message: str | None = Field(None, description="Details about the failed request.")
