from pydantic import BaseModel, Field


class Datum2(BaseModel):
    b64_json: str | None = Field(None, description="Base64 encoded image data")
    revised_prompt: str | None = Field(None, description="Revised prompt")
    url: str | None = Field(None, description="URL of the image")


class InputTokensDetails(BaseModel):
    image_tokens: int | None = Field(None)
    text_tokens: int | None = Field(None)


class Usage(BaseModel):
    input_tokens: int | None = Field(None)
    input_tokens_details: InputTokensDetails | None = Field(None)
    output_tokens: int | None = Field(None)
    total_tokens: int | None = Field(None)


class OpenAIImageGenerationResponse(BaseModel):
    data: list[Datum2] | None = Field(None)
    usage: Usage | None = Field(None)


class OpenAIImageEditRequest(BaseModel):
    background: str | None = Field(None, description="Background transparency")
    model: str = Field(...)
    moderation: str | None = Field(None)
    n: int | None = Field(None, description="The number of images to generate")
    output_compression: int | None = Field(None, description="Compression level for JPEG or WebP (0-100)")
    output_format: str | None = Field(None)
    prompt: str = Field(...)
    quality: str | None = Field(None, description="Size of the image (e.g., 1024x1024, 1536x1024, auto)")
    size: str | None = Field(None, description="Size of the output image")


class OpenAIImageGenerationRequest(BaseModel):
    background: str | None = Field(None, description="Background transparency")
    model: str | None = Field(None)
    moderation: str | None = Field(None)
    n: int | None = Field(
        None,
        description="The number of images to generate.",
    )
    output_compression: int | None = Field(None, description="Compression level for JPEG or WebP (0-100)")
    output_format: str | None = Field(None)
    prompt: str = Field(...)
    quality: str | None = Field(None, description="The quality of the generated image")
    size: str | None = Field(None, description="Size of the image (e.g., 1024x1024, 1536x1024, auto)")
    style: str | None = Field(None, description="Style of the image (only for dall-e-3)")


class ModelResponseProperties(BaseModel):
    instructions: str | None = Field(None)
    max_output_tokens: int | None = Field(None)
    model: str | None = Field(None)
    temperature: float | None = Field(None, description="Controls randomness in the response", ge=0.0, le=2.0)
    top_p: float | None = Field(
        None,
        description="Controls diversity of the response via nucleus sampling",
        ge=0.0,
        le=1.0,
    )
    truncation: str | None = Field(None, description="Allowed values: 'auto' or 'disabled'")


class ResponseProperties(BaseModel):
    instructions: str | None = Field(None)
    max_output_tokens: int | None = Field(None)
    model: str | None = Field(None)
    previous_response_id: str | None = Field(None)
    truncation: str | None = Field("disabled", description="Allowed values: 'auto' or 'disabled'")


class ResponseError(BaseModel):
    code: str = Field(...)
    message: str = Field(...)


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = Field(..., description="The number of reasoning tokens.")


class CachedTokensDetails(BaseModel):
    cached_tokens: int = Field(
        ...,
        description="The number of tokens that were retrieved from the cache.",
    )


class ResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens.")
    input_tokens_details: CachedTokensDetails = Field(...)
    output_tokens: int = Field(..., description="The number of output tokens.")
    output_tokens_details: OutputTokensDetails = Field(...)
    total_tokens: int = Field(..., description="The total number of tokens used.")


class InputTextContent(BaseModel):
    text: str = Field(..., description="The text input to the model.")
    type: str = Field("input_text")


class OutputContent(BaseModel):
    type: str = Field(..., description="The type of output content")
    text: str | None = Field(None, description="The text content")
    data: str | None = Field(None, description="Base64-encoded audio data")
    transcript: str | None = Field(None, description="Transcript of the audio")


class OutputMessage(BaseModel):
    type: str = Field(..., description="The type of output item")
    content: list[OutputContent] | None = Field(None, description="The content of the message")
    role: str | None = Field(None, description="The role of the message")


class OpenAIResponse(ModelResponseProperties, ResponseProperties):
    created_at: float | None = Field(
        None,
        description="Unix timestamp (in seconds) of when this Response was created.",
    )
    error: ResponseError | None = Field(None)
    id: str | None = Field(None, description="Unique identifier for this Response.")
    object: str | None = Field(None, description="The object type of this resource - always set to `response`.")
    output: list[OutputMessage] | None = Field(None)
    parallel_tool_calls: bool | None = Field(True)
    status: str | None = Field(
        None,
        description="One of `completed`, `failed`, `in_progress`, or `incomplete`.",
    )
    usage: ResponseUsage | None = Field(None)


class InputImageContent(BaseModel):
    detail: str = Field(..., description="One of `high`, `low`, or `auto`. Defaults to `auto`.")
    file_id: str | None = Field(None)
    image_url: str | None = Field(None)
    type: str = Field(..., description="The type of the input item. Always `input_image`.")


class InputFileContent(BaseModel):
    file_data: str | None = Field(None)
    file_id: str | None = Field(None)
    filename: str | None = Field(None, description="The name of the file to be sent to the model.")
    type: str = Field(..., description="The type of the input item. Always `input_file`.")


class InputMessage(BaseModel):
    content: list[InputTextContent | InputImageContent | InputFileContent] = Field(
        ...,
        description="A list of one or many input items to the model, containing different content types.",
    )
    role: str | None = Field(None)
    type: str | None = Field(None)


class OpenAICreateResponse(ModelResponseProperties, ResponseProperties):
    include: str | None = Field(None)
    input: list[InputMessage] = Field(...)
    parallel_tool_calls: bool | None = Field(
        True, description="Whether to allow the model to run tool calls in parallel."
    )
    store: bool | None = Field(
        True,
        description="Whether to store the generated model response for later retrieval via API.",
    )
    stream: bool | None = Field(False)
    usage: ResponseUsage | None = Field(None)
