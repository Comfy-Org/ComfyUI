from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class GeminiSafetyCategory(str, Enum):
    HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"


class GeminiSafetyThreshold(str, Enum):
    OFF = "OFF"
    BLOCK_NONE = "BLOCK_NONE"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"


class GeminiSafetySetting(BaseModel):
    category: GeminiSafetyCategory
    threshold: GeminiSafetyThreshold


class GeminiRole(str, Enum):
    user = "user"
    model = "model"


class GeminiMimeType(str, Enum):
    application_pdf = "application/pdf"
    audio_mpeg = "audio/mpeg"
    audio_mp3 = "audio/mp3"
    audio_wav = "audio/wav"
    image_png = "image/png"
    image_jpeg = "image/jpeg"
    image_webp = "image/webp"
    text_plain = "text/plain"
    video_mov = "video/mov"
    video_mpeg = "video/mpeg"
    video_mp4 = "video/mp4"
    video_mpg = "video/mpg"
    video_avi = "video/avi"
    video_wmv = "video/wmv"
    video_mpegps = "video/mpegps"
    video_flv = "video/flv"


class GeminiInlineData(BaseModel):
    data: str | None = Field(
        None,
        description="The base64 encoding of the image, PDF, or video to include inline in the prompt. "
        "When including media inline, you must also specify the media type (mimeType) of the data. Size limit: 20MB",
    )
    mimeType: GeminiMimeType | None = Field(None)


class GeminiFileData(BaseModel):
    fileUri: str | None = Field(None)
    mimeType: GeminiMimeType | None = Field(None)


class GeminiPart(BaseModel):
    inlineData: GeminiInlineData | None = Field(None)
    fileData: GeminiFileData | None = Field(None)
    text: str | None = Field(None)
    thought: bool | None = Field(None)


class GeminiTextPart(BaseModel):
    text: str | None = Field(None)


class GeminiContent(BaseModel):
    parts: list[GeminiPart] = Field([])
    role: GeminiRole = Field(..., examples=["user"])


class GeminiSystemInstructionContent(BaseModel):
    parts: list[GeminiTextPart] = Field(
        ...,
        description="A list of ordered parts that make up a single message. "
        "Different parts may have different IANA MIME types.",
    )
    role: GeminiRole | None = Field(..., description="The role field of systemInstruction may be ignored.")


class GeminiFunctionDeclaration(BaseModel):
    description: str | None = Field(None)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(..., description="JSON schema for the function parameters")


class GeminiTool(BaseModel):
    functionDeclarations: list[GeminiFunctionDeclaration] | None = Field(None)


class GeminiOffset(BaseModel):
    nanos: int | None = Field(None, ge=0, le=999999999)
    seconds: int | None = Field(None, ge=-315576000000, le=315576000000)


class GeminiVideoMetadata(BaseModel):
    endOffset: GeminiOffset | None = Field(None)
    startOffset: GeminiOffset | None = Field(None)


class GeminiThinkingConfig(BaseModel):
    includeThoughts: bool | None = Field(None)
    thinkingLevel: str = Field(...)


class GeminiGenerationConfig(BaseModel):
    maxOutputTokens: int | None = Field(None, ge=16, le=65536)
    seed: int | None = Field(None)
    stopSequences: list[str] | None = Field(None)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    topK: int | None = Field(None, ge=1)
    topP: float | None = Field(None, ge=0.0, le=1.0)
    thinkingConfig: GeminiThinkingConfig | None = Field(None)
    responseModalities: list[str] | None = Field(None)


class GeminiImageOutputOptions(BaseModel):
    mimeType: str = Field("image/png")
    compressionQuality: int | None = Field(None)


class GeminiImageConfig(BaseModel):
    aspectRatio: str | None = Field(None)
    imageSize: str | None = Field(None)
    imageOutputOptions: GeminiImageOutputOptions = Field(default_factory=GeminiImageOutputOptions)


class GeminiImageGenerationConfig(GeminiGenerationConfig):
    responseModalities: list[str] | None = Field(None)
    imageConfig: GeminiImageConfig | None = Field(None)
    thinkingConfig: GeminiThinkingConfig | None = Field(None)


class GeminiImageGenerateContentRequest(BaseModel):
    contents: list[GeminiContent] = Field(...)
    generationConfig: GeminiImageGenerationConfig | None = Field(None)
    safetySettings: list[GeminiSafetySetting] | None = Field(None)
    systemInstruction: GeminiSystemInstructionContent | None = Field(None)
    tools: list[GeminiTool] | None = Field(None)
    videoMetadata: GeminiVideoMetadata | None = Field(None)
    uploadImagesToStorage: bool = Field(True)


class GeminiGenerateContentRequest(BaseModel):
    contents: list[GeminiContent] = Field(...)
    generationConfig: GeminiGenerationConfig | None = Field(None)
    safetySettings: list[GeminiSafetySetting] | None = Field(None)
    systemInstruction: GeminiSystemInstructionContent | None = Field(None)
    tools: list[GeminiTool] | None = Field(None)
    videoMetadata: GeminiVideoMetadata | None = Field(None)


class Modality(str, Enum):
    MODALITY_UNSPECIFIED = "MODALITY_UNSPECIFIED"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    DOCUMENT = "DOCUMENT"


class ModalityTokenCount(BaseModel):
    modality: Modality | None = None
    tokenCount: int | None = Field(None, description="Number of tokens for the given modality.")


class Probability(str, Enum):
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class GeminiSafetyRating(BaseModel):
    category: GeminiSafetyCategory | None = None
    probability: Probability | None = Field(
        None,
        description="The probability that the content violates the specified safety category",
    )


class GeminiCitation(BaseModel):
    authors: list[str] | None = None
    endIndex: int | None = None
    license: str | None = None
    publicationDate: date | None = None
    startIndex: int | None = None
    title: str | None = None
    uri: str | None = None


class GeminiCitationMetadata(BaseModel):
    citations: list[GeminiCitation] | None = None


class GeminiCandidate(BaseModel):
    citationMetadata: GeminiCitationMetadata | None = None
    content: GeminiContent | None = None
    finishReason: str | None = None
    safetyRatings: list[GeminiSafetyRating] | None = None


class GeminiPromptFeedback(BaseModel):
    blockReason: str | None = None
    blockReasonMessage: str | None = None
    safetyRatings: list[GeminiSafetyRating] | None = None


class GeminiUsageMetadata(BaseModel):
    cachedContentTokenCount: int | None = Field(
        None,
        description="Output only. Number of tokens in the cached part in the input (the cached content).",
    )
    candidatesTokenCount: int | None = Field(None, description="Number of tokens in the response(s).")
    candidatesTokensDetails: list[ModalityTokenCount] | None = Field(
        None, description="Breakdown of candidate tokens by modality."
    )
    promptTokenCount: int | None = Field(
        None,
        description="Number of tokens in the request. When cachedContent is set, this is still the total effective prompt size meaning this includes the number of tokens in the cached content.",
    )
    promptTokensDetails: list[ModalityTokenCount] | None = Field(
        None, description="Breakdown of prompt tokens by modality."
    )
    thoughtsTokenCount: int | None = Field(None, description="Number of tokens present in thoughts output.")
    toolUsePromptTokenCount: int | None = Field(None, description="Number of tokens present in tool-use prompt(s).")


class GeminiGenerateContentResponse(BaseModel):
    candidates: list[GeminiCandidate] | None = Field(None)
    promptFeedback: GeminiPromptFeedback | None = Field(None)
    usageMetadata: GeminiUsageMetadata | None = Field(None)
    modelVersion: str | None = Field(None)


class GeminiInteractionTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(...)


class GeminiInteractionMediaPart(BaseModel):
    type: str = Field(..., description="One of: image, video, audio, document.")
    data: str | None = Field(None, description="Base64-encoded media bytes.")
    uri: str | None = Field(None, description="URI of the media, as an alternative to inline data.")
    mime_type: str | None = Field(None)


class GeminiInteractionVideoConfig(BaseModel):
    task: str | None = Field(
        None, description="One of: text_to_video, image_to_video, reference_to_video, edit, extend."
    )


class GeminiInteractionGenerationConfig(BaseModel):
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    video_config: GeminiInteractionVideoConfig | None = Field(None)


class GeminiInteractionResponseFormat(BaseModel):
    type: Literal["video"] = "video"
    resolution: str | None = Field(None, description="One of: 360p, 720p, 1080p, 4k.")
    aspect_ratio: str | None = Field(None, description="One of: 16:9, 9:16.")
    delivery: str | None = Field(
        None, description="Set to 'uri' to receive a Files API URI instead of inline base64 data."
    )


class GeminiInteractionRequest(BaseModel):
    model: str = Field(...)
    input: list[GeminiInteractionTextPart | GeminiInteractionMediaPart] = Field(...)
    generation_config: GeminiInteractionGenerationConfig | None = Field(None)
    response_format: GeminiInteractionResponseFormat | None = Field(None)


class GeminiInteractionModalityTokens(BaseModel):
    modality: str | None = Field(None, description="One of: text, image, audio, video, document.")
    tokens: int | None = Field(None)


class GeminiInteractionUsage(BaseModel):
    input_tokens_by_modality: list[GeminiInteractionModalityTokens] | None = Field(None)
    output_tokens_by_modality: list[GeminiInteractionModalityTokens] | None = Field(None)
    total_thought_tokens: int | None = Field(None)


class GeminiInteractionContent(BaseModel):
    type: str | None = Field(None)
    text: str | None = Field(None)
    data: str | None = Field(None)
    uri: str | None = Field(None)
    mime_type: str | None = Field(None)


class GeminiInteractionStep(BaseModel):
    type: str | None = Field(None)
    content: list[GeminiInteractionContent] | None = Field(None)


class GeminiInteraction(BaseModel):
    id: str | None = Field(None)
    status: str | None = Field(
        None,
        description="One of: in_progress, requires_action, completed, failed, cancelled, incomplete.",
    )
    steps: list[GeminiInteractionStep] | None = Field(None)
    usage: GeminiInteractionUsage | None = Field(None)


class GeminiFile(BaseModel):
    name: str | None = Field(None, description="Resource name of the file, in the form 'files/<id>'.")
    uri: str | None = Field(None)
    state: str | None = Field(None, description="One of: PROCESSING, ACTIVE, FAILED.")
