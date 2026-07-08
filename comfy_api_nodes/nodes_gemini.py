"""
API Nodes for Gemini Multimodal LLM Usage via Remote API
See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
"""

import base64
import os
from fnmatch import fnmatch
from io import BytesIO
from typing import Any, Literal

import torch
from typing_extensions import override

import folder_paths
from comfy_api.latest import IO, ComfyExtension, Input, InputImpl, Types
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiFileData,
    GeminiGenerateContentRequest,
    GeminiGenerationConfig,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiInlineData,
    GeminiMimeType,
    GeminiPart,
    GeminiRole,
    GeminiSystemInstructionContent,
    GeminiTextPart,
    GeminiThinkingConfig,
    Modality,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    download_url_to_video_output,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_audio_to_comfyapi,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
    video_to_base64_string,
)

GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"
GEMINI_MAX_INPUT_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
GEMINI_URL_INPUT_BUDGET = 10
GEMINI_MAX_INLINE_BYTES = 18 * 1024 * 1024
GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)

GEMINI_IMAGE_2_PRICE_BADGE = IO.PriceBadge(
    depends_on=IO.PriceBadgeDepends(widgets=["model", "resolution"]),
    expr="""
    (
      $m := widgets.model;
      $r := widgets.resolution;
      $isFlash := $contains($m, "nano banana 2");
      $flashPrices := {"1k": 0.0696, "2k": 0.1014, "4k": 0.154};
      $proPrices := {"1k": 0.134, "2k": 0.134, "4k": 0.24};
      $prices := $isFlash ? $flashPrices : $proPrices;
      {"type":"usd","usd": $lookup($prices, $r), "format":{"suffix":"/Image","approximate":true}}
    )
    """,
)


async def create_image_parts(
    cls: type[IO.ComfyNode],
    images: Input.Image | list[Input.Image],
    image_limit: int = 0,
) -> list[GeminiPart]:
    image_parts: list[GeminiPart] = []
    if image_limit < 0:
        raise ValueError("image_limit must be greater than or equal to 0 when creating Gemini image parts.")

    # Accept either a single (possibly-batched) tensor or a list of them; share URL budget across all.
    images_list: list[Input.Image] = images if isinstance(images, list) else [images]
    total_images = sum(get_number_of_images(img) for img in images_list)
    if total_images <= 0:
        raise ValueError("No images provided to create_image_parts; at least one image is required.")

    # If image_limit == 0 --> use all images; otherwise clamp to image_limit.
    effective_max = total_images if image_limit == 0 else min(total_images, image_limit)

    # Number of images we'll send as URLs (fileData)
    num_url_images = min(effective_max, 10)  # Vertex API max number of image links
    upload_kwargs: dict = {"wait_label": "Uploading reference images"}
    if effective_max > num_url_images:
        # Split path (e.g. 11+ images): suppress per-image counter to avoid a confusing dual-fraction label.
        upload_kwargs = {
            "wait_label": f"Uploading reference images ({num_url_images}+)",
            "show_batch_index": False,
        }
    reference_images_urls = await upload_images_to_comfyapi(
        cls,
        images_list,
        max_images=num_url_images,
        **upload_kwargs,
    )
    for reference_image_url in reference_images_urls:
        image_parts.append(
            GeminiPart(
                fileData=GeminiFileData(
                    mimeType=GeminiMimeType.image_png,
                    fileUri=reference_image_url,
                )
            )
        )
    if effective_max > num_url_images:
        flat: list[torch.Tensor] = []
        for tensor in images_list:
            if len(tensor.shape) == 4:
                flat.extend(tensor[i] for i in range(tensor.shape[0]))
            else:
                flat.append(tensor)
        for idx in range(num_url_images, effective_max):
            image_parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=tensor_to_base64_string(flat[idx]),
                    )
                )
            )
    return image_parts


def _mime_matches(mime: GeminiMimeType | None, pattern: str) -> bool:
    """Check if a MIME type matches a pattern. Supports fnmatch globs (e.g. 'image/*')."""
    if mime is None:
        return False
    return fnmatch(mime.value, pattern)


def get_parts_by_type(response: GeminiGenerateContentResponse, part_type: Literal["text"] | str) -> list[GeminiPart]:
    """
    Filter response parts by their type.

    Args:
        response: The API response from Gemini.
        part_type: Type of parts to extract ("text" or a MIME type).

    Returns:
        List of response parts matching the requested type.
    """
    if not response.candidates:
        if response.promptFeedback and response.promptFeedback.blockReason:
            feedback = response.promptFeedback
            raise ValueError(
                f"Gemini API blocked the request. Reason: {feedback.blockReason} ({feedback.blockReasonMessage})"
            )
        raise ValueError(
            "Gemini API returned no response candidates. If you are using the `IMAGE` modality, "
            "try changing it to `IMAGE+TEXT` to view the model's reasoning and understand why image generation failed."
        )
    parts = []
    blocked_reasons = []
    for candidate in response.candidates:
        if candidate.finishReason and candidate.finishReason.upper() == "IMAGE_PROHIBITED_CONTENT":
            blocked_reasons.append(candidate.finishReason)
            continue
        if candidate.content is None or candidate.content.parts is None:
            continue
        for part in candidate.content.parts:
            if part_type == "text" and part.text:
                parts.append(part)
            elif part.inlineData and _mime_matches(part.inlineData.mimeType, part_type):
                parts.append(part)
            elif part.fileData and _mime_matches(part.fileData.mimeType, part_type):
                parts.append(part)

    if not parts and blocked_reasons:
        raise ValueError(f"Gemini API blocked the request. Reasons: {blocked_reasons}")

    return parts


def get_text_from_response(response: GeminiGenerateContentResponse) -> str:
    """
    Extract and concatenate all text parts from the response.

    Args:
        response: The API response from Gemini.

    Returns:
        Combined text from all text parts in the response.
    """
    parts = get_parts_by_type(response, "text")
    return "\n".join([part.text for part in parts])


async def get_image_from_response(response: GeminiGenerateContentResponse, thought: bool = False) -> Input.Image:
    image_tensors: list[Input.Image] = []
    parts = get_parts_by_type(response, "image/*")
    for part in parts:
        if (part.thought is True) != thought:
            continue
        if part.inlineData:
            image_data = base64.b64decode(part.inlineData.data)
            returned_image = bytesio_to_image_tensor(BytesIO(image_data))
        else:
            returned_image = await download_url_to_image_tensor(part.fileData.fileUri)
        image_tensors.append(returned_image)
    if len(image_tensors) == 0:
        if not thought:
            # No images generated --> extract text response for a meaningful error
            model_message = get_text_from_response(response).strip()
            if model_message:
                raise ValueError(f"Gemini did not generate an image. Model response: {model_message}")
            raise ValueError(
                "Gemini did not generate an image. "
                "Try rephrasing your prompt or changing the response modality to 'IMAGE+TEXT' "
                "to see the model's reasoning."
            )
        return torch.zeros((1, 1024, 1024, 4))
    return torch.cat(image_tensors, dim=0)


async def get_video_from_response(
    response: GeminiGenerateContentResponse, cls: type[IO.ComfyNode] | None = None
) -> InputImpl.VideoFromFile:
    parts = get_parts_by_type(response, "video/*")
    for part in parts:
        if part.inlineData and part.inlineData.data:
            return InputImpl.VideoFromFile(BytesIO(base64.b64decode(part.inlineData.data)))
        if part.fileData and part.fileData.fileUri:
            return await download_url_to_video_output(part.fileData.fileUri, cls=cls)
    model_message = get_text_from_response(response).strip()
    if model_message:
        raise ValueError(f"Gemini did not generate a video. Model response: {model_message}")
    raise ValueError(
        "Gemini did not generate a video. Try rephrasing your prompt, "
        "shortening the requested duration, or reducing the number of input images/videos."
    )


def calculate_tokens_price(response: GeminiGenerateContentResponse) -> float | None:
    if not response.modelVersion:
        return None
    # Define prices (Cost per 1,000,000 tokens), see https://cloud.google.com/vertex-ai/generative-ai/pricing
    output_video_tokens_price = 0.0
    if response.modelVersion == "gemini-2.5-pro":
        input_tokens_price = 1.25
        output_text_tokens_price = 10.0
        output_image_tokens_price = 0.0
    elif response.modelVersion == "gemini-2.5-flash":
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 0.0
    elif response.modelVersion == "gemini-2.5-flash-image":
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 30.0
    elif response.modelVersion in ("gemini-3-pro-preview", "gemini-3.1-pro-preview"):
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 0.0
    elif response.modelVersion in ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite"):
        input_tokens_price = 0.25
        output_text_tokens_price = 1.50
        output_image_tokens_price = 0.0
    elif response.modelVersion in ("gemini-3-pro-image-preview", "gemini-3-pro-image"):
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 120.0
    elif response.modelVersion in ("gemini-3.1-flash-image-preview", "gemini-3.1-flash-image"):
        input_tokens_price = 0.5
        output_text_tokens_price = 3.0
        output_image_tokens_price = 60.0
    elif response.modelVersion == "gemini-3.1-flash-lite-image":
        input_tokens_price = 0.25
        output_text_tokens_price = 1.50
        output_image_tokens_price = 30.0
    elif response.modelVersion == "gemini-omni-flash-preview":
        input_tokens_price = 2.145
        output_text_tokens_price = 12.87
        output_image_tokens_price = 0.0
        output_video_tokens_price = 25.025
    else:
        return None
    final_price = response.usageMetadata.promptTokenCount * input_tokens_price
    if response.usageMetadata.candidatesTokensDetails:
        for i in response.usageMetadata.candidatesTokensDetails:
            if i.modality == Modality.IMAGE:
                final_price += output_image_tokens_price * i.tokenCount  # for Nano Banana models
            elif i.modality == Modality.VIDEO:
                final_price += output_video_tokens_price * i.tokenCount  # for Omni Flash
            else:
                final_price += output_text_tokens_price * i.tokenCount
    if response.usageMetadata.thoughtsTokenCount:
        final_price += output_text_tokens_price * response.usageMetadata.thoughtsTokenCount
    return final_price / 1_000_000.0


def create_video_parts(video_input: Input.Video) -> list[GeminiPart]:
    """Convert a single video input to Gemini API compatible parts (inline MP4/H.264)."""
    base_64_string = video_to_base64_string(
        video_input, container_format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264
    )
    return [
        GeminiPart(
            inlineData=GeminiInlineData(
                mimeType=GeminiMimeType.video_mp4,
                data=base_64_string,
            )
        )
    ]


def create_audio_parts(audio_input: Input.Audio) -> list[GeminiPart]:
    """Convert an audio input to Gemini API compatible parts (one inline MP3 part per batch item)."""
    audio_parts: list[GeminiPart] = []
    for batch_index in range(audio_input["waveform"].shape[0]):
        # Recreate an IO.AUDIO object for the given batch dimension index
        audio_at_index = Input.Audio(
            waveform=audio_input["waveform"][batch_index].unsqueeze(0),
            sample_rate=audio_input["sample_rate"],
        )
        # Convert to MP3 format for compatibility with Gemini API
        audio_bytes = audio_to_base64_string(
            audio_at_index,
            container_format="mp3",
            codec_name="libmp3lame",
        )
        audio_parts.append(
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.audio_mp3,
                    data=audio_bytes,
                )
            )
        )
    return audio_parts


def _flatten_images(images: list[Input.Image]) -> list[torch.Tensor]:
    """Expand any batched image tensors into individual (H, W, C) frames, preserving order."""
    frames: list[torch.Tensor] = []
    for img in images:
        if len(img.shape) == 4:
            frames.extend(img[i] for i in range(img.shape[0]))
        else:
            frames.append(img)
    return frames


def _flatten_audio(audios: list[Input.Audio]) -> list[Input.Audio]:
    """Expand any batched audio inputs into individual single-clip audio inputs, preserving order."""
    clips: list[Input.Audio] = []
    for audio in audios:
        waveform = audio["waveform"]
        for i in range(waveform.shape[0]):
            clips.append(Input.Audio(waveform=waveform[i].unsqueeze(0), sample_rate=audio["sample_rate"]))
    return clips


async def _media_url_part(cls: type[IO.ComfyNode], kind: str, payload: Any) -> GeminiPart:
    """Upload a single media unit to ComfyAPI storage and return a fileData (URL) part."""
    if kind == "image":
        url = await upload_image_to_comfyapi(cls, payload, mime_type="image/png", wait_label="Uploading image")
        return GeminiPart(fileData=GeminiFileData(mimeType=GeminiMimeType.image_png, fileUri=url))
    if kind == "audio":
        url = await upload_audio_to_comfyapi(
            cls, payload, container_format="mp3", codec_name="libmp3lame", mime_type="audio/mp3"
        )
        return GeminiPart(fileData=GeminiFileData(mimeType=GeminiMimeType.audio_mp3, fileUri=url))
    url = await upload_video_to_comfyapi(cls, payload, wait_label="Uploading video")
    return GeminiPart(fileData=GeminiFileData(mimeType=GeminiMimeType.video_mp4, fileUri=url))


def _media_inline_part(kind: str, payload: Any) -> tuple[GeminiPart, int]:
    """Encode a single media unit as an inline base64 part; returns (part, base64_length)."""
    if kind == "image":
        data = tensor_to_base64_string(payload, mime_type="image/webp")
        mime = GeminiMimeType.image_webp
    elif kind == "audio":
        data = audio_to_base64_string(payload, container_format="mp3", codec_name="libmp3lame")
        mime = GeminiMimeType.audio_mp3
    else:
        data = video_to_base64_string(
            payload, container_format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264
        )
        mime = GeminiMimeType.video_mp4
    return GeminiPart(inlineData=GeminiInlineData(mimeType=mime, data=data)), len(data)


async def build_gemini_media_parts(
    cls: type[IO.ComfyNode],
    images: list[Input.Image],
    audios: list[Input.Audio],
    videos: list[Input.Video],
    *,
    url_budget: int = GEMINI_URL_INPUT_BUDGET,
    max_inline_bytes: int = GEMINI_MAX_INLINE_BYTES,
) -> list[GeminiPart]:
    """Build Gemini parts for multimodal inputs (images, audio, video).

    fileData URLs are preferred for every media type: the upload is fetched directly by the
    model, keeping the request body tiny regardless of media size. The URL budget is shared
    across all media and assigned largest-first (video, then audio, then images), so that if it
    is ever exhausted the inline-base64 overflow is limited to the smallest items. Total inline
    payload is capped by `max_inline_bytes`.
    """
    units: list[tuple[str, Any]] = (
        [("video", v) for v in videos]
        + [("audio", a) for a in _flatten_audio(audios)]
        + [("image", f) for f in _flatten_images(images)]
    )

    parts: list[GeminiPart] = []
    url_used = 0
    inline_bytes = 0
    for kind, payload in units:
        if url_used < url_budget:
            parts.append(await _media_url_part(cls, kind, payload))
            url_used += 1
            continue
        part, nbytes = _media_inline_part(kind, payload)
        inline_bytes += nbytes
        if inline_bytes > max_inline_bytes:
            raise ValueError(
                f"Too much media to send inline (over {max_inline_bytes // (1024 * 1024)}MB after the first "
                f"{url_budget} inputs are uploaded as URLs). Reduce the number or size of attached media."
            )
        parts.append(part)
    return parts


class GeminiNode(IO.ComfyNode):
    """
    Node to generate text responses from a Gemini model.

    This node allows users to interact with Google's Gemini AI models, providing
    multimodal inputs (text, images, audio, video, files) to generate coherent
    text responses. The node works with the latest Gemini models, handling the
    API communication and response parsing.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiNode",
            display_name="Google Gemini",
            category="partner/text/Gemini",
            description="Generate text responses with Google's Gemini AI model. "
            "You can provide multiple types of inputs (text, images, audio, video) "
            "as context for generating more relevant and meaningful responses.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text inputs to the model, used to generate a response. "
                    "You can include detailed instructions, questions, or context for the model.",
                ),
                IO.Combo.Input(
                    "model",
                    options=[
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-3-pro-preview",
                        "gemini-3-1-pro",
                        "gemini-3-1-flash-lite",
                    ],
                    default="gemini-3-1-pro",
                    tooltip="The Gemini model to use for generating responses.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model. "
                    "To include multiple images, you can use the Batch Images node.",
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Optional audio to use as context for the model.",
                ),
                IO.Video.Input(
                    "video",
                    optional=True,
                    tooltip="Optional video to use as context for the model.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $m := widgets.model;
                  $contains($m, "gemini-2.5-flash") ? {
                    "type": "list_usd",
                    "usd": [0.0003, 0.0025],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens"}
                  }
                  : $contains($m, "gemini-2.5-pro") ? {
                    "type": "list_usd",
                    "usd": [0.00125, 0.01],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : ($contains($m, "gemini-3-pro-preview") or $contains($m, "gemini-3-1-pro")) ? {
                    "type": "list_usd",
                    "usd": [0.002, 0.012],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gemini-3-1-flash-lite") ? {
                    "type": "list_usd",
                    "usd": [0.00025, 0.0015],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : {"type":"text", "text":"Token-based"}
                )
                """,
            ),
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        images: Input.Image | None = None,
        audio: Input.Audio | None = None,
        video: Input.Video | None = None,
        files: list[GeminiPart] | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        if model == "gemini-3-pro-preview":
            model = "gemini-3.1-pro-preview"  # model "gemini-3-pro-preview" will be soon deprecated by Google
        elif model == "gemini-3-1-pro":
            model = "gemini-3.1-pro-preview"
        elif model == "gemini-3-1-flash-lite":
            model = "gemini-3.1-flash-lite-preview"

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            parts.extend(await create_image_parts(cls, images))
        if audio is not None:
            parts.extend(create_audio_parts(audio))
        if video is not None:
            parts.extend(create_video_parts(video))
        if files is not None:
            parts.extend(files)

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model}", method="POST"),
            data=GeminiGenerateContentRequest(
                contents=[
                    GeminiContent(
                        role=GeminiRole.user,
                        parts=parts,
                    )
                ],
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )

        output_text = get_text_from_response(response)
        return IO.NodeOutput(output_text or "Empty response from Gemini model...")


GEMINI_V2_MODELS: dict[str, str] = {
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite-preview",
}


def _gemini_text_model_inputs(thinking_default: str) -> list[Input]:
    """Per-model inputs revealed by the model DynamicCombo (shared media + sampling controls)."""
    return [
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 17)],
                min=0,
            ),
            tooltip="Optional image(s) to use as context for the model. Up to 16 images.",
        ),
        IO.Autogrow.Input(
            "audio",
            template=IO.Autogrow.TemplateNames(
                IO.Audio.Input("audio"),
                names=["audio_1"],
                min=0,
            ),
            tooltip="Optional audio clip to use as context for the model.",
        ),
        IO.Autogrow.Input(
            "video",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("video"),
                names=["video_1"],
                min=0,
            ),
            tooltip="Optional video clip to use as context for the model.",
        ),
        IO.Custom("GEMINI_INPUT_FILES").Input(
            "files",
            optional=True,
            tooltip="Optional file(s) to use as context for the model. "
            "Accepts inputs from the Gemini Input Files node.",
        ),
        IO.Combo.Input(
            "thinking_level",
            options=["LOW", "HIGH"],
            default=thinking_default,
            tooltip="How hard the model reasons internally before answering. "
            "HIGH improves quality on difficult tasks but costs more (thinking) tokens and is slower.",
        ),
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.01,
            tooltip="Controls randomness. Lower is more focused/deterministic, higher is more creative.",
            advanced=True,
        ),
        IO.Float.Input(
            "top_p",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Nucleus sampling: sample from the smallest token set whose cumulative probability reaches top_p.",
            advanced=True,
        ),
        IO.Int.Input(
            "max_output_tokens",
            default=32768,
            min=16,
            max=65536,
            tooltip="Maximum tokens to generate, including the model's internal thinking. "
            "With thinking_level HIGH, a low value can leave no room for the answer; raise this if "
            "responses come back empty or truncated. The model stops early when finished, so a higher "
            "cap costs nothing extra for short replies.",
            advanced=True,
        ),
    ]


class GeminiNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiNodeV2",
            display_name="Google Gemini",
            category="partner/text/Gemini",
            essentials_category="Text Generation",
            description="Generate text responses with Google's Gemini models. Provide a text prompt and, "
            "optionally, one or more images, audio clips, videos, or files as multimodal context.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text input to the model. Include detailed instructions, questions, or context.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Gemini 3.1 Pro", _gemini_text_model_inputs("HIGH")),
                        IO.DynamicCombo.Option("Gemini 3.1 Flash-Lite", _gemini_text_model_inputs("LOW")),
                    ],
                    tooltip="The Gemini model used to generate the response.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for sampling. Set to 0 for a random seed. Deterministic output isn't guaranteed.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Foundational instructions that dictate the model's behavior.",
                ),
            ],
            outputs=[
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr="""
                (
                  $m := widgets.model;
                  $contains($m, "lite") ? {
                    "type": "list_usd",
                    "usd": [0.00025, 0.0015],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  } : {
                    "type": "list_usd",
                    "usd": [0.002, 0.012],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = GEMINI_V2_MODELS[model["model"]]

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        images = [t for t in (model.get("images") or {}).values() if t is not None]
        audios = [a for a in (model.get("audio") or {}).values() if a is not None]
        videos = [v for v in (model.get("video") or {}).values() if v is not None]
        if images or audios or videos:
            parts.extend(await build_gemini_media_parts(cls, images, audios, videos))
        files = model.get("files")
        if files is not None:
            parts.extend(files)

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model_id}", method="POST"),
            data=GeminiGenerateContentRequest(
                contents=[
                    GeminiContent(
                        role=GeminiRole.user,
                        parts=parts,
                    )
                ],
                generationConfig=GeminiGenerationConfig(
                    temperature=model["temperature"],
                    topP=model["top_p"],
                    maxOutputTokens=model["max_output_tokens"],
                    seed=seed if seed > 0 else None,
                    thinkingConfig=GeminiThinkingConfig(thinkingLevel=model["thinking_level"]),
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )

        output_text = get_text_from_response(response)
        return IO.NodeOutput(output_text or "Empty response from Gemini model...")


class GeminiInputFiles(IO.ComfyNode):
    """
    Loads and formats input files for use with the Gemini API.

    This node allows users to include text (.txt) and PDF (.pdf) files as input
    context for the Gemini model. Files are converted to the appropriate format
    required by the API and can be chained together to include multiple files
    in a single request.
    """

    @classmethod
    def define_schema(cls):
        """
        For details about the supported file input types, see:
        https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf"))
            and f.stat().st_size < GEMINI_MAX_INPUT_FILE_SIZE
        ]
        input_files = sorted(input_files, key=lambda x: x.name)
        input_files = [f.name for f in input_files]
        return IO.Schema(
            node_id="GeminiInputFiles",
            display_name="Gemini Input Files",
            category="partner/text/Gemini",
            description="Loads and prepares input files to include as inputs for Gemini LLM nodes. "
            "The files will be read by the Gemini model when generating a response. "
            "The contents of the text file count toward the token limit. "
            "🛈 TIP: Can be chained together with other Gemini Input File nodes.",
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=input_files,
                    default=input_files[0] if input_files else None,
                    tooltip="Input files to include as context for the model. "
                    "Only accepts text (.txt) and PDF (.pdf) files for now.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "GEMINI_INPUT_FILES",
                    optional=True,
                    tooltip="An optional additional file(s) to batch together with the file loaded from this node. "
                    "Allows chaining of input files so that a single message can include multiple input files.",
                ),
            ],
            outputs=[
                IO.Custom("GEMINI_INPUT_FILES").Output(),
            ],
        )

    @classmethod
    def create_file_part(cls, file_path: str) -> GeminiPart:
        mime_type = GeminiMimeType.application_pdf if file_path.endswith(".pdf") else GeminiMimeType.text_plain
        # Use base64 string directly, not the data URI
        with open(file_path, "rb") as f:
            file_content = f.read()
        base64_str = base64.b64encode(file_content).decode("utf-8")

        return GeminiPart(
            inlineData=GeminiInlineData(
                mimeType=mime_type,
                data=base64_str,
            )
        )

    @classmethod
    def execute(cls, file: str, GEMINI_INPUT_FILES: list[GeminiPart] | None = None) -> IO.NodeOutput:
        """Loads and formats input files for Gemini API."""
        if GEMINI_INPUT_FILES is None:
            GEMINI_INPUT_FILES = []
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = cls.create_file_part(file_path)
        return IO.NodeOutput([input_file_content] + GEMINI_INPUT_FILES)


class GeminiImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiImageNode",
            display_name="Nano Banana (Google Gemini Image)",
            category="partner/image/Gemini",
            description="Edit images synchronously via Google API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt for generation",
                    default="",
                ),
                IO.Combo.Input(
                    "model",
                    options=["gemini-2.5-flash-image"],
                    tooltip="The Gemini model to use for generating responses.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model. "
                    "To include multiple images, you can use the Batch Images node.",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    tooltip="Defaults to matching the output image size to that of your input image, "
                    "or otherwise generates 1:1 squares.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    tooltip="Choose 'IMAGE' for image-only output, or "
                    "'IMAGE+TEXT' to return both the generated image and a text response.",
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.039,"format":{"suffix":"/Image (1K)","approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        aspect_ratio: str = "auto",
        response_modalities: str = "IMAGE+TEXT",
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]

        if not aspect_ratio:
            aspect_ratio = "auto"  # for backward compatability with old workflows; to-do remove this in December
        image_config = GeminiImageConfig() if aspect_ratio == "auto" else GeminiImageConfig(aspectRatio=aspect_ratio)

        if images is not None:
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(await get_image_from_response(response), get_text_from_response(response))


class GeminiImage2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiImage2Node",
            display_name="Nano Banana Pro (Google Gemini Image)",
            category="partner/image/Gemini",
            description="Generate or edit images synchronously via Google Vertex API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or the edits to apply. "
                    "Include any constraints, styles, or details the model should follow.",
                    default="",
                ),
                IO.Combo.Input(
                    "model",
                    options=["gemini-3-pro-image-preview", "Nano Banana 2 (Gemini 3.1 Flash Image)"],
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When the seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    default="auto",
                    tooltip="If set to 'auto', matches your input image's aspect ratio; "
                    "if no image is provided, a 16:9 square is usually generated.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    tooltip="Target output resolution. For 2K/4K the native Gemini upscaler is used.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE+TEXT", "IMAGE"],
                    tooltip="Choose 'IMAGE' for image-only output, or "
                    "'IMAGE+TEXT' to return both the generated image and a text response.",
                    advanced=True,
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). "
                    "To include multiple images, use the Batch Images node (up to 14).",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=GEMINI_IMAGE_2_PRICE_BADGE,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if model == "Nano Banana 2 (Gemini 3.1 Flash Image)":
            model = "gemini-3.1-flash-image-preview"

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(await get_image_from_response(response), get_text_from_response(response))


class GeminiNanoBanana2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiNanoBanana2",
            display_name="Nano Banana 2",
            category="partner/image/Gemini",
            description="Generate or edit images synchronously via Google Vertex API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or the edits to apply. "
                    "Include any constraints, styles, or details the model should follow.",
                    default="",
                ),
                IO.Combo.Input(
                    "model",
                    options=["Nano Banana 2 (Gemini 3.1 Flash Image)"],
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When the seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=[
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    default="auto",
                    tooltip="If set to 'auto', matches your input image's aspect ratio; "
                    "if no image is provided, a 16:9 square is usually generated.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K", "4K"],
                    tooltip="Target output resolution. For 2K/4K the native Gemini upscaler is used.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE", "IMAGE+TEXT"],
                    advanced=True,
                ),
                IO.Combo.Input(
                    "thinking_level",
                    options=["MINIMAL", "HIGH"],
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). "
                    "To include multiple images, use the Batch Images node (up to 14).",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
                IO.Image.Output(
                    display_name="thought_image",
                    tooltip="First image from the model's thinking process. "
                    "Only available with thinking_level HIGH and IMAGE+TEXT modality.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=GEMINI_IMAGE_2_PRICE_BADGE,
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        thinking_level: str,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        if model == "Nano Banana 2 (Gemini 3.1 Flash Image)":
            model = "gemini-3.1-flash-image-preview"

        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                    thinkingConfig=GeminiThinkingConfig(thinkingLevel=thinking_level),
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(
            await get_image_from_response(response),
            get_text_from_response(response),
            await get_image_from_response(response, thought=True),
        )


def _nano_banana_2_v2_model_inputs(resolutions: list[str]):
    return [
        IO.Combo.Input(
            "aspect_ratio",
            options=[
                "auto",
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
                "1:4",
                "4:1",
                "8:1",
                "1:8",
            ],
            default="auto",
            tooltip="If set to 'auto', matches your input image's aspect ratio; "
            "if no image is provided, a 16:9 square is usually generated.",
        ),
        IO.Combo.Input(
            "resolution",
            options=resolutions,
            tooltip="Target output resolution.",
        ),
        IO.Combo.Input(
            "thinking_level",
            options=["MINIMAL", "HIGH"],
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 15)],
                min=0,
            ),
            tooltip="Optional reference image(s). Up to 14 images total.",
        ),
        IO.Custom("GEMINI_INPUT_FILES").Input(
            "files",
            optional=True,
            tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
        ),
    ]


class GeminiNanoBanana2V2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiNanoBanana2V2",
            display_name="Nano Banana 2",
            category="partner/image/Gemini",
            description="Generate or edit images synchronously via Google Vertex API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or the edits to apply. "
                    "Include any constraints, styles, or details the model should follow.",
                    default="",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "Nano Banana 2 (Gemini 3.1 Flash Image)",
                            _nano_banana_2_v2_model_inputs(resolutions=["1K", "2K", "4K"]),
                        ),
                        IO.DynamicCombo.Option(
                            "Nano Banana 2 Lite",
                            _nano_banana_2_v2_model_inputs(resolutions=["1K"]),
                        ),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When the seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE", "IMAGE+TEXT"],
                    advanced=True,
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                    advanced=True,
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    tooltip="Controls randomness in generation. Lower is more focused/deterministic.",
                    advanced=True,
                ),
                IO.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Nucleus sampling threshold. Lower is more focused, higher more diverse.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
                IO.Image.Output(
                    display_name="thought_image",
                    tooltip="First image from the model's thinking process. "
                    "Only available with thinking_level HIGH and IMAGE+TEXT modality.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.resolution"]),
                expr="""
                (
                  $contains(widgets.model, "lite")
                    ? {"type":"usd","usd": 0.034, "format":{"suffix":"/Image","approximate":true}}
                    : (
                        $r := $lookup(widgets, "model.resolution");
                        $prices := {"1k": 0.0696, "2k": 0.1014, "4k": 0.154};
                        {"type":"usd","usd": $lookup($prices, $r), "format":{"suffix":"/Image","approximate":true}}
                      )
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
        response_modalities: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_choice = model["model"]
        if model_choice == "Nano Banana 2 (Gemini 3.1 Flash Image)":
            model_id = "gemini-3.1-flash-image-preview"
        elif model_choice == "Nano Banana 2 Lite":
            model_id = "gemini-3.1-flash-lite-image"
        else:
            model_id = model_choice

        images = model.get("images") or {}
        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images:
            image_tensors: list[Input.Image] = [t for t in images.values() if t is not None]
            if image_tensors:
                if sum(get_number_of_images(t) for t in image_tensors) > 14:
                    raise ValueError("The current maximum number of supported images is 14.")
                parts.extend(await create_image_parts(cls, image_tensors))
        files = model.get("files")
        if files is not None:
            parts.extend(files)

        image_config = GeminiImageConfig(imageSize=model["resolution"])
        if model["aspect_ratio"] != "auto":
            image_config.aspectRatio = model["aspect_ratio"]

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model_id}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                    thinkingConfig=GeminiThinkingConfig(thinkingLevel=model["thinking_level"]),
                    temperature=temperature,
                    topP=top_p,
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(
            await get_image_from_response(response),
            get_text_from_response(response),
            await get_image_from_response(response, thought=True),
        )


OMNI_MAX_IMAGES = 14
OMNI_MAX_VIDEOS = 3

OMNI_MODELS: dict[str, str] = {
    "Omni Flash": "gemini-omni-flash-preview",
}


def _omni_flash_inputs() -> list[Input]:
    """Per-model inputs for the Omni video DynamicCombo (prompt + reference media + sampling)."""
    return [
        IO.String.Input(
            "prompt",
            multiline=True,
            default="",
            tooltip="Describe the video to generate. Specify the length and aspect ratio directly in the "
            'prompt, e.g. "a 6-second clip in 16:9". Length may be 3-10 seconds; the aspect ratio must be '
            "16:9 (landscape) or 9:16 (portrait). The output is 720p, 24 FPS, with audio.",
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, OMNI_MAX_IMAGES + 1)],
                min=0,
            ),
            tooltip=f"Optional reference image(s) to guide or animate the video. Up to {OMNI_MAX_IMAGES} images.",
        ),
        IO.Autogrow.Input(
            "videos",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("video"),
                names=[f"video_{i}" for i in range(1, OMNI_MAX_VIDEOS + 1)],
                min=0,
            ),
            tooltip=f"Optional reference video(s) to guide or edit. Up to {OMNI_MAX_VIDEOS} videos, "
            f"each up to 10 seconds long.",
        ),
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.01,
            tooltip="Controls randomness. Lower is more focused/deterministic, higher is more varied.",
            advanced=True,
        ),
        IO.Float.Input(
            "top_p",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Nucleus sampling: sample from the smallest token set whose cumulative probability reaches top_p.",
            advanced=True,
        ),
    ]


class GeminiVideoOmni(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GeminiVideoOmni",
            display_name="Google Gemini Omni (Video)",
            category="partner/video/Gemini",
            essentials_category="Video Generation",
            description="Generate a video with audio from a text prompt using Google's Gemini Omni Flash model. "
            "Optionally provide reference images and/or videos to guide or edit the result. Describe the desired "
            "length (3-10s) and aspect ratio (16:9 or 9:16) directly in the prompt.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Omni Flash", _omni_flash_inputs()),
                    ],
                    tooltip="The Gemini video model used to generate the video.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Video.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.146,"format":{"suffix":"/second","approximate":true}}'
            ),
        )

    @classmethod
    async def execute(cls, model: dict, seed: int) -> IO.NodeOutput:
        prompt = model.get("prompt") or ""
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_id = OMNI_MODELS[model["model"]]

        images = [t for t in (model.get("images") or {}).values() if t is not None]
        videos = [v for v in (model.get("videos") or {}).values() if v is not None]
        if sum(get_number_of_images(t) for t in images) > OMNI_MAX_IMAGES:
            raise ValueError(f"The current maximum number of supported images is {OMNI_MAX_IMAGES}.")
        if len(videos) > OMNI_MAX_VIDEOS:
            raise ValueError(f"The current maximum number of supported videos is {OMNI_MAX_VIDEOS}.")
        for video in videos:
            validate_video_duration(video, max_duration=10)

        parts: list[GeminiPart] = []
        if images or videos:
            parts.extend(await build_gemini_media_parts(cls, images, [], videos))
        parts.append(GeminiPart(text=prompt))
        response = await sync_op(
            cls,
            ApiEndpoint(path=f"{GEMINI_BASE_ENDPOINT}/{model_id}", method="POST"),
            data=GeminiGenerateContentRequest(
                contents=[GeminiContent(role=GeminiRole.user, parts=parts)],
                generationConfig=GeminiGenerationConfig(
                    responseModalities=["TEXT", "VIDEO"],
                    temperature=model.get("temperature", 1.0),
                    topP=model.get("top_p", 0.95),
                ),
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(
            await get_video_from_response(response, cls=cls),
            get_text_from_response(response),
        )


class GeminiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            GeminiNode,
            GeminiNodeV2,
            GeminiImage,
            GeminiImage2,
            GeminiNanoBanana2,
            GeminiNanoBanana2V2,
            GeminiVideoOmni,
            GeminiInputFiles,
        ]


async def comfy_entrypoint() -> GeminiExtension:
    return GeminiExtension()
