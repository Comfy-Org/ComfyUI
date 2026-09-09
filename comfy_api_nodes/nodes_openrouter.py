"""API Nodes for OpenRouter chat completions: LLM text generation and image generation."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.openrouter import (
    OpenRouterChatRequest,
    OpenRouterChatResponse,
    OpenRouterContentBlock,
    OpenRouterError,
    OpenRouterImageContent,
    OpenRouterImageData,
    OpenRouterImageRequest,
    OpenRouterImageResponse,
    OpenRouterImageUrl,
    OpenRouterMessage,
    OpenRouterReasoningConfig,
    OpenRouterTextContent,
    OpenRouterVideoContent,
    OpenRouterVideoUrl,
    OpenRouterWebSearchOptions,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    get_number_of_images,
    pad_images_to_common_channels,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)

OPENROUTER_CHAT_ENDPOINT = "/proxy/openrouter/api/v1/chat/completions"
OPENROUTER_IMAGES_ENDPOINT = "/proxy/openrouter/api/v1/images"


Profile = Literal["standard", "reasoning", "frontier_reasoning", "perplexity", "perplexity_reasoning"]


@dataclass(frozen=True)
class _ModelSpec:
    slug: str  # exact OpenRouter model id
    profile: Profile
    price_in: float  # USD per token (prompt)
    price_out: float  # USD per token (completion)
    max_images: int = 0  # 0 = no image input; otherwise max URL-passed images supported
    max_videos: int = 0  # 0 = no video input; otherwise max URL-passed videos supported


MODELS: list[_ModelSpec] = [
    _ModelSpec("anthropic/claude-opus-5", "frontier_reasoning", 0.00000715, 0.00003575, max_images=20),
    _ModelSpec("anthropic/claude-opus-4.8", "frontier_reasoning", 0.00000715, 0.00003575, max_images=20),
    _ModelSpec("anthropic/claude-opus-4.7", "frontier_reasoning", 0.00000715, 0.00003575, max_images=20),
    _ModelSpec("anthropic/claude-fable-5", "frontier_reasoning", 0.0000143, 0.0000715, max_images=20),
    _ModelSpec("anthropic/claude-sonnet-5", "frontier_reasoning", 0.00000286, 0.0000143, max_images=20),
    _ModelSpec("anthropic/claude-haiku-4.5", "frontier_reasoning", 0.00000143, 0.00000715, max_images=20),
    _ModelSpec("openai/gpt-5.6-sol-pro", "frontier_reasoning", 0.00000715, 0.0000429, max_images=20),
    _ModelSpec("openai/gpt-5.6-sol", "frontier_reasoning", 0.00000715, 0.0000429, max_images=20),
    _ModelSpec("openai/gpt-5.6-terra-pro", "frontier_reasoning", 0.000003575, 0.00002145, max_images=20),
    _ModelSpec("openai/gpt-5.6-terra", "frontier_reasoning", 0.000003575, 0.00002145, max_images=20),
    _ModelSpec("openai/gpt-5.6-luna-pro", "frontier_reasoning", 0.00000143, 0.00000858, max_images=20),
    _ModelSpec("openai/gpt-5.6-luna", "frontier_reasoning", 0.00000143, 0.00000858, max_images=20),
    _ModelSpec("openai/gpt-5.5-pro", "frontier_reasoning", 0.0000429, 0.0002574, max_images=20),
    _ModelSpec("openai/gpt-5.5", "frontier_reasoning", 0.00000715, 0.0000429, max_images=20),
    _ModelSpec("google/gemini-3.5-flash", "reasoning", 0.000002145, 0.00001287, max_images=20, max_videos=4),
    _ModelSpec("x-ai/grok-4.5", "reasoning", 0.00000286, 0.00000858, max_images=20),
    _ModelSpec("x-ai/grok-4.20", "reasoning", 0.0000017875, 0.000003575, max_images=20),
    _ModelSpec("x-ai/grok-4.3", "reasoning", 0.0000017875, 0.000003575, max_images=20),
    _ModelSpec("deepseek/deepseek-v4-pro", "reasoning", 0.00000062205, 0.0000012441),
    _ModelSpec("deepseek/deepseek-v4-flash", "reasoning", 0.00000016016, 0.00000032032),
    _ModelSpec("deepseek/deepseek-v3.2", "reasoning", 0.00000036036, 0.00000054054),
    _ModelSpec("qwen/qwen3.6-max-preview", "reasoning", 0.0000014872, 0.0000089232),
    _ModelSpec("qwen/qwen3.6-plus", "reasoning", 0.00000046475, 0.0000027885, max_images=10, max_videos=4),
    _ModelSpec("qwen/qwen3.6-flash", "reasoning", 0.000000268125, 0.00000160875, max_images=10, max_videos=4),
    _ModelSpec("mistralai/mistral-large-2512", "standard", 0.000000715, 0.000002145, max_images=8),
    _ModelSpec("mistralai/mistral-medium-3-5", "reasoning", 0.000002145, 0.000010725, max_images=8),
    _ModelSpec("z-ai/glm-4.6", "reasoning", 0.0000006149, 0.0000024882),
    _ModelSpec("z-ai/glm-5", "reasoning", 0.000000858, 0.0000027456),
    _ModelSpec("moonshotai/kimi-k3", "reasoning", 0.00000429, 0.00002145, max_images=10),
    _ModelSpec("moonshotai/kimi-k2.6", "reasoning", 0.0000010439, 0.0000049907, max_images=10),
    _ModelSpec("moonshotai/kimi-k2-thinking", "reasoning", 0.000000858, 0.000003575),
    _ModelSpec("perplexity/sonar-pro", "perplexity", 0.00000429, 0.00002145),
    _ModelSpec("perplexity/sonar-reasoning-pro", "perplexity_reasoning", 0.00000286, 0.00001144),
    _ModelSpec("perplexity/sonar-deep-research", "perplexity_reasoning", 0.00000286, 0.00001144),
]

_MODELS_BY_SLUG: dict[str, _ModelSpec] = {m.slug: m for m in MODELS}
_REASONING_EFFORTS = ["off", "low", "medium", "high"]
_SEARCH_CONTEXT_SIZES = ["low", "medium", "high"]


def _reasoning_extra_inputs() -> list:
    return [
        IO.Combo.Input(
            "reasoning_effort",
            options=_REASONING_EFFORTS,
            default="off",
            tooltip="Reasoning effort. 'off' disables reasoning entirely.",
            advanced=True,
        ),
    ]


def _perplexity_extra_inputs() -> list:
    return [
        IO.Combo.Input(
            "search_context_size",
            options=_SEARCH_CONTEXT_SIZES,
            default="medium",
            tooltip="How much web search context to retrieve. Larger = more grounded but slower/pricier.",
            advanced=True,
        ),
    ]


def _profile_inputs(profile: Profile) -> list:
    if profile == "standard":
        return []
    if profile in ("reasoning", "frontier_reasoning"):
        return _reasoning_extra_inputs()
    if profile == "perplexity":
        return _perplexity_extra_inputs()
    if profile == "perplexity_reasoning":
        return _perplexity_extra_inputs() + _reasoning_extra_inputs()
    raise ValueError(f"Unknown profile: {profile}")


def _media_inputs(spec: _ModelSpec) -> list:
    extras: list = []
    if spec.max_images > 0:
        extras.append(
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=[f"image_{i}" for i in range(1, spec.max_images + 1)],
                    min=0,
                ),
                tooltip=f"Optional reference image(s) — up to {spec.max_images}. Sent as URLs.",
            )
        )
    if spec.max_videos > 0:
        extras.append(
            IO.Autogrow.Input(
                "videos",
                template=IO.Autogrow.TemplateNames(
                    IO.Video.Input("video"),
                    names=[f"video_{i}" for i in range(1, spec.max_videos + 1)],
                    min=0,
                ),
                tooltip=f"Optional reference video(s) — up to {spec.max_videos}. Sent as URLs.",
            )
        )
    return extras


def _inputs_for_model(spec: _ModelSpec) -> list:
    return _profile_inputs(spec.profile) + _media_inputs(spec)


def _build_model_options() -> list[IO.DynamicCombo.Option]:
    return [IO.DynamicCombo.Option(spec.slug, _inputs_for_model(spec)) for spec in MODELS]


def _price_badge_jsonata() -> str:
    rates_pairs = []
    for spec in MODELS:
        prompt_per_1k = spec.price_in * 1000
        completion_per_1k = spec.price_out * 1000
        rates_pairs.append(f'  "{spec.slug}": [{prompt_per_1k:.8g}, {completion_per_1k:.8g}]')
    rates_block = ",\n".join(rates_pairs)
    return (
        "(\n"
        "  $rates := {\n"
        f"{rates_block}\n"
        "  };\n"
        "  $r := $lookup($rates, widgets.model);\n"
        "  $r ? {\n"
        '    "type": "list_usd",\n'
        '    "usd": $r,\n'
        '    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }\n'
        '  } : {"type": "text", "text": "Token-based"}\n'
        ")"
    )


async def _build_image_blocks(
    cls: type[IO.ComfyNode], spec: _ModelSpec, images: list[Input.Image]
) -> list[OpenRouterImageContent]:
    urls = await upload_images_to_comfyapi(
        cls,
        images,
        max_images=spec.max_images,
        total_pixels=2048 * 2048,
        mime_type="image/png",
        wait_label="Uploading reference images",
    )
    return [OpenRouterImageContent(image_url=OpenRouterImageUrl(url=url)) for url in urls]


async def _build_video_blocks(cls: type[IO.ComfyNode], videos: list[Input.Video]) -> list[OpenRouterVideoContent]:
    blocks: list[OpenRouterVideoContent] = []
    total = len(videos)
    for idx, video in enumerate(videos):
        label = "Uploading reference video"
        if total > 1:
            label = f"{label} ({idx + 1}/{total})"
        url = await upload_video_to_comfyapi(cls, video, wait_label=label)
        blocks.append(OpenRouterVideoContent(video_url=OpenRouterVideoUrl(url=url)))
    return blocks


def _user_message(prompt: str, media_blocks: list[OpenRouterContentBlock]) -> OpenRouterMessage:
    if not media_blocks:
        return OpenRouterMessage(role="user", content=prompt)
    blocks: list[OpenRouterContentBlock] = list(media_blocks)
    blocks.append(OpenRouterTextContent(text=prompt))
    return OpenRouterMessage(role="user", content=blocks)


def _build_messages(
    system_prompt: str, prompt: str, media_blocks: list[OpenRouterContentBlock]
) -> list[OpenRouterMessage]:
    messages: list[OpenRouterMessage] = []
    if system_prompt:
        messages.append(OpenRouterMessage(role="system", content=system_prompt))
    messages.append(_user_message(prompt, media_blocks))
    return messages


def _build_request(
    slug: str,
    system_prompt: str,
    prompt: str,
    media_blocks: list[OpenRouterContentBlock],
    *,
    seed: int,
    reasoning_effort: str | None,
    search_context_size: str | None,
) -> OpenRouterChatRequest:
    reasoning_cfg: OpenRouterReasoningConfig | None = None
    if reasoning_effort and reasoning_effort != "off":
        # exclude=True asks providers to reason internally but not return the trace
        reasoning_cfg = OpenRouterReasoningConfig(effort=reasoning_effort, exclude=True)
    web_search_cfg: OpenRouterWebSearchOptions | None = None
    if search_context_size:
        web_search_cfg = OpenRouterWebSearchOptions(search_context_size=search_context_size)
    return OpenRouterChatRequest(
        model=slug,
        messages=_build_messages(system_prompt, prompt, media_blocks),
        seed=seed if seed > 0 else None,
        reasoning=reasoning_cfg,
        web_search_options=web_search_cfg,
    )


def _raise_on_error(error: OpenRouterError | None) -> None:
    if error:
        code = error.code if error.code is not None else "unknown"
        raise ValueError(f"OpenRouter error ({code}): {error.message or 'no message'}")


def _extract_text(response: OpenRouterChatResponse) -> str:
    _raise_on_error(response.error)
    if not response.choices:
        raise ValueError("Empty response from OpenRouter (no choices).")
    message = response.choices[0].message
    if not message:
        raise ValueError("Empty response from OpenRouter (no message).")
    if message.refusal:
        raise ValueError(f"Model refused to respond: {message.refusal}")
    return message.content or ""


def _image_data_to_tensor(item: OpenRouterImageData) -> torch.Tensor:
    try:
        return bytesio_to_image_tensor(BytesIO(base64.b64decode(item.b64_json)))
    except Exception as e:
        raise ValueError(f"OpenRouter returned an image that could not be decoded: {e}") from e


def _extract_images(response: OpenRouterImageResponse) -> torch.Tensor:
    _raise_on_error(response.error)
    tensors = [_image_data_to_tensor(item) for item in response.data or [] if item.b64_json]
    if not tensors:
        raise ValueError("OpenRouter returned no image.")
    return torch.cat(pad_images_to_common_channels(tensors))


class OpenRouterLLMNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenRouterLLMNode",
            display_name="OpenRouter LLM",
            category="partner/text/OpenRouter",
            essentials_category="Text Generation",
            description=(
                "Generate text responses through OpenRouter. Routes to a curated set of popular "
                "models from Anthropic (Claude), OpenAI (GPT), Google (Gemini), xAI (Grok), "
                "DeepSeek, Qwen, Mistral, Z.AI (GLM), Moonshot (Kimi), and Perplexity Sonar."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text input to the model.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=_build_model_options(),
                    tooltip="The OpenRouter model used to generate the response.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed for sampling. Set to 0 to omit. Most models treat this as a hint only.",
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
            outputs=[IO.String.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model"]),
                expr=_price_badge_jsonata(),
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
        slug: str = model["model"]
        spec = _MODELS_BY_SLUG.get(slug)
        if spec is None:
            raise ValueError(f"Unknown OpenRouter model: {slug}")

        reasoning_effort: str | None = model.get("reasoning_effort")
        search_context_size: str | None = model.get("search_context_size")

        image_tensors: list[Input.Image] = [t for t in (model.get("images") or {}).values() if t is not None]
        if image_tensors and sum(get_number_of_images(t) for t in image_tensors) > spec.max_images:
            raise ValueError(f"Up to {spec.max_images} images are supported for {slug}.")
        video_inputs: list[Input.Video] = [v for v in (model.get("videos") or {}).values() if v is not None]
        if video_inputs and len(video_inputs) > spec.max_videos:
            raise ValueError(f"Up to {spec.max_videos} videos are supported for {slug}.")

        media_blocks: list[OpenRouterContentBlock] = []
        if image_tensors:
            media_blocks.extend(await _build_image_blocks(cls, spec, image_tensors))
        if video_inputs:
            media_blocks.extend(await _build_video_blocks(cls, video_inputs))

        request = _build_request(
            slug,
            system_prompt,
            prompt,
            media_blocks,
            seed=seed,
            reasoning_effort=reasoning_effort,
            search_context_size=search_context_size,
        )

        response = await sync_op(
            cls,
            ApiEndpoint(path=OPENROUTER_CHAT_ENDPOINT, method="POST"),
            response_model=OpenRouterChatResponse,
            data=request,
        )
        return IO.NodeOutput(_extract_text(response))


@dataclass(frozen=True)
class _ImageModelSpec:
    slug: str
    price_text: float
    price_image_in: float
    price_image_out: float


IMAGE_MODELS: list[_ImageModelSpec] = [
    _ImageModelSpec("microsoft/mai-image-2.6", 0.000005, 0.000008, 0.000038),
    _ImageModelSpec("microsoft/mai-image-2.6-flash", 0.00000175, 0.0000025, 0.000019),
]

_IMAGE_MODELS_BY_SLUG: dict[str, _ImageModelSpec] = {m.slug: m for m in IMAGE_MODELS}
_IMAGE_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "1K": {
        "1:1": (1024, 1024),
        "16:9": (1360, 768),
        "9:16": (768, 1360),
        "3:2": (1152, 768),
        "2:3": (768, 1152),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
    },
    "1.5K": {
        "1:1": (1536, 1536),
        "16:9": (2048, 1152),
        "9:16": (1152, 2048),
        "3:2": (1872, 1248),
        "2:3": (1248, 1872),
        "4:3": (1760, 1312),
        "3:4": (1312, 1760),
    },
}
_IMAGE_ASPECT_RATIOS = list(_IMAGE_SIZES["1K"])
_IMAGE_PROMPT_MAX_CHARS = 20000
_IMAGE_MAX_REFERENCES = 5
_IMAGE_REFERENCE_MAX_PIXELS = 2048 * 2048
_IMAGE_REFERENCE_TEXT_OVERHEAD_TOKENS = 256


def _image_tokens(width: int, height: int) -> int:
    return width * height // 1024


def _image_model_option(spec: _ImageModelSpec) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        spec.slug,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Describes the image to generate, or the edit to apply to the reference images. "
                f"Up to {_IMAGE_PROMPT_MAX_CHARS} characters.",
            ),
            IO.Combo.Input(
                "aspect_ratio",
                options=_IMAGE_ASPECT_RATIOS,
                default="1:1",
                tooltip="Aspect ratio of the generated image, also applied when reference images are connected.",
            ),
            IO.Combo.Input(
                "resolution",
                options=list(_IMAGE_SIZES),
                default="1K",
                tooltip="Output size tier. 1K is about 1 megapixel (1:1 is 1024x1024, 16:9 is 1360x768); "
                "1.5K is about 2.3 megapixels (1:1 is 1536x1536, 16:9 is 2048x1152).",
            ),
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=[f"image_{i}" for i in range(1, _IMAGE_MAX_REFERENCES + 1)],
                    min=0,
                ),
                tooltip=f"Up to {_IMAGE_MAX_REFERENCES} reference images for image-guided editing; "
                "a batched input counts once per image.",
            ),
            IO.Int.Input(
                "seed",
                default=42,
                min=0,
                max=2147483647,
                step=1,
                display_mode=IO.NumberDisplay.number,
                control_after_generate=True,
                tooltip="Seed to determine if node should re-run; the API has no seed, "
                "so actual results are nondeterministic regardless of this value.",
            ),
        ],
    )


def _image_price_badge_jsonata() -> str:
    rates_pairs = []
    for spec in IMAGE_MODELS:
        per_million = [spec.price_text * 1e6, spec.price_image_in * 1e6, spec.price_image_out * 1e6]
        rates_pairs.append(f'    "{spec.slug}": [{", ".join(f"{p:.8g}" for p in per_million)}]')
    rates_block = ",\n".join(rates_pairs)
    size_tables = []
    for tier, table in _IMAGE_SIZES.items():
        ratio_tokens = ", ".join(f'"{ratio}": {_image_tokens(w, h)}' for ratio, (w, h) in table.items())
        size_tables.append(f'"{tier.lower()}": {{{ratio_tokens}}}')
    default_out = _image_tokens(*_IMAGE_SIZES["1K"]["1:1"])
    ref_max_tokens = _IMAGE_REFERENCE_MAX_PIXELS // 1024
    return (
        "(\n"
        "  $rates := {\n"
        f"{rates_block}\n"
        "  };\n"
        f"  $outTokens := {{{', '.join(size_tables)}}};\n"
        "  $r := $lookup($rates, widgets.model);\n"
        '  $ar := $lookup(widgets, "model.aspect_ratio");\n'
        '  $res := $lookup(widgets, "model.resolution");\n'
        '  $prompt := $lookup(widgets, "model.prompt");\n'
        '  $links := $lookup(inputGroups, "model.images");\n'
        '  $refs := $type($links) = "number" ? $links : 0;\n'
        '  $promptTokens := $type($prompt) = "string"\n'
        "    ? ($length($prompt) + 2 * $count($match($prompt, /[^\\x00-\\x7F]/))) / 4 : 0;\n"
        '  $table := $type($res) = "string" ? $lookup($outTokens, $res) : null;\n'
        '  $sized := ($type($table) = "object" and $type($ar) = "string") ? $lookup($table, $ar) : null;\n'
        f'  $out := $type($sized) = "number" ? $sized : {default_out};\n'
        "  $r ? ($refs > 0 ? {\n"
        '    "type": "range_usd",\n'
        '    "min_usd": ($promptTokens * $r[0] + $out * $r[2]) * 1.43 / 1000000,\n'
        f'    "max_usd": (($promptTokens + $refs * {_IMAGE_REFERENCE_TEXT_OVERHEAD_TOKENS}) * $r[0]'
        f" + $refs * {ref_max_tokens} * $r[1] + $out * $r[2]) * 1.43 / 1000000,\n"
        '    "format": {"approximate": true}\n'
        "  } : {\n"
        '    "type": "usd",\n'
        '    "usd": ($promptTokens * $r[0] + $out * $r[2]) * 1.43 / 1000000,\n'
        '    "format": {"approximate": true}\n'
        '  }) : {"type": "text", "text": "Token-based"}\n'
        ")"
    )


class OpenRouterImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenRouterImageNode",
            display_name="OpenRouter Image",
            category="partner/image/OpenRouter",
            description=(
                "Generate or edit images through OpenRouter with Microsoft's MAI-Image-2.6 models: "
                "text to image, or image-guided editing with up to five reference images, "
                "in seven aspect ratios at 1K or 1.5K."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_image_model_option(spec) for spec in IMAGE_MODELS],
                    tooltip="The OpenRouter image model used to generate the image.",
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "model.aspect_ratio", "model.resolution", "model.prompt"],
                    input_groups=["model.images"],
                ),
                expr=_image_price_badge_jsonata(),
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        slug: str = model["model"]
        if slug not in _IMAGE_MODELS_BY_SLUG:
            raise ValueError(f"Unknown OpenRouter model: {slug}")
        prompt: str = model["prompt"]
        validate_string(prompt, strip_whitespace=True, min_length=1)
        validate_string(prompt, strip_whitespace=False, max_length=_IMAGE_PROMPT_MAX_CHARS)
        width, height = _IMAGE_SIZES[model["resolution"]][model["aspect_ratio"]]

        reference_images = [
            image for images in (model.get("images") or {}).values() if images is not None for image in images
        ]
        if len(reference_images) > _IMAGE_MAX_REFERENCES:
            raise ValueError(
                f"A maximum of {_IMAGE_MAX_REFERENCES} reference images is supported; got {len(reference_images)} "
                "(a batched input counts once per image)."
            )
        input_references: list[OpenRouterImageContent] | None = None
        if reference_images:
            urls = await upload_images_to_comfyapi(
                cls,
                [image[..., :3] for image in reference_images],
                max_images=_IMAGE_MAX_REFERENCES,
                mime_type="image/png",
                total_pixels=_IMAGE_REFERENCE_MAX_PIXELS,
                wait_label="Uploading reference images",
            )
            input_references = [OpenRouterImageContent(image_url=OpenRouterImageUrl(url=url)) for url in urls]

        response = await sync_op(
            cls,
            ApiEndpoint(path=OPENROUTER_IMAGES_ENDPOINT, method="POST"),
            response_model=OpenRouterImageResponse,
            data=OpenRouterImageRequest(
                model=slug,
                prompt=prompt,
                size=f"{width}x{height}",
                input_references=input_references,
            ),
        )
        return IO.NodeOutput(_extract_images(response))


class OpenRouterExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [OpenRouterLLMNode, OpenRouterImageNode]


async def comfy_entrypoint() -> OpenRouterExtension:
    return OpenRouterExtension()
