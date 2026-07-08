"""API Nodes for OpenRouter LLM chat completions."""

from dataclasses import dataclass
from typing import Literal

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.openrouter import (
    OpenRouterChatRequest,
    OpenRouterChatResponse,
    OpenRouterContentBlock,
    OpenRouterImageContent,
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
    get_number_of_images,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)

OPENROUTER_CHAT_ENDPOINT = "/proxy/openrouter/api/v1/chat/completions"


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
    _ModelSpec("anthropic/claude-opus-4.7", "frontier_reasoning", 0.000005, 0.000025, max_images=20),
    _ModelSpec("openai/gpt-5.5-pro", "frontier_reasoning", 0.00003, 0.00018, max_images=20),
    _ModelSpec("openai/gpt-5.5", "frontier_reasoning", 0.000005, 0.00003, max_images=20),
    _ModelSpec("google/gemini-3.5-flash", "reasoning", 0.0000015, 0.000009, max_images=20, max_videos=4),
    _ModelSpec("x-ai/grok-4.20", "reasoning", 0.00000125, 0.0000025, max_images=20),
    _ModelSpec("x-ai/grok-4.3", "reasoning", 0.00000125, 0.0000025, max_images=20),
    _ModelSpec("deepseek/deepseek-v4-pro", "reasoning", 0.000000435, 0.00000087),
    _ModelSpec("deepseek/deepseek-v4-flash", "reasoning", 0.000000112, 0.000000224),
    _ModelSpec("deepseek/deepseek-v3.2", "reasoning", 0.000000252, 0.000000378),
    _ModelSpec("qwen/qwen3.6-max-preview", "reasoning", 0.00000104, 0.00000624),
    _ModelSpec("qwen/qwen3.6-plus", "reasoning", 0.000000325, 0.00000195, max_images=10, max_videos=4),
    _ModelSpec("qwen/qwen3.6-flash", "reasoning", 0.0000001875, 0.000001125, max_images=10, max_videos=4),
    _ModelSpec("mistralai/mistral-large-2512", "standard", 0.0000005, 0.0000015, max_images=8),
    _ModelSpec("mistralai/mistral-medium-3-5", "reasoning", 0.0000015, 0.0000075, max_images=8),
    _ModelSpec("z-ai/glm-4.6", "reasoning", 0.00000043, 0.00000174),
    _ModelSpec("z-ai/glm-5", "reasoning", 0.0000006, 0.00000192),
    _ModelSpec("moonshotai/kimi-k2.6", "reasoning", 0.00000073, 0.00000349, max_images=10),
    _ModelSpec("moonshotai/kimi-k2-thinking", "reasoning", 0.0000006, 0.0000025),
    _ModelSpec("perplexity/sonar-pro", "perplexity", 0.000003, 0.000015),
    _ModelSpec("perplexity/sonar-reasoning-pro", "perplexity_reasoning", 0.000002, 0.000008),
    _ModelSpec("perplexity/sonar-deep-research", "perplexity_reasoning", 0.000002, 0.000008),
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


def _calculate_price(response: OpenRouterChatResponse) -> float | None:
    if response.usage and response.usage.cost is not None:
        return float(response.usage.cost)
    return None


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


def _extract_text(response: OpenRouterChatResponse) -> str:
    if response.error:
        code = response.error.code if response.error.code is not None else "unknown"
        raise ValueError(f"OpenRouter error ({code}): {response.error.message or 'no message'}")
    if not response.choices:
        raise ValueError("Empty response from OpenRouter (no choices).")
    message = response.choices[0].message
    if not message:
        raise ValueError("Empty response from OpenRouter (no message).")
    if message.refusal:
        raise ValueError(f"Model refused to respond: {message.refusal}")
    return message.content or ""


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
                "models from xAI, DeepSeek, Qwen, Mistral, Z.AI (GLM), Moonshot (Kimi), and "
                "Perplexity Sonar."
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
            price_extractor=_calculate_price,
        )
        return IO.NodeOutput(_extract_text(response))


class OpenRouterExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [OpenRouterLLMNode]


async def comfy_entrypoint() -> OpenRouterExtension:
    return OpenRouterExtension()
