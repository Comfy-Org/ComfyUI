"""API Nodes for Anthropic Claude (Messages API). See: https://docs.anthropic.com/en/api/messages"""

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.anthropic import (
    AnthropicImageContent,
    AnthropicImageSourceUrl,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicOutputConfig,
    AnthropicResponseTextBlock,
    AnthropicRole,
    AnthropicTextContent,
    AnthropicThinkingConfig,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    get_number_of_images,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)

ANTHROPIC_MESSAGES_ENDPOINT = "/proxy/anthropic/v1/messages"
ANTHROPIC_IMAGE_MAX_PIXELS = 1568 * 1568
CLAUDE_MAX_IMAGES = 20

CLAUDE_MODELS: dict[str, str] = {
    "Opus 5": "claude-opus-5",
    "Opus 4.8": "claude-opus-4-8",
    "Fable 5.1": "claude-fable-5-1",
    "Fable 5": "claude-fable-5",
    "Sonnet 5": "claude-sonnet-5",
    "Opus 4.7": "claude-opus-4-7",
    "Opus 4.6": "claude-opus-4-6",
    "Sonnet 4.6": "claude-sonnet-4-6",
    "Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Haiku 4.5": "claude-haiku-4-5-20251001",
}

_THINKING_UNSUPPORTED = {"Haiku 4.5"}
# Models that use the newer "adaptive" thinking mode (Opus 4.7+ require it; older models keep the explicit budget API).
# Anthropic decides the actual budget when adaptive is used, based on the `output_config.effort` hint.
_ADAPTIVE_THINKING_MODELS = {"Opus 4.8", "Sonnet 5", "Opus 4.7", "Opus 4.6", "Sonnet 4.6"}
_ALWAYS_THINKING_MODELS = {"Opus 5", "Fable 5.1", "Fable 5"}
_XHIGH_EFFORT_MODELS = {"Opus 5", "Opus 4.8", "Fable 5.1", "Fable 5", "Sonnet 5", "Opus 4.7"}
_MAX_EFFORT_MODELS = _XHIGH_EFFORT_MODELS | {"Opus 4.6", "Sonnet 4.6"}
_EXPLICIT_THINKING_OFF_MODELS = {"Sonnet 5"}
_NO_TEMPERATURE_MODELS = {"Opus 5", "Opus 4.8", "Fable 5.1", "Fable 5", "Sonnet 5"}

# Budget mode (Sonnet 4.5): effort -> reasoning budget in tokens. Must be < max_tokens.
# Sized so even the "high" budget fits comfortably under the default max_tokens=32768.
_REASONING_BUDGET: dict[str, int] = {
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}
_REASONING_EFFORTS = ["off", "low", "medium", "high"]


def _reasoning_effort_options(model_label: str) -> list[str]:
    options = list(_REASONING_EFFORTS)
    if model_label in _ALWAYS_THINKING_MODELS:
        options.remove("off")
    if model_label in _XHIGH_EFFORT_MODELS:
        options.append("xhigh")
    if model_label in _MAX_EFFORT_MODELS:
        options.append("max")
    return options


def _claude_model_inputs(model_label: str):
    inputs: list = [
        IO.Int.Input(
            "max_tokens",
            default=32768,
            min=4096,
            max=64000,
            tooltip="Maximum number of tokens to generate (includes reasoning tokens when enabled).",
            advanced=True,
        ),
    ]
    if model_label not in _NO_TEMPERATURE_MODELS:
        inputs.append(
            IO.Float.Input(
                "temperature",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip=(
                    "Controls randomness. 0.0 is deterministic, 1.0 is most random. "
                    "Ignored for Opus 4.7 and any model when reasoning_effort is set."
                ),
                advanced=True,
            )
        )
    if model_label in _ALWAYS_THINKING_MODELS:
        inputs.append(
            IO.Combo.Input(
                "reasoning_effort",
                options=_reasoning_effort_options(model_label),
                default="high",
                tooltip="Extended thinking effort. Reasoning is always enabled for this model.",
                advanced=True,
            )
        )
    elif model_label not in _THINKING_UNSUPPORTED:
        inputs.append(
            IO.Combo.Input(
                "reasoning_effort",
                options=_reasoning_effort_options(model_label),
                default="off",
                tooltip="Extended thinking effort. 'off' disables reasoning.",
                advanced=True,
            )
        )
    return inputs


def _get_text_from_response(response: AnthropicMessagesResponse) -> str:
    if not response.content:
        return ""
    # Thinking blocks are silently dropped — we never want reasoning in the output.
    return "\n".join(
        block.text for block in response.content
        if isinstance(block, AnthropicResponseTextBlock) and block.text
    )


async def _build_image_content_blocks(
    cls: type[IO.ComfyNode],
    image_tensors: list[Input.Image],
) -> list[AnthropicImageContent]:
    urls = await upload_images_to_comfyapi(
        cls,
        image_tensors,
        max_images=CLAUDE_MAX_IMAGES,
        total_pixels=ANTHROPIC_IMAGE_MAX_PIXELS,
        wait_label="Uploading reference images",
    )
    return [AnthropicImageContent(source=AnthropicImageSourceUrl(url=url)) for url in urls]


class ClaudeNode(IO.ComfyNode):
    """Generate text responses from an Anthropic Claude model."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ClaudeNode",
            display_name="Anthropic Claude",
            category="partner/text/Anthropic",
            essentials_category="Text Generation",
            description="Generate text responses with Anthropic's Claude models. "
            "Provide a text prompt and optionally one or more images for multimodal context.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text input to the model.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(label, _claude_model_inputs(label))
                        for label in CLAUDE_MODELS
                    ],
                    tooltip="The Claude model used to generate the response.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, CLAUDE_MAX_IMAGES + 1)],
                        min=0,
                    ),
                    tooltip=f"Optional image(s) to use as context for the model. Up to {CLAUDE_MAX_IMAGES} images.",
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
                expr="""
                (
                  $m := widgets.model;
                  $contains($m, "fable") ? {
                    "type": "list_usd",
                    "usd": [0.0143, 0.0715],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "opus 4.8") ? {
                    "type": "list_usd",
                    "usd": [0.00715, 0.03575],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "sonnet 5") ? {
                    "type": "list_usd",
                    "usd": [0.00286, 0.0143],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "opus 5") ? {
                    "type": "list_usd",
                    "usd": [0.00715, 0.03575],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "opus") ? {
                    "type": "list_usd",
                    "usd": [0.005, 0.025],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "sonnet") ? {
                    "type": "list_usd",
                    "usd": [0.003, 0.015],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "haiku") ? {
                    "type": "list_usd",
                    "usd": [0.001, 0.005],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : {"type":"text", "text":"Token-based"}
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
        images: dict | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_label = model["model"]
        max_tokens = model.get("max_tokens", 32768)
        reasoning_effort = model.get("reasoning_effort", "off")
        always_thinking = model_label in _ALWAYS_THINKING_MODELS
        thinking_enabled = always_thinking or (
            reasoning_effort not in ("off", None) and model_label not in _THINKING_UNSUPPORTED
        )

        # Anthropic requires temperature to be unset (defaults to 1.0) when thinking is enabled.
        # Opus 4.7 also rejects user-supplied temperature.
        if model_label in _NO_TEMPERATURE_MODELS or thinking_enabled or model_label == "Opus 4.7":
            temperature = None
        else:
            temperature = model.get("temperature", 1.0)

        thinking_cfg: AnthropicThinkingConfig | None = None
        output_cfg: AnthropicOutputConfig | None = None
        if always_thinking:
            output_cfg = AnthropicOutputConfig(effort=reasoning_effort)
        elif thinking_enabled:
            if model_label in _ADAPTIVE_THINKING_MODELS:
                # Adaptive mode - Anthropic chooses the budget based on effort hint
                thinking_cfg = AnthropicThinkingConfig(type="adaptive")
                output_cfg = AnthropicOutputConfig(effort=reasoning_effort)
            else:
                # Budget mode (Sonnet 4.5). Leave at least 1024 tokens for the actual response
                budget = _REASONING_BUDGET[reasoning_effort]
                budget = min(budget, max(1024, max_tokens - 1024))
                thinking_cfg = AnthropicThinkingConfig(type="enabled", budget_tokens=budget)
        elif model_label in _EXPLICIT_THINKING_OFF_MODELS:
            thinking_cfg = AnthropicThinkingConfig(type="disabled")

        image_tensors: list[Input.Image] = [t for t in (images or {}).values() if t is not None]
        if sum(get_number_of_images(t) for t in image_tensors) > CLAUDE_MAX_IMAGES:
            raise ValueError(f"Up to {CLAUDE_MAX_IMAGES} images are supported per request.")

        content: list[AnthropicTextContent | AnthropicImageContent] = []
        if image_tensors:
            content.extend(await _build_image_content_blocks(cls, image_tensors))
        content.append(AnthropicTextContent(text=prompt))

        response = await sync_op(
            cls,
            ApiEndpoint(path=ANTHROPIC_MESSAGES_ENDPOINT, method="POST"),
            response_model=AnthropicMessagesResponse,
            data=AnthropicMessagesRequest(
                model=CLAUDE_MODELS[model_label],
                max_tokens=max_tokens,
                messages=[AnthropicMessage(role=AnthropicRole.user, content=content)],
                system=system_prompt or None,
                temperature=temperature,
                thinking=thinking_cfg,
                output_config=output_cfg,
            ),
        )
        if response.stop_reason == "refusal":
            raise ValueError(
                "Claude declined to answer this request for safety reasons. "
                "Rephrase the prompt or try a different model."
            )
        return IO.NodeOutput(_get_text_from_response(response) or "Empty response from Claude model.")


class AnthropicExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [ClaudeNode]


async def comfy_entrypoint() -> AnthropicExtension:
    return AnthropicExtension()
