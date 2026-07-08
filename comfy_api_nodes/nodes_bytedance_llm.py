"""API Nodes for ByteDance Seed LLM via the BytePlus ModelArk Responses API.

See: https://docs.byteplus.com/en/docs/ModelArk/1585128
"""

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bytedance_llm import (
    BytePlusInputImage,
    BytePlusInputMessage,
    BytePlusInputText,
    BytePlusInputVideo,
    BytePlusMessageContent,
    BytePlusResponseCreateRequest,
    BytePlusResponseObject,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    get_number_of_images,
    sync_op,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
)

BYTEPLUS_RESPONSES_ENDPOINT = "/proxy/byteplus/api/v3/responses"
SEED_MAX_IMAGES = 20
SEED_MAX_VIDEOS = 4

SEED_MODELS: dict[str, str] = {
    "Seed 2.0 Pro": "seed-2-0-pro-260328",
    "Seed 2.0 Lite": "seed-2-0-lite-260228",
    "Seed 2.0 Mini": "seed-2-0-mini-260215",
}

# USD per 1M tokens: (input, cache_hit_input, output)
_SEED_PRICES_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "seed-2-0-pro-260328": (0.50, 0.10, 3.00),
    "seed-2-0-lite-260228": (0.25, 0.05, 2.00),
    "seed-2-0-mini-260215": (0.10, 0.02, 0.40),
}


def _seed_model_inputs(max_images: int = SEED_MAX_IMAGES, max_videos: int = SEED_MAX_VIDEOS):
    return [
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_images + 1)],
                min=0,
            ),
            tooltip=f"Optional image(s) to use as context for the model. Up to {max_images} images.",
        ),
        IO.Autogrow.Input(
            "videos",
            template=IO.Autogrow.TemplateNames(
                IO.Video.Input("video"),
                names=[f"video_{i}" for i in range(1, max_videos + 1)],
                min=0,
            ),
            tooltip=f"Optional video(s) to use as context for the model. Up to {max_videos} videos.",
        ),
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.01,
            tooltip="Controls randomness. 0.0 is deterministic, higher values are more random.",
            advanced=True,
        ),
    ]


def _calculate_price(model_id: str, response: BytePlusResponseObject) -> float | None:
    """Compute approximate USD price from response usage."""
    if not response.usage:
        return None
    rates = _SEED_PRICES_PER_MILLION.get(model_id)
    if rates is None:
        return None
    input_rate, cache_hit_rate, output_rate = rates
    input_tokens = response.usage.input_tokens or 0
    output_tokens = response.usage.output_tokens or 0
    cached = 0
    if response.usage.input_tokens_details:
        cached = response.usage.input_tokens_details.cached_tokens or 0
    fresh_input = max(0, input_tokens - cached)
    total = fresh_input * input_rate + cached * cache_hit_rate + output_tokens * output_rate
    return total / 1_000_000.0


def _get_text_from_response(response: BytePlusResponseObject) -> str:
    """Extract concatenated text from all assistant message output_text blocks."""
    if not response.output:
        return ""
    chunks: list[str] = []
    for item in response.output:
        if item.type != "message" or not item.content:
            continue
        for block in item.content:
            if block.type == "output_text" and block.text:
                chunks.append(block.text)
            elif block.type == "refusal" and block.refusal:
                raise ValueError(f"Model refused to respond: {block.refusal}")
    return "\n".join(chunks)


async def _build_image_content_blocks(
    cls: type[IO.ComfyNode],
    image_tensors: list[Input.Image],
) -> list[BytePlusInputImage]:
    urls = await upload_images_to_comfyapi(
        cls,
        image_tensors,
        max_images=SEED_MAX_IMAGES,
        wait_label="Uploading reference images",
    )
    return [BytePlusInputImage(image_url=url) for url in urls]


async def _build_video_content_blocks(
    cls: type[IO.ComfyNode],
    videos: list[Input.Video],
) -> list[BytePlusInputVideo]:
    blocks: list[BytePlusInputVideo] = []
    total = len(videos)
    for idx, video in enumerate(videos):
        label = "Uploading reference video"
        if total > 1:
            label = f"{label} ({idx + 1}/{total})"
        url = await upload_video_to_comfyapi(cls, video, wait_label=label)
        blocks.append(BytePlusInputVideo(video_url=url))
    return blocks


class ByteDanceSeedNode(IO.ComfyNode):
    """Generate text responses from a ByteDance Seed 2.0 model."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ByteDanceSeedNode",
            display_name="ByteDance Seed",
            category="partner/text/ByteDance",
            essentials_category="Text Generation",
            description="Generate text responses with ByteDance's Seed 2.0 models. "
            "Provide a text prompt and optionally one or more images or videos for multimodal context.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text input to the model.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[IO.DynamicCombo.Option(label, _seed_model_inputs()) for label in SEED_MODELS],
                    tooltip="The Seed model used to generate the response.",
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
                  $contains($m, "mini") ? {
                    "type": "list_usd",
                    "usd": [0.00025, 0.0009],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "lite") ? {
                    "type": "list_usd",
                    "usd": [0.0003, 0.002],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "pro") ? {
                    "type": "list_usd",
                    "usd": [0.0005, 0.003],
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
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)
        model_label = model["model"]
        temperature = model["temperature"]
        model_id = SEED_MODELS[model_label]

        image_tensors: list[Input.Image] = [t for t in (model.get("images") or {}).values() if t is not None]
        if sum(get_number_of_images(t) for t in image_tensors) > SEED_MAX_IMAGES:
            raise ValueError(f"Up to {SEED_MAX_IMAGES} images are supported per request.")

        video_inputs: list[Input.Video] = [v for v in (model.get("videos") or {}).values() if v is not None]
        if len(video_inputs) > SEED_MAX_VIDEOS:
            raise ValueError(f"Up to {SEED_MAX_VIDEOS} videos are supported per request.")

        content: list[BytePlusMessageContent] = []
        if image_tensors:
            content.extend(await _build_image_content_blocks(cls, image_tensors))
        if video_inputs:
            content.extend(await _build_video_content_blocks(cls, video_inputs))
        content.append(BytePlusInputText(text=prompt))

        response = await sync_op(
            cls,
            ApiEndpoint(path=BYTEPLUS_RESPONSES_ENDPOINT, method="POST"),
            response_model=BytePlusResponseObject,
            data=BytePlusResponseCreateRequest(
                model=model_id,
                input=[BytePlusInputMessage(role="user", content=content)],
                instructions=system_prompt or None,
                temperature=temperature,
                store=False,
                stream=False,
            ),
            price_extractor=lambda r: _calculate_price(model_id, r),
        )
        if response.error:
            raise ValueError(f"Seed API error ({response.error.code}): {response.error.message}")
        result = _get_text_from_response(response)
        if not result:
            raise ValueError("Empty response from Seed model.")
        return IO.NodeOutput(result)


class ByteDanceLLMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [ByteDanceSeedNode]


async def comfy_entrypoint() -> ByteDanceLLMExtension:
    return ByteDanceLLMExtension()
