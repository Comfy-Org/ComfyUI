from typing_extensions import override

from comfy_api.index import io as IO, ComfyExtension
from comfy_api_nodes.apis.hunyuan_llm import (
    TencentHyChatRequest,
    TencentHyChatResponse,
    TencentHyImagePart,
    TencentHyImageUrl,
    TencentHyMessage,
    TencentHyTextPart,
    TencentHyThinking,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    get_number_of_images,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)

CHAT_COMPLETIONS_PATH = "/proxy/tencent/v1/chat/completions"
HY_MODELS = {"Hy4": "hy4"}
HY_MAX_IMAGES = 20


def _hy_model_option(label: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        label,
        [
            IO.String.Input("prompt", multiline=True, default=""),
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=[f"image_{i}" for i in range(1, HY_MAX_IMAGES + 1)],
                    min=0,
                ),
                optional=True,
                tooltip=f"Up to {HY_MAX_IMAGES} images. Each image in a batch counts.",
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
            IO.String.Input(
                "system_prompt",
                multiline=True,
                default="",
                optional=True,
                advanced=True,
                tooltip="Foundational instructions that dictate the model's behavior.",
            ),
            IO.Boolean.Input(
                "reasoning",
                default=False,
                advanced=True,
                tooltip="The model thinks before answering. Answers take longer and use more tokens, "
                "and very hard tasks can time out.",
            ),
            IO.Float.Input(
                "temperature",
                default=1.0,
                min=0.0,
                max=1.4,
                step=0.01,
                advanced=True,
                tooltip="Controls randomness. Results still vary at 0.",
            ),
            IO.Int.Input(
                "max_tokens",
                default=32000,
                min=1,
                max=64000,
                advanced=True,
                tooltip="Maximum number of tokens to generate, including reasoning tokens when reasoning is on. "
                "Very long answers can time out.",
            ),
        ],
    )


def _get_text_from_response(response: TencentHyChatResponse) -> str:
    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise ValueError("Empty response from Hy4.")
    if choice.finish_reason == "content_filter":
        raise ValueError("Tencent blocked this request with its content filter.")
    text = choice.message.content if choice.message else None
    if text:
        return text
    if choice.finish_reason == "length":
        raise ValueError("Hy4 used all max_tokens before answering. Raise max_tokens or turn reasoning off.")
    raise ValueError("Empty response from Hy4.")


class TencentHyNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TencentHyNode",
            display_name="Tencent HY",
            category="partner/text/Tencent",
            essentials_category="Text Generation",
            search_aliases=["HY4"],
            description="Generate text responses with Tencent's Hy4 model. "
            "Provide a text prompt and optionally one or more images for context.",
            inputs=[
                IO.DynamicCombo.Input("model", options=[_hy_model_option(label) for label in HY_MODELS]),
            ],
            outputs=[IO.String.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""
                {
                  "type": "list_usd",
                  "usd": [0.00119262, 0.00357643],
                  "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                }
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_string(model["prompt"], min_length=1)
        images = [image for image in model["images"].values() if image is not None]
        if sum(get_number_of_images(image) for image in images) > HY_MAX_IMAGES:
            raise ValueError(f"Up to {HY_MAX_IMAGES} images are supported per request.")

        messages = []
        if model.get("system_prompt"):
            messages.append(TencentHyMessage(role="system", content=[TencentHyTextPart(text=model["system_prompt"])]))
        content = []
        if images:
            urls = await upload_images_to_comfyapi(
                cls,
                images,
                max_images=HY_MAX_IMAGES,
                mime_type="image/webp",
                wait_label="Uploading images",
            )
            content.extend(TencentHyImagePart(image_url=TencentHyImageUrl(url=url)) for url in urls)
        content.append(TencentHyTextPart(text=model["prompt"]))
        messages.append(TencentHyMessage(role="user", content=content))

        response = await sync_op(
            cls,
            ApiEndpoint(path=CHAT_COMPLETIONS_PATH, method="POST"),
            response_model=TencentHyChatResponse,
            data=TencentHyChatRequest(
                model=HY_MODELS[model["model"]],
                messages=messages,
                thinking=TencentHyThinking(type="enabled" if model["reasoning"] else "disabled"),
                temperature=model["temperature"],
                max_tokens=model["max_tokens"],
                seed=model["seed"],
                stream=False,
            ),
            max_retries=1,
        )
        return IO.NodeOutput(_get_text_from_response(response))


class TencentHyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [TencentHyNode]


async def comfy_entrypoint() -> TencentHyExtension:
    return TencentHyExtension()
