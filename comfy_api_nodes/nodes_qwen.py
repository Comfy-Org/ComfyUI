import math
import re

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.qwen import (
    QwenImageContentItem,
    QwenImageGenerationRequest,
    QwenImageGenerationResponse,
    QwenImageInputField,
    QwenImageMessage,
    QwenImageParametersField,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    sync_op,
    tensor_to_base64_string,
    validate_string,
)

GENERATION_PATH = "/proxy/qwen/api/v1/services/aigc/multimodal-generation/generation"
QWEN_IMAGE_MODELS = ["qwen-image-3.0-pro", "qwen-image-3.0"]
MIN_AREA = 262144  # 512*512
MAX_AREA = 6553600  # 2560*2560
MAX_ASPECT = 8  # the API allows aspect ratios from 1:8 to 8:1
MAX_INPUT_BYTES = 10 * 1024 * 1024  # the API rejects decoded input images over 10MB

_IMAGE_REF_RE = re.compile(r"@image(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)


def _resolve_image_refs(prompt: str, total_images: int) -> str:
    """Rewrite @Image1-style references (shared partner-node syntax, 1-based; an unnumbered
    @image means the first image) into the plain 'Image N' wording the model resolves
    natively. A tag counts only at a word boundary or right after a previous tag, so
    adjacent tags like '@Image1@Image2' all resolve while addresses like user@image1.com
    pass through untouched."""
    parts = []
    pos = 0
    prev_end = -1
    for match in _IMAGE_REF_RE.finditer(prompt):
        start = match.start()
        if start > 0 and start != prev_end and (prompt[start - 1].isalnum() or prompt[start - 1] == "_"):
            continue
        idx = int(match.group("idx") or 1)
        if not 1 <= idx <= total_images:
            raise ValueError(
                f"The prompt references @Image{idx}, but only {total_images} reference images "
                f"are connected (a batched input counts once per image)."
            )
        parts.append(prompt[pos:start])
        parts.append(f"Image {idx}")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


def _validate_size(width: int, height: int) -> None:
    if not MIN_AREA <= width * height <= MAX_AREA:
        raise ValueError(
            f"Image area must be between {MIN_AREA} (512x512) and {MAX_AREA} (2560x2560) pixels; "
            f"got {width}x{height} = {width * height}."
        )
    if width > MAX_ASPECT * height or height > MAX_ASPECT * width:
        raise ValueError(f"Aspect ratio must be between 1:8 and 8:1; got {width}x{height}.")


def _fit_to_size(width: int, height: int) -> tuple[int, int]:
    """Scale dimensions into the supported pixel area and 1:8..8:1 aspect range, preserving
    the aspect ratio where possible."""
    if width > MAX_ASPECT * height:
        height = math.ceil(width / MAX_ASPECT)
    elif height > MAX_ASPECT * width:
        width = math.ceil(height / MAX_ASPECT)
    area = width * height
    if area < MIN_AREA:
        scale = math.sqrt(MIN_AREA / area)
        width, height = math.ceil(width * scale), math.ceil(height * scale)
    elif area > MAX_AREA:
        scale = math.sqrt(MAX_AREA / area)
        width, height = math.floor(width * scale), math.floor(height * scale)
    # rounding can push the ratio a hair past the limit; trimming only ever shrinks the area
    return min(width, MAX_ASPECT * height), min(height, MAX_ASPECT * width)


def _image_data_uri(image: torch.Tensor) -> str:
    """PNG data URI of an RGB view of the image, downscaled to <=2048x2048; falls back to
    JPEG when the PNG exceeds the API's decoded-size cap (e.g. noisy, incompressible images)."""
    image = image[..., :3]
    b64 = tensor_to_base64_string(image, total_pixels=2048 * 2048)
    if len(b64) * 3 > MAX_INPUT_BYTES * 4:
        return "data:image/jpeg;base64," + tensor_to_base64_string(
            image, total_pixels=2048 * 2048, mime_type="image/jpeg"
        )
    return "data:image/png;base64," + b64


async def _download_result_images(response: QwenImageGenerationResponse) -> torch.Tensor:
    if not response.output:
        raise Exception(f"An unknown error occurred: {response.code} - {response.message}")
    urls = [
        item.image
        for choice in response.output.choices
        if choice.message
        for item in choice.message.content
        if item.image
    ]
    if not urls:
        raise Exception(f"The response contains no images: {response.code} - {response.message}")
    return torch.cat([await download_url_to_image_tensor(url) for url in urls])


def _size_inputs() -> list[IO.Int.Input]:
    return [
        IO.Int.Input(
            "width",
            default=1024,
            min=256,
            max=2560,
            step=16,
            tooltip="The total pixel area must be between 512x512 and 2560x2560; "
            "any aspect ratio within that area works.",
        ),
        IO.Int.Input(
            "height",
            default=1024,
            min=256,
            max=2560,
            step=16,
            tooltip="The total pixel area must be between 512x512 and 2560x2560; "
            "any aspect ratio within that area works.",
        ),
    ]


def _t2i_model_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Prompt describing the image. Supports English and Chinese.",
            ),
            IO.String.Input(
                "negative_prompt",
                multiline=True,
                default="",
                tooltip="Negative prompt describing what to avoid.",
            ),
            *_size_inputs(),
        ],
    )


def _edit_model_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=["image_1", "image_2", "image_3"],
                    min=1,
                ),
                tooltip="1-3 reference images. Refer to them in the prompt as @Image1, @Image2, "
                "@Image3, numbered in input order; a batched input counts once per image.",
            ),
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Editing instructions. Supports English and Chinese, "
                "and @Image1-style references to the input images.",
            ),
            IO.String.Input(
                "negative_prompt",
                multiline=True,
                default="",
                tooltip="Negative prompt describing what to avoid.",
            ),
        ],
    )


class QwenImageTextToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QwenImageTextToImageApi",
            display_name="Qwen Image 3 Text to Image",
            category="partner/image/Qwen",
            description="Generates images from a text prompt using the Qwen-Image 3.0 models.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_t2i_model_option(model_id) for model_id in QWEN_IMAGE_MODELS],
                    tooltip="Model to use.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Number of images to generate, returned as a batch.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.width", "model.height", "n"]),
                expr="""
                (
                  $isPro := widgets.model = "qwen-image-3.0-pro";
                  $area := $lookup(widgets, "model.width") * $lookup(widgets, "model.height");
                  $rate := $isPro ? ($area > 2250000 ? 0.10725 : 0.0572) : 0.0429;
                  {"type":"usd","usd": $rate * widgets.n}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        n: int = 1,
        seed: int = 42,
        prompt_extend: bool = True,
        watermark: bool = False,
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        width, height = model["width"], model["height"]
        _validate_size(width, height)
        response = await sync_op(
            cls,
            ApiEndpoint(path=GENERATION_PATH, method="POST"),
            response_model=QwenImageGenerationResponse,
            data=QwenImageGenerationRequest(
                model=model["model"],
                input=QwenImageInputField(
                    messages=[QwenImageMessage(content=[QwenImageContentItem(text=model["prompt"])])],
                ),
                parameters=QwenImageParametersField(
                    size=f"{width}*{height}",
                    n=n,
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    negative_prompt=model["negative_prompt"] or None,
                ),
            ),
        )
        return IO.NodeOutput(await _download_result_images(response))


class QwenImageEditApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="QwenImageEditApi",
            display_name="Qwen Image 3 Edit",
            category="partner/image/Qwen",
            description="Edits or combines up to 3 reference images guided by a text prompt "
            "using the Qwen-Image 3.0 models.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_edit_model_option(model_id) for model_id in QWEN_IMAGE_MODELS],
                    tooltip="Model to use.",
                ),
                IO.DynamicCombo.Input(
                    "size",
                    options=[
                        IO.DynamicCombo.Option("match input", []),
                        IO.DynamicCombo.Option("auto", []),
                        IO.DynamicCombo.Option("custom", _size_inputs()),
                    ],
                    tooltip="Output resolution. 'match input' reuses the first reference image's size, "
                    "'auto' lets the model pick a size with the same aspect ratio, "
                    "'custom' sets an explicit width and height.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=6,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Number of images to generate, returned as a batch.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed to use for generation.",
                ),
                IO.Boolean.Input(
                    "prompt_extend",
                    default=True,
                    tooltip="Whether to enhance the prompt with AI assistance.",
                    advanced=True,
                ),
                IO.Boolean.Input(
                    "watermark",
                    default=False,
                    tooltip="Whether to add an AI-generated watermark to the result.",
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["model", "size", "size.width", "size.height", "n"],
                    input_groups=["model.images"],
                ),
                expr="""
                (
                  $isPro := widgets.model = "qwen-image-3.0-pro";
                  $mode := widgets.size;
                  $count := $max([$lookup(inputGroups, "model.images"), 1]);
                  $inputCost := 0.00429 * $count;
                  $area := $mode = "custom"
                    ? $lookup(widgets, "size.width") * $lookup(widgets, "size.height") : 0;
                  $customRate := $area > 2250000 ? 0.10725 : 0.0572;
                  $isPro and $mode != "custom"
                    ? {"type":"range_usd",
                       "min_usd": 0.0572 * widgets.n + $inputCost,
                       "max_usd": 0.10725 * widgets.n + $inputCost}
                    : {"type":"usd",
                       "usd": ($isPro ? $customRate : 0.0429) * widgets.n + $inputCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        size: dict,
        n: int = 1,
        seed: int = 42,
        prompt_extend: bool = True,
        watermark: bool = False,
    ):
        validate_string(model["prompt"], strip_whitespace=False, min_length=1)
        reference_images = [image for key in model["images"] for image in model["images"][key]]
        if len(reference_images) > 3:
            raise ValueError(
                f"A maximum of 3 reference images is supported; got {len(reference_images)} "
                f"(a batched input counts once per image)."
            )
        prompt = _resolve_image_refs(model["prompt"], len(reference_images))
        if size["size"] == "custom":
            _validate_size(size["width"], size["height"])
            size_str = f"{size['width']}*{size['height']}"
        elif size["size"] == "match input":
            height, width = reference_images[0].shape[0], reference_images[0].shape[1]
            width, height = _fit_to_size(width, height)
            size_str = f"{width}*{height}"
        else:  # auto: the API picks a size preserving the input aspect ratio (1.9-4.2 MP)
            size_str = None
        content = [QwenImageContentItem(image=_image_data_uri(image)) for image in reference_images]
        content.append(QwenImageContentItem(text=prompt))
        response = await sync_op(
            cls,
            ApiEndpoint(path=GENERATION_PATH, method="POST"),
            response_model=QwenImageGenerationResponse,
            data=QwenImageGenerationRequest(
                model=model["model"],
                input=QwenImageInputField(messages=[QwenImageMessage(content=content)]),
                parameters=QwenImageParametersField(
                    size=size_str,
                    n=n,
                    seed=seed,
                    prompt_extend=prompt_extend,
                    watermark=watermark,
                    negative_prompt=model["negative_prompt"] or None,
                ),
            ),
        )
        return IO.NodeOutput(await _download_result_images(response))


class QwenApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            QwenImageTextToImageApi,
            QwenImageEditApi,
        ]


async def comfy_entrypoint() -> QwenApiExtension:
    return QwenApiExtension()
