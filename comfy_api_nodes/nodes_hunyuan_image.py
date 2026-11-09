import math
import re

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.hunyuan_image import (
    HunyuanImageContentItem,
    HunyuanImageMessage,
    HunyuanImageRequest,
    HunyuanImageResponse,
    HunyuanImageUrl,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_image_tensor,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)

GENERATION_PATH = "/proxy/tencent/v1/wand/hunyuan-image/v35-generation"
HUNYUAN_IMAGE_MODELS = {"hy-image-3.5-preview": "hy-image-v3.5-preview"}
MAX_INPUT_IMAGES = 5
ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]
RESOLUTIONS = {"1K": 1048576, "2K": 4194304, "4K": 16777216}
AUTO_MAX_AREA = 4194304
CUSTOM_MAX_AREA = 16777216
AUTO_MAX_ASPECT = 4
REFERENCE_DETAIL = {"standard": 1048576, "high": 4194304}
INPUT_DOWNLOAD_RETRIES = 2

_IMAGE_REF_RE = re.compile(r"@image(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)
_BUSINESS_ERROR_RE = re.compile(r"msg:\s*(?P<msg>.+)$", re.DOTALL)


def _resolve_image_refs(prompt: str, total_images: int) -> str:
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


def _fit_size(ratio: float, area: int) -> tuple[int, int]:
    width = math.floor(math.sqrt(area * ratio) / 16) * 16
    height = round(width / ratio / 16) * 16
    if width * height > area:
        height -= 16
    return width, height


def _size_inputs(auto_tooltip: str) -> list[IO.Input]:
    custom_tooltip = (
        "Used only when resolution is 'custom'. Must be a multiple of 16; the total pixel area can be up to "
        "4096x4096. Any aspect ratio works; beyond about 6:1 the model starts repeating the subject."
    )
    return [
        IO.Combo.Input(
            "aspect_ratio",
            options=["auto", *ASPECT_RATIOS],
            tooltip=f"Aspect ratio of the output. 'auto': {auto_tooltip} Ignored when resolution is 'custom'.",
        ),
        IO.Combo.Input(
            "resolution",
            options=[*RESOLUTIONS, "custom"],
            default="2K",
            tooltip="Pixel area of the output: 1K is about 1024x1024, 2K is about 2048x2048, 4K is about 4096x4096. "
            "Anything above 2K is rendered at 2K and upscaled by the model. "
            "'custom': use the width and height below instead of an aspect ratio and a preset area.",
        ),
        IO.Int.Input("width", default=2048, min=256, max=8192, step=16, tooltip=custom_tooltip),
        IO.Int.Input("height", default=2048, min=256, max=8192, step=16, tooltip=custom_tooltip),
    ]


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        step=1,
        display_mode=IO.NumberDisplay.number,
        control_after_generate=True,
        tooltip="Seed to use for generation; results still vary between runs with the same seed.",
    )


def _watermark_input() -> IO.Boolean.Input:
    return IO.Boolean.Input(
        "watermark",
        default=False,
        advanced=True,
        tooltip="Whether to add an AI-generated watermark to the result.",
    )


def _t2i_model_option(model_name: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_name,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Prompt describing the image. The model rewrites and expands it before rendering.",
            ),
            *_size_inputs("the model picks the aspect ratio from the prompt; not available at 4K."),
            _seed_input(),
            _watermark_input(),
        ],
    )


def _edit_model_option(model_name: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_name,
        [
            IO.Autogrow.Input(
                "images",
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=[f"image_{i}" for i in range(1, MAX_INPUT_IMAGES + 1)],
                    min=1,
                ),
                tooltip=f"1-{MAX_INPUT_IMAGES} reference images to edit or combine. Refer to them in the prompt "
                "as @Image1, @Image2, ..., numbered in input order; a batched input counts once per image.",
            ),
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Editing instructions. Supports @Image1-style references to the input images.",
            ),
            *_size_inputs("the output follows the aspect ratio of the first reference image."),
            _seed_input(),
            IO.Combo.Input(
                "reference_detail",
                options=list(REFERENCE_DETAIL),
                advanced=True,
                tooltip="How much detail of the reference images the model sees: 'standard' is up to 1024x1024 "
                "pixels per image, 'high' is up to 2048x2048 and preserves small text and fine detail "
                "better, but takes longer.",
            ),
            _watermark_input(),
        ],
    )


def _resolve_size(model: dict, reference_image: torch.Tensor | None = None) -> tuple[str | None, int | None]:
    aspect_ratio = model["aspect_ratio"]
    if model["resolution"] == "custom":
        width, height = model["width"], model["height"]
        if width % 16 or height % 16:
            raise ValueError(f"Width and height must be multiples of 16; got {width}x{height}.")
        if width * height > CUSTOM_MAX_AREA:
            raise ValueError(f"The total pixel area must not exceed 4096x4096; got {width}x{height}.")
        return f"{width}x{height}", None
    area = RESOLUTIONS[model["resolution"]]
    if aspect_ratio != "auto":
        ratio_width, ratio_height = aspect_ratio.split(":")
        width, height = _fit_size(int(ratio_width) / int(ratio_height), area)
    elif reference_image is not None:
        ratio = reference_image.shape[1] / reference_image.shape[0]
        width, height = _fit_size(min(max(ratio, 1 / AUTO_MAX_ASPECT), AUTO_MAX_ASPECT), area)
    elif area > AUTO_MAX_AREA:
        raise ValueError(
            "The 'auto' aspect ratio is not available at 4K; choose an aspect ratio or the 'custom' resolution."
        )
    else:
        return None, area
    return f"{width}x{height}", None


def _error_message(response: HunyuanImageResponse) -> str:
    message = (response.error.message if response.error else None) or "The response contains no image."
    match = _BUSINESS_ERROR_RE.search(message)
    return match.group("msg").strip() if match else message


async def _generate(
    cls: type[IO.ComfyNode],
    model: dict,
    content: list[HunyuanImageContentItem],
    size: str | None,
    generate_max_pixels: int | None,
    resize_max_pixels: int | None = None,
) -> torch.Tensor:
    request = HunyuanImageRequest(
        model=HUNYUAN_IMAGE_MODELS[model["model"]],
        messages=[HunyuanImageMessage(content=content)],
        size=size,
        generate_max_pixels=generate_max_pixels,
        resize_max_pixels=resize_max_pixels,
        seed=model["seed"],
        logo_add=int(model["watermark"]),
    )
    for attempt in range(INPUT_DOWNLOAD_RETRIES + 1):
        try:
            response = await sync_op(
                cls,
                ApiEndpoint(path=GENERATION_PATH, method="POST"),
                response_model=HunyuanImageResponse,
                data=request,
            )
            break
        except Exception as e:
            if attempt == INPUT_DOWNLOAD_RETRIES or "download image failed" not in str(e):
                raise
    image = response.choices[0].delta.image if response.choices and response.choices[0].delta else None
    if response.error or image is None or not image.url:
        raise Exception(f"Hunyuan Image generation failed: {_error_message(response)}")
    return await download_url_to_image_tensor(image.url, cls=cls)


def _price_badge() -> IO.PriceBadge:
    return IO.PriceBadge(
        depends_on=IO.PriceBadgeDepends(widgets=["model.resolution", "model.width", "model.height"]),
        expr="""
        (
          $resolution := $lookup(widgets, "model.resolution");
          $upscaled := $resolution = "custom"
            ? $lookup(widgets, "model.width") * $lookup(widgets, "model.height") > 4194304
            : $resolution = "4k";
          {"type":"usd","usd": $upscaled ? 0.04576 : 0.03432}
        )
        """,
    )


class HunyuanImageTextToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImageTextToImageApi",
            display_name="Tencent HY Image: Text to Image",
            category="partner/image/Tencent",
            description="Generates images from a text prompt using Tencent's Hunyuan Image model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_t2i_model_option(model_name) for model_name in HUNYUAN_IMAGE_MODELS],
                    tooltip="Model to use.",
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
            price_badge=_price_badge(),
        )

    @classmethod
    async def execute(cls, model: dict):
        validate_string(model["prompt"], min_length=1)
        size, generate_max_pixels = _resolve_size(model)
        content = [HunyuanImageContentItem(type="text", text=model["prompt"])]
        return IO.NodeOutput(await _generate(cls, model, content, size, generate_max_pixels))


class HunyuanImageEditApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HunyuanImageEditApi",
            display_name="Tencent HY Image: Edit",
            category="partner/image/Tencent",
            description=f"Edits or combines up to {MAX_INPUT_IMAGES} reference images guided by a text prompt "
            "using Tencent's Hunyuan Image model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_edit_model_option(model_name) for model_name in HUNYUAN_IMAGE_MODELS],
                    tooltip="Model to use.",
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
            price_badge=_price_badge(),
        )

    @classmethod
    async def execute(cls, model: dict):
        validate_string(model["prompt"], min_length=1)
        reference_images = [image for key in model["images"] for image in model["images"][key]]
        if len(reference_images) > MAX_INPUT_IMAGES:
            raise ValueError(
                f"A maximum of {MAX_INPUT_IMAGES} reference images is supported; got {len(reference_images)} "
                f"(a batched input counts once per image)."
            )
        size, generate_max_pixels = _resolve_size(model, reference_images[0])
        max_pixels = REFERENCE_DETAIL[model["reference_detail"]]
        content = [
            HunyuanImageContentItem(type="text", text=_resolve_image_refs(model["prompt"], len(reference_images)))
        ]
        urls = await upload_images_to_comfyapi(
            cls,
            [image[..., :3] for image in reference_images],
            max_images=MAX_INPUT_IMAGES,
            mime_type="image/png",
            wait_label="Uploading reference images",
            total_pixels=max_pixels,
        )
        content.extend(HunyuanImageContentItem(type="image_url", image_url=HunyuanImageUrl(url=url)) for url in urls)
        return IO.NodeOutput(
            await _generate(cls, model, content, size, generate_max_pixels, resize_max_pixels=max_pixels)
        )


class HunyuanImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            HunyuanImageTextToImageApi,
            HunyuanImageEditApi,
        ]


async def comfy_entrypoint() -> HunyuanImageExtension:
    return HunyuanImageExtension()
