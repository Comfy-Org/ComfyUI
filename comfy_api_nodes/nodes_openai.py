import base64
import os
from enum import Enum
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageSequence
from typing_extensions import override

import folder_paths
from comfy.utils import common_upscale
from comfy_api.v3 import IO, ComfyExtension, Input
from comfy_extras.nodes_images import SVG
from comfy_api_nodes.apis.openai import (
    InputFileContent,
    InputImageContent,
    InputMessage,
    InputTextContent,
    ModelResponseProperties,
    OpenAICreateResponse,
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
    OpenAIImageSVGOptions,
    OpenAIResponse,
    OutputContent,
    Reasoning,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    downscale_image_tensor,
    get_number_of_images,
    poll_op,
    sync_op,
    tensor_to_base64_string,
    text_filepath_to_data_uri,
    validate_string,
)

RESPONSES_ENDPOINT = "/proxy/openai/v1/responses"
STARTING_POINT_ID_PATTERN = r"<starting_point_id:(.*)>"


class SupportedOpenAIModel(str, Enum):
    gpt_6_astra = "gpt-6-astra"
    gpt_6_1_sol = "gpt-6.1-sol"
    gpt_6_1_terra = "gpt-6.1-terra"
    gpt_6_sol = "gpt-6-sol"
    gpt_6_luna = "gpt-6-luna"
    gpt_5_6_sol = "gpt-5.6-sol"
    gpt_5_6_terra = "gpt-5.6-terra"
    gpt_5_6_luna = "gpt-5.6-luna"
    gpt_5_5_pro = "gpt-5.5-pro"
    gpt_5_5 = "gpt-5.5"
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    o3 = "o3"


REASONING_EFFORT_OPTIONS = ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]
_O_SERIES_EFFORTS = ("low", "medium", "high")
_GPT_5_EFFORTS = ("minimal", "low", "medium", "high")
_GPT_5_6_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
SUPPORTED_REASONING_EFFORTS: dict[str, tuple[str, ...]] = {
    SupportedOpenAIModel.gpt_6_astra: ("low", "medium", "high", "xhigh", "max"),
    SupportedOpenAIModel.gpt_6_1_sol: ("low", "medium", "high", "xhigh", "max"),
    SupportedOpenAIModel.gpt_6_1_terra: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_6_sol: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_6_luna: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_5_6_sol: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_5_6_terra: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_5_6_luna: _GPT_5_6_EFFORTS,
    SupportedOpenAIModel.gpt_5_5_pro: ("medium", "high", "xhigh"),
    SupportedOpenAIModel.gpt_5_5: ("none", "low", "medium", "high", "xhigh"),
    SupportedOpenAIModel.gpt_5: _GPT_5_EFFORTS,
    SupportedOpenAIModel.gpt_5_mini: _GPT_5_EFFORTS,
    SupportedOpenAIModel.gpt_5_nano: _GPT_5_EFFORTS,
    SupportedOpenAIModel.gpt_4_1: (),
    SupportedOpenAIModel.gpt_4_1_mini: (),
    SupportedOpenAIModel.o3: _O_SERIES_EFFORTS,
}


async def _image_payload_bytes(img_data, timeout: int | None, cls: type[IO.ComfyNode] | None) -> BytesIO:
    if img_data.b64_json:
        return BytesIO(base64.b64decode(img_data.b64_json))
    if img_data.url:
        img_io = BytesIO()
        await download_url_to_bytesio(img_data.url, img_io, timeout=timeout, cls=cls)
        return img_io
    raise ValueError("Invalid image payload – neither URL nor base64 data present.")


async def _single_image_payload(response, cls: type[IO.ComfyNode]) -> BytesIO:
    if not response.data:
        raise ValueError("No images returned from API endpoint")
    return await _image_payload_bytes(response.data[0], None, cls)


async def validate_and_cast_response(
    response, timeout: int = None, cls: type[IO.ComfyNode] = None
) -> torch.Tensor:
    """Validates and casts a response to a torch.Tensor.

    Args:
        response: The response to validate and cast.
        timeout: Request timeout in seconds. Defaults to None (no timeout).
        cls: The calling node class; required for relative `/proxy/` URLs so they can be expanded and authenticated.

    Returns:
        A torch.Tensor of shape (N, H, W, C) with all returned images; images whose
        dimensions differ from the first image's are resized to match it.

    Raises:
        ValueError: If the response is not valid.
    """
    # validate raw JSON response
    data = response.data
    if not data or len(data) == 0:
        raise ValueError("No images returned from API endpoint")

    # Initialize list to store image tensors
    image_tensors: list[torch.Tensor] = []

    # Process each image in the data array
    for img_data in data:
        pil_img = Image.open(await _image_payload_bytes(img_data, timeout, cls)).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        image_tensors.append(torch.from_numpy(arr))

    # With size="auto" the API can return images whose dimensions differ by a few pixels within a single response
    # resize them to the first image's dimensions so they can be stacked into one batch.
    ref_h, ref_w = image_tensors[0].shape[:2]
    for i, t in enumerate(image_tensors):
        if t.shape[:2] != (ref_h, ref_w):
            samples = t.unsqueeze(0).movedim(-1, 1)
            samples = common_upscale(samples, ref_w, ref_h, "bilinear", "center")
            image_tensors[i] = samples.movedim(1, -1).squeeze(0)
    return torch.stack(image_tensors, dim=0)


class OpenAIGPTImage1(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImage1",
            display_name="OpenAI GPT Image 2",
            category="partner/image/OpenAI",
            description="Generates images synchronously via OpenAI's GPT Image endpoint.",
            is_deprecated=True,
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for GPT Image",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
                    optional=True,
                ),
                IO.Combo.Input(
                    "quality",
                    default="low",
                    options=["low", "medium", "high"],
                    tooltip="Image quality, affects cost and generation time.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "background",
                    default="auto",
                    options=["auto", "opaque", "transparent"],
                    tooltip="Return image with or without background",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    default="auto",
                    options=[
                        "auto",
                        "1024x1024",
                        "1024x1536",
                        "1536x1024",
                        "2048x2048",
                        "2048x1152",
                        "1152x2048",
                        "3840x2160",
                        "2160x3840",
                        "Custom",
                    ],
                    tooltip="Image size. Select 'Custom' to use the custom width and height (GPT Image 2 only).",
                    optional=True,
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate",
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image for image editing.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="Optional mask for inpainting (white areas will be replaced)",
                    optional=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=["gpt-image-1.5", "gpt-image-2"],
                    default="gpt-image-2",
                    optional=True,
                ),
                IO.Int.Input(
                    "custom_width",
                    default=1024,
                    min=1024,
                    max=3840,
                    step=16,
                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16 (GPT Image 2 only).",
                    optional=True,
                ),
                IO.Int.Input(
                    "custom_height",
                    default=1024,
                    min=1024,
                    max=3840,
                    step=16,
                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16 (GPT Image 2 only).",
                    optional=True,
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
                depends_on=IO.PriceBadgeDepends(widgets=["quality", "n", "model"]),
                expr="""
                (
                  $ranges := {
                    "gpt-image-1.5": {
                      "low":    [0.009, 0.02],
                      "medium": [0.034, 0.062],
                      "high":   [0.133, 0.22]
                    },
                    "gpt-image-2": {
                      "low":    [0.0058, 0.0228],
                      "medium": [0.0492, 0.2016],
                      "high":   [0.198, 0.804]
                    }
                  };
                  $range := $lookup($lookup($ranges, widgets.model), widgets.quality);
                  $nRaw := widgets.n;
                  $n := ($nRaw != null and $nRaw != 0) ? $nRaw : 1;
                  ($n = 1)
                    ? {"type":"range_usd","min_usd": $range[0], "max_usd": $range[1], "format": {"approximate": true}}
                    : {
                        "type":"range_usd",
                        "min_usd": $range[0] * $n,
                        "max_usd": $range[1] * $n,
                        "format": { "suffix": "/Run", "approximate": true }
                      }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int = 0,
        quality: str = "low",
        background: str = "opaque",
        image: Input.Image | None = None,
        mask: Input.Image | None = None,
        n: int = 1,
        size: str = "1024x1024",
        custom_width: int = 1024,
        custom_height: int = 1024,
        model: str = "gpt-image-2",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)

        if mask is not None and image is None:
            raise ValueError("Cannot use a mask without an input image")

        if size == "Custom":
            if model != "gpt-image-2":
                raise ValueError("Custom resolution is only supported by GPT Image 2 model")
            if custom_width % 16 != 0 or custom_height % 16 != 0:
                raise ValueError(f"Custom width and height must be multiples of 16, got {custom_width}x{custom_height}")
            if max(custom_width, custom_height) > 3840:
                raise ValueError(f"Custom resolution max edge must be <= 3840, got {custom_width}x{custom_height}")
            ratio = max(custom_width, custom_height) / min(custom_width, custom_height)
            if ratio > 3:
                raise ValueError(
                    f"Custom resolution aspect ratio must not exceed 3:1, got {custom_width}x{custom_height}"
                )
            total_pixels = custom_width * custom_height
            if not 655_360 <= total_pixels <= 8_294_400:
                raise ValueError(
                    f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total_pixels}"
                )
            size = f"{custom_width}x{custom_height}"
        elif model == "gpt-image-1.5":
            if size not in ("auto", "1024x1024", "1024x1536", "1536x1024"):
                raise ValueError(f"Resolution {size} is only supported by GPT Image 2 model")

        if model not in ("gpt-image-1.5", "gpt-image-2"):
            raise ValueError(f"Unknown model: {model}")

        if image is not None:
            files = []
            batch_size = image.shape[0]
            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                if batch_size == 1:
                    files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
                else:
                    files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))

            if mask is not None:
                if image.shape[0] != 1:
                    raise Exception("Cannot use a mask with multiple image")
                if mask.shape[1:] != image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                _, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

                scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0), total_pixels=2048 * 2048).squeeze()

                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_img_byte_arr = BytesIO()
                mask_img.save(mask_img_byte_arr, format="PNG")
                mask_img_byte_arr.seek(0)
                files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/edits", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageEditRequest(
                    model=model,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    seed=seed,
                    size=size,
                    moderation="low",
                ),
                content_type="multipart/form-data",
                files=files,
            )
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageGenerationRequest(
                    model=model,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    seed=seed,
                    size=size,
                    moderation="low",
                ),
            )
        return IO.NodeOutput(await validate_and_cast_response(response, cls=cls))


GPT_IMAGE_QUALITIES = ("low", "medium", "high")
GPT_IMAGE_25_QUALITIES = ("low", "medium", "high", "xhigh", "max")
GPT_IMAGE_26_MODELS = ("gpt-image-2.6-flare", "gpt-image-2.6-sunburst")
GPT_IMAGE_MODELS = (
    *GPT_IMAGE_26_MODELS,
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2",
    "gpt-image-1.5",
)


def _gpt_image_shared_inputs(qualities: tuple[str, ...] = GPT_IMAGE_QUALITIES):
    """Inputs shared by all GPT Image models (quality + reference images + mask)."""
    return [
        IO.Combo.Input(
            "quality",
            default="low",
            options=list(qualities),
            tooltip="Image quality, affects cost and generation time.",
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 17)],
                min=0,
            ),
            tooltip="Optional reference image(s) for image editing. Up to 16 images.",
        ),
        IO.Mask.Input(
            "mask",
            optional=True,
            tooltip="Optional mask for inpainting (white areas will be replaced). "
            "Requires exactly one reference image.",
        ),
    ]


def _gpt_image_legacy_model_inputs():
    """Per-model widget set for legacy gpt-image-1.5 (4 base sizes, transparent bg allowed)."""
    return [
        IO.Combo.Input(
            "size",
            default="auto",
            options=["auto", "1024x1024", "1024x1536", "1536x1024"],
            tooltip="Image size.",
        ),
        IO.Combo.Input(
            "background",
            default="auto",
            options=["auto", "opaque", "transparent"],
            tooltip="Return image with or without background.",
        ),
        *_gpt_image_shared_inputs(),
    ]


def _gpt_image_size_inputs():
    return [
        IO.Combo.Input(
            "size",
            default="auto",
            options=[
                "auto",
                "1024x1024",
                "1024x1536",
                "1536x1024",
                "2048x2048",
                "2048x1152",
                "1152x2048",
                "3840x2160",
                "2160x3840",
                "Custom",
            ],
            tooltip="Image size. Select 'Custom' to use the custom width and height.",
        ),
        IO.Int.Input(
            "custom_width",
            default=1024,
            min=480,
            max=3840,
            step=16,
            tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
        ),
        IO.Int.Input(
            "custom_height",
            default=1024,
            min=480,
            max=3840,
            step=16,
            tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
        ),
    ]


def _gpt_image_2_model_inputs(backgrounds: tuple[str, ...], qualities: tuple[str, ...]):
    return [
        *_gpt_image_size_inputs(),
        IO.Combo.Input(
            "background",
            default="auto",
            options=list(backgrounds),
            tooltip="Return image with or without background.",
        ),
        *_gpt_image_shared_inputs(qualities),
    ]


def _gpt_image_26_model_inputs():
    return [
        *_gpt_image_2_model_inputs(("auto", "opaque", "transparent"), GPT_IMAGE_25_QUALITIES),
        IO.Combo.Input(
            "complexity",
            default="high",
            options=["low", "medium", "high"],
            tooltip="How much the model plans the image before rendering.",
        ),
    ]


def _gpt_image_resolve_size(size: str, custom_width: int, custom_height: int) -> str:
    if size != "Custom":
        return size
    if custom_width % 16 != 0 or custom_height % 16 != 0:
        raise ValueError(
            f"Custom width and height must be multiples of 16, got {custom_width}x{custom_height}"
        )
    if max(custom_width, custom_height) > 3840:
        raise ValueError(
            f"Custom resolution max edge must be <= 3840, got {custom_width}x{custom_height}"
        )
    ratio = max(custom_width, custom_height) / min(custom_width, custom_height)
    if ratio > 3:
        raise ValueError(
            f"Custom resolution aspect ratio must not exceed 3:1, got {custom_width}x{custom_height}"
        )
    total_pixels = custom_width * custom_height
    if not 655_360 <= total_pixels <= 8_294_400:
        raise ValueError(
            f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total_pixels}"
        )
    return f"{custom_width}x{custom_height}"


def _gpt_image_flatten_references(image_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for tensor in image_tensors:
        if len(tensor.shape) == 4:
            flat.extend(tensor[i : i + 1] for i in range(tensor.shape[0]))
        else:
            flat.append(tensor.unsqueeze(0))
    return flat


def _gpt_image_reference_files(flat: list[torch.Tensor]) -> list:
    files = []
    for i, single_image in enumerate(flat):
        scaled_image = downscale_image_tensor(single_image, total_pixels=2048 * 2048).squeeze()
        image_np = (scaled_image.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image_np)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        if len(flat) == 1:
            files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
        else:
            files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))
    return files


class OpenAIGPTImageNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageNodeV2",
            display_name="OpenAI GPT Image",
            category="partner/image/OpenAI",
            description="Generates images via OpenAI's GPT Image endpoint.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for GPT Image",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("gpt-image-2.6-flare", _gpt_image_26_model_inputs()),
                        IO.DynamicCombo.Option("gpt-image-2.6-sunburst", _gpt_image_26_model_inputs()),
                        IO.DynamicCombo.Option(
                            "gpt-image-2.5-flare",
                            _gpt_image_2_model_inputs(("auto", "opaque", "transparent"), GPT_IMAGE_25_QUALITIES),
                        ),
                        IO.DynamicCombo.Option(
                            "gpt-image-2.5-sunburst",
                            _gpt_image_2_model_inputs(("auto", "opaque", "transparent"), GPT_IMAGE_25_QUALITIES),
                        ),
                        IO.DynamicCombo.Option(
                            "gpt-image-2",
                            _gpt_image_2_model_inputs(("auto", "opaque"), GPT_IMAGE_QUALITIES),
                        ),
                        IO.DynamicCombo.Option("gpt-image-1.5", _gpt_image_legacy_model_inputs()),
                    ],
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate",
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="not implemented yet in backend",
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.quality", "model.size", "n"], input_groups=["model.images"]),
                expr="""
                (
                  $ranges := {
                    "gpt-image-1.5": {
                      "low":    [0.009, 0.02],
                      "medium": [0.034, 0.062],
                      "high":   [0.133, 0.22]
                    },
                    "gpt-image-2": {
                      "low":    [0.0019, 0.0237],
                      "medium": [0.0186, 0.2135],
                      "high":   [0.0744, 0.8539]
                    },
                    "gpt-image-2.5-flare": {
                      "low":    [0.0023, 0.0283],
                      "medium": [0.0056, 0.0636],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    },
                    "gpt-image-2.5-sunburst": {
                      "low":    [0.0023, 0.0283],
                      "medium": [0.0056, 0.0636],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    },
                    "gpt-image-2.6-flare": {
                      "low":    [0.0097, 0.1188],
                      "medium": [0.0181, 0.2100],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    },
                    "gpt-image-2.6-sunburst": {
                      "low":    [0.0097, 0.1188],
                      "medium": [0.0181, 0.2100],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    }
                  };
                  $presets := {
                    "gpt-image-2": {
                      "low": {"1024x1024": 0.0071, "1024x1536": 0.0057, "1536x1024": 0.0057, "2048x2048": 0.0143, "2048x1152": 0.0057, "1152x2048": 0.0057, "3840x2160": 0.0134, "2160x3840": 0.0134},
                      "medium": {"1024x1024": 0.0632, "1024x1536": 0.0494, "1536x1024": 0.0494, "2048x2048": 0.1284, "2048x1152": 0.0509, "1152x2048": 0.0509, "3840x2160": 0.1201, "2160x3840": 0.1201},
                      "high": {"1024x1024": 0.2529, "1024x1536": 0.1976, "1536x1024": 0.1976, "2048x2048": 0.5138, "2048x1152": 0.2034, "1152x2048": 0.2034, "3840x2160": 0.4803, "2160x3840": 0.4803}
                    },
                    "gpt-image-2.5": {
                      "low": {"1024x1024": 0.0084, "1024x1536": 0.0068, "1536x1024": 0.0068, "2048x2048": 0.0170, "2048x1152": 0.0067, "1152x2048": 0.0067, "3840x2160": 0.0159, "2160x3840": 0.0159},
                      "medium": {"1024x1024": 0.0188, "1024x1536": 0.0147, "1536x1024": 0.0147, "2048x2048": 0.0383, "2048x1152": 0.0157, "1152x2048": 0.0157, "3840x2160": 0.0371, "2160x3840": 0.0371},
                      "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589, "2048x2048": 0.1531, "2048x1152": 0.0606, "1152x2048": 0.0606, "3840x2160": 0.1431, "2160x3840": 0.1431},
                      "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055, "2048x2048": 0.2721, "2048x1152": 0.1077, "1152x2048": 0.1077, "3840x2160": 0.2544, "2160x3840": 0.2544},
                      "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354, "2048x2048": 0.6123, "2048x1152": 0.2424, "1152x2048": 0.2424, "3840x2160": 0.5724, "2160x3840": 0.5724}
                    },
                    "gpt-image-2.6": {
                      "low": {"1024x1024": 0.0353, "1024x1536": 0.0285, "1536x1024": 0.0285, "2048x2048": 0.0716, "2048x1152": 0.0283, "1152x2048": 0.0283, "3840x2160": 0.0669, "2160x3840": 0.0669},
                      "medium": {"1024x1024": 0.0622, "1024x1536": 0.0486, "1536x1024": 0.0486, "2048x2048": 0.1263, "2048x1152": 0.0520, "1152x2048": 0.0520, "3840x2160": 0.1225, "2160x3840": 0.1225},
                      "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589, "2048x2048": 0.1531, "2048x1152": 0.0606, "1152x2048": 0.0606, "3840x2160": 0.1431, "2160x3840": 0.1431},
                      "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055, "2048x2048": 0.2721, "2048x1152": 0.1077, "1152x2048": 0.1077, "3840x2160": 0.2544, "2160x3840": 0.2544},
                      "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354, "2048x2048": 0.6123, "2048x1152": 0.2424, "1152x2048": 0.2424, "3840x2160": 0.5724, "2160x3840": 0.5724}
                    }
                  };
                  $perImage := {
                    "gpt-image-1.5": [0.0016, 0.0016],
                    "gpt-image-2": [0.0098, 0.0147],
                    "gpt-image-2.5-flare": [0.0117, 0.0176],
                    "gpt-image-2.5-sunburst": [0.0117, 0.0176],
                    "gpt-image-2.6-flare": [0.0117, 0.0176],
                    "gpt-image-2.6-sunburst": [0.0117, 0.0176]
                  };
                  $model := widgets.model;
                  $family := ($model = "gpt-image-2.6-flare" or $model = "gpt-image-2.6-sunburst") ? "gpt-image-2.6"
                    : ($model = "gpt-image-2.5-flare" or $model = "gpt-image-2.5-sunburst") ? "gpt-image-2.5"
                    : $model;
                  $qualityRaw := $lookup(widgets, "model.quality");
                  $quality := ($qualityRaw != null) ? $qualityRaw : "";
                  $sizeRaw := $lookup(widgets, "model.size");
                  $size := ($sizeRaw != null) ? $sizeRaw : "";
                  $range := $lookup($lookup($ranges, $model), $quality);
                  $preset := $lookup($lookup($lookup($presets, $family), $quality), $size);
                  $out := ($preset != null) ? [$preset, $preset] : $range;
                  $image := $lookup($perImage, $model);
                  $refsRaw := $lookup(inputGroups, "model.images");
                  $refs := ($refsRaw != null) ? $refsRaw : 0;
                  $nRaw := widgets.n;
                  $n := ($nRaw != null and $nRaw != 0) ? $nRaw : 1;
                  $min := ($out[0] + $refs * $image[0]) * $n;
                  $max := ($out[1] + $refs * $image[1]) * $n;
                  $format := ($n = 1) ? {"approximate": true} : {"suffix": "/Run", "approximate": true};
                  ($min = $max)
                    ? {"type": "usd", "usd": $min, "format": $format}
                    : {"type": "range_usd", "min_usd": $min, "max_usd": $max, "format": $format}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        n: int,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)

        model_id = model["model"]
        size = model["size"]
        background = model["background"]
        quality = model["quality"]
        custom_width = model.get("custom_width", 1024)
        custom_height = model.get("custom_height", 1024)
        complexity = model.get("complexity")

        images_dict = model.get("images") or {}
        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_images = sum(get_number_of_images(t) for t in image_tensors)
        mask = model.get("mask")

        if mask is not None and n_images == 0:
            raise ValueError("Cannot use a mask without an input image")

        size = _gpt_image_resolve_size(size, custom_width, custom_height)

        if model_id not in GPT_IMAGE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        if image_tensors:
            flat = _gpt_image_flatten_references(image_tensors)
            files = _gpt_image_reference_files(flat)

            if mask is not None:
                if len(flat) != 1:
                    raise Exception("Cannot use a mask with multiple image")
                ref_image = flat[0]
                if mask.shape[1:] != ref_image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")
                _, height, width = mask.shape
                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
                scaled_mask = downscale_image_tensor(
                    rgba_mask.unsqueeze(0), total_pixels=2048 * 2048
                ).squeeze()
                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_img_byte_arr = BytesIO()
                mask_img.save(mask_img_byte_arr, format="PNG")
                mask_img_byte_arr.seek(0)
                files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/edits", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageEditRequest(
                    model=model_id,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    size=size,
                    moderation="low",
                    complexity=complexity,
                ),
                content_type="multipart/form-data",
                files=files,
            )
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageGenerationRequest(
                    model=model_id,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    size=size,
                    moderation="low",
                    complexity=complexity,
                ),
            )
        return IO.NodeOutput(await validate_and_cast_response(response, cls=cls))


def _gpt_image_v3_inputs(has_complexity: bool):
    complexity = [
        IO.Combo.Input(
            "complexity",
            default="high",
            options=["low", "medium", "high"],
            tooltip="How much the model plans the image before rendering.",
        ),
    ] if has_complexity else []
    return [
        IO.String.Input("prompt", default="", multiline=True),
        *_gpt_image_size_inputs(),
        IO.Combo.Input("background", default="auto", options=["auto", "opaque", "transparent"]),
        IO.Combo.Input("quality", default="low", options=list(GPT_IMAGE_25_QUALITIES)),
        *complexity,
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 17)],
                min=0,
            ),
            optional=True,
            tooltip="Optional reference images. Up to 16.",
        ),
        IO.Image.Input(
            "markup",
            optional=True,
            tooltip="Annotations drawn over the first reference image, showing where to edit. "
            "Must match that image's size. Connect it together with markup_alpha.",
        ),
        IO.Mask.Input(
            "markup_alpha",
            optional=True,
            tooltip="White marks the annotations in markup. The Painter node's MASK output fits directly. "
            "A Load Image mask needs Invert Mask first.",
        ),
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=2147483647,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
            tooltip="Seed controls whether the node should re-run; results are non-deterministic regardless of seed.",
        ),
    ]


def _gpt_image_markup_file(markup: torch.Tensor, markup_alpha: torch.Tensor):
    if markup.shape[1:3] != markup_alpha.shape[-2:]:
        raise ValueError("markup and markup_alpha must be the same size")
    alpha = markup_alpha.reshape(-1, *markup_alpha.shape[-2:])[:1].unsqueeze(-1)
    layer = torch.cat((markup[:1, ..., :3].cpu(), alpha.cpu()), dim=-1)
    scaled = downscale_image_tensor(layer, total_pixels=2048 * 2048).squeeze(0).clamp(0.0, 1.0)
    buffer = BytesIO()
    Image.fromarray((scaled.numpy() * 255).astype(np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    return "markup", ("markup.png", buffer, "image/png")


class OpenAIGPTImageNodeV3(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageNodeV3",
            display_name="OpenAI GPT Image 2.6",
            category="partner/image/OpenAI",
            description="Generates and edits images via OpenAI's GPT Image endpoint, with annotation-guided edits.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        *[IO.DynamicCombo.Option(model_id, _gpt_image_v3_inputs(True)) for model_id in GPT_IMAGE_26_MODELS],
                        IO.DynamicCombo.Option("gpt-image-2.5-flare", _gpt_image_v3_inputs(False)),
                        IO.DynamicCombo.Option("gpt-image-2.5-sunburst", _gpt_image_v3_inputs(False)),
                    ],
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
                    widgets=["model", "model.quality", "model.size"], input_groups=["model.images"]
                ),
                expr="""
                (
                  $ranges := {
                    "gpt-image-2.6": {
                      "low":    [0.0097, 0.1188],
                      "medium": [0.0181, 0.2100],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    },
                    "gpt-image-2.5": {
                      "low":    [0.0023, 0.0283],
                      "medium": [0.0056, 0.0636],
                      "high":   [0.0222, 0.2544],
                      "xhigh":  [0.0388, 0.4523],
                      "max":    [0.0887, 1.0175]
                    }
                  };
                  $presets := {
                    "gpt-image-2.6": {
                      "low": {"1024x1024": 0.0353, "1024x1536": 0.0285, "1536x1024": 0.0285, "2048x2048": 0.0716, "2048x1152": 0.0283, "1152x2048": 0.0283, "3840x2160": 0.0669, "2160x3840": 0.0669},
                      "medium": {"1024x1024": 0.0622, "1024x1536": 0.0486, "1536x1024": 0.0486, "2048x2048": 0.1263, "2048x1152": 0.0520, "1152x2048": 0.0520, "3840x2160": 0.1225, "2160x3840": 0.1225},
                      "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589, "2048x2048": 0.1531, "2048x1152": 0.0606, "1152x2048": 0.0606, "3840x2160": 0.1431, "2160x3840": 0.1431},
                      "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055, "2048x2048": 0.2721, "2048x1152": 0.1077, "1152x2048": 0.1077, "3840x2160": 0.2544, "2160x3840": 0.2544},
                      "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354, "2048x2048": 0.6123, "2048x1152": 0.2424, "1152x2048": 0.2424, "3840x2160": 0.5724, "2160x3840": 0.5724}
                    },
                    "gpt-image-2.5": {
                      "low": {"1024x1024": 0.0084, "1024x1536": 0.0068, "1536x1024": 0.0068, "2048x2048": 0.0170, "2048x1152": 0.0067, "1152x2048": 0.0067, "3840x2160": 0.0159, "2160x3840": 0.0159},
                      "medium": {"1024x1024": 0.0188, "1024x1536": 0.0147, "1536x1024": 0.0147, "2048x2048": 0.0383, "2048x1152": 0.0157, "1152x2048": 0.0157, "3840x2160": 0.0371, "2160x3840": 0.0371},
                      "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589, "2048x2048": 0.1531, "2048x1152": 0.0606, "1152x2048": 0.0606, "3840x2160": 0.1431, "2160x3840": 0.1431},
                      "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055, "2048x2048": 0.2721, "2048x1152": 0.1077, "1152x2048": 0.1077, "3840x2160": 0.2544, "2160x3840": 0.2544},
                      "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354, "2048x2048": 0.6123, "2048x1152": 0.2424, "1152x2048": 0.2424, "3840x2160": 0.5724, "2160x3840": 0.5724}
                    }
                  };
                  $model := widgets.model;
                  $family := ($model = "gpt-image-2.6-flare" or $model = "gpt-image-2.6-sunburst") ? "gpt-image-2.6" : "gpt-image-2.5";
                  $qualityRaw := $lookup(widgets, "model.quality");
                  $quality := ($qualityRaw != null) ? $qualityRaw : "";
                  $sizeRaw := $lookup(widgets, "model.size");
                  $size := ($sizeRaw != null) ? $sizeRaw : "";
                  $range := $lookup($lookup($ranges, $family), $quality);
                  $preset := $lookup($lookup($lookup($presets, $family), $quality), $size);
                  $out := ($preset != null) ? [$preset, $preset] : $range;
                  $refsRaw := $lookup(inputGroups, "model.images");
                  $refs := ($refsRaw != null) ? $refsRaw : 0;
                  $min := $out[0] + $refs * 0.0117;
                  $max := $out[1] + $refs * 0.0176;
                  $format := {"approximate": true};
                  ($min = $max)
                    ? {"type": "usd", "usd": $min, "format": $format}
                    : {"type": "range_usd", "min_usd": $min, "max_usd": $max, "format": $format}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=False)
        size = _gpt_image_resolve_size(model["size"], model["custom_width"], model["custom_height"])
        image_tensors = [t for t in (model.get("images") or {}).values() if t is not None]
        markup = model.get("markup")
        markup_alpha = model.get("markup_alpha")
        if (markup is None) != (markup_alpha is None):
            raise ValueError("Connect markup and markup_alpha together")
        if markup_alpha is not None and not markup_alpha.any():
            markup = None
        if markup is not None and not image_tensors:
            raise ValueError("Cannot use markup without an input image")

        if image_tensors:
            files = _gpt_image_reference_files(_gpt_image_flatten_references(image_tensors))
            if markup is not None:
                files.append(_gpt_image_markup_file(markup, markup_alpha))
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/edits", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageEditRequest(
                    model=model["model"],
                    prompt=model["prompt"],
                    quality=model["quality"],
                    background=model["background"],
                    n=1,
                    size=size,
                    moderation="low",
                    complexity=model.get("complexity"),
                ),
                content_type="multipart/form-data",
                files=files,
            )
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                asset_urls=True,
                data=OpenAIImageGenerationRequest(
                    model=model["model"],
                    prompt=model["prompt"],
                    quality=model["quality"],
                    background=model["background"],
                    n=1,
                    size=size,
                    moderation="low",
                    complexity=model.get("complexity"),
                ),
            )
        return IO.NodeOutput(await validate_and_cast_response(response, cls=cls))


def _gpt_image_gif_inputs():
    return [
        IO.String.Input("prompt", default="", multiline=True),
        IO.Combo.Input(
            "size",
            default="512x512",
            options=[
                "384x384",
                "512x512",
                "640x640",
                "720x720",
                "768x768",
                "960x960",
                "512x288",
                "288x512",
                "1024x576",
                "576x1024",
                "1280x720",
                "720x1280",
            ],
            tooltip="Size of each frame. Larger sizes only fit lower frame counts.",
        ),
        IO.Int.Input(
            "frame_count",
            default=9,
            min=2,
            max=16,
            tooltip="Frames are billed in steps: 2-8, 9, 10-12, 13-16.",
        ),
        IO.Int.Input("frame_duration_ms", default=200, min=20, max=2000, step=10),
        IO.Combo.Input(
            "background",
            default="opaque",
            options=["opaque", "transparent"],
            tooltip="Transparency is kept by Save WEBM with vp9, Save Animated WEBP and Save Animated PNG.",
        ),
        IO.Combo.Input("quality", default="low", options=list(GPT_IMAGE_25_QUALITIES)),
        IO.Combo.Input(
            "complexity",
            default="high",
            options=["low", "medium", "high"],
            tooltip="How much the model plans the animation before rendering.",
        ),
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=2147483647,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
            tooltip="Seed controls whether the node should re-run; results are non-deterministic regardless of seed.",
        ),
    ]


def _gif_to_frames(gif: BytesIO, mode: str) -> tuple[torch.Tensor, float]:
    img = Image.open(gif)
    fps = 1000.0 / img.info["duration"]
    frames = [np.asarray(frame.convert(mode)).astype(np.float32) / 255.0 for frame in ImageSequence.Iterator(img)]
    return torch.from_numpy(np.stack(frames)), fps


class OpenAIGPTImageGIFNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageGIFNode",
            display_name="OpenAI GPT Image 2.6 GIF",
            category="partner/image/OpenAI",
            description="Generates an animated GIF via OpenAI's GPT Image endpoint and outputs its frames and frame rate.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[IO.DynamicCombo.Option(model_id, _gpt_image_gif_inputs()) for model_id in GPT_IMAGE_26_MODELS],
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="frames"),
                IO.Float.Output(display_name="fps"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model.quality", "model.size", "model.frame_count"]),
                expr="""
                (
                  $prices := {
                    "low": {
                      "384x384": [0.0184, 0.0384, 0.0326, 0.0503],
                      "512x512": [0.0238, 0.0503, 0.0445, 0.0716],
                      "640x640": [0.0305, 0.0656, 0.0598, 0.0988],
                      "720x720": [0.0355, 0.0770, 0.0712, 0.1187],
                      "768x768": [0.0387, 0.0843, 0.0786, null],
                      "960x960": [0.0541, 0.1187, null, null],
                      "512x288": [0.0323, 0.0216, 0.0326, 0.0283],
                      "288x512": [0.0323, 0.0216, 0.0326, 0.0283],
                      "1024x576": [0.0680, 0.0476, 0.0786, null],
                      "576x1024": [0.0680, 0.0476, 0.0786, null],
                      "1280x720": [0.0946, 0.0669, null, null],
                      "720x1280": [0.0946, 0.0669, null, null]
                    },
                    "medium": {
                      "384x384": [0.0324, 0.0680, 0.0578, 0.0889],
                      "512x512": [0.0418, 0.0889, 0.0787, 0.1263],
                      "640x640": [0.0538, 0.1160, 0.1058, 0.1744],
                      "720x720": [0.0627, 0.1359, 0.1257, 0.2100],
                      "768x768": [0.0686, 0.1491, 0.1389, null],
                      "960x960": [0.0956, 0.2100, null, null],
                      "512x288": [0.0568, 0.0396, 0.0578, 0.0520],
                      "288x512": [0.0568, 0.0396, 0.0578, 0.0520],
                      "1024x576": [0.1199, 0.0870, 0.1389, null],
                      "576x1024": [0.1199, 0.0870, 0.1389, null],
                      "1280x720": [0.1672, 0.1225, null, null],
                      "720x1280": [0.1672, 0.1225, null, null]
                    },
                    "high": {
                      "384x384": [0.0393, 0.0822, 0.0699, 0.1077],
                      "512x512": [0.0506, 0.1077, 0.0954, 0.1531],
                      "640x640": [0.0652, 0.1405, 0.1282, 0.2114],
                      "720x720": [0.0760, 0.1647, 0.1524, 0.2544],
                      "768x768": [0.0830, 0.1806, 0.1683, null],
                      "960x960": [0.1158, 0.2544, null, null],
                      "512x288": [0.0704, 0.0462, 0.0699, 0.0606],
                      "288x512": [0.0704, 0.0462, 0.0699, 0.0606],
                      "1024x576": [0.1487, 0.1016, 0.1683, null],
                      "576x1024": [0.1487, 0.1016, 0.1683, null],
                      "1280x720": [0.2075, 0.1431, null, null],
                      "720x1280": [0.2075, 0.1431, null, null]
                    },
                    "xhigh": {
                      "384x384": [0.0698, 0.1462, 0.1242, 0.1915],
                      "512x512": [0.0900, 0.1915, 0.1695, 0.2721],
                      "640x640": [0.1159, 0.2498, 0.2278, 0.3758],
                      "720x720": [0.1350, 0.2928, 0.2709, 0.4523],
                      "768x768": [0.1476, 0.3211, 0.2991, null],
                      "960x960": [0.2059, 0.4523, null, null],
                      "512x288": [0.1244, 0.0822, 0.1242, 0.1077],
                      "288x512": [0.1244, 0.0822, 0.1242, 0.1077],
                      "1024x576": [0.2629, 0.1806, 0.2991, null],
                      "576x1024": [0.2629, 0.1806, 0.2991, null],
                      "1280x720": [0.3667, 0.2544, null, null],
                      "720x1280": [0.3667, 0.2544, null, null]
                    },
                    "max": {
                      "384x384": [0.1571, 0.3289, 0.2795, 0.4309],
                      "512x512": [0.2025, 0.4309, 0.3815, 0.6123],
                      "640x640": [0.2608, 0.5621, 0.5127, 0.8455],
                      "720x720": [0.3038, 0.6589, 0.6094, 1.0175],
                      "768x768": [0.3320, 0.7224, 0.6730, null],
                      "960x960": [0.4632, 1.0175, null, null],
                      "512x288": [0.2783, 0.1850, 0.2795, 0.2424],
                      "288x512": [0.2783, 0.1850, 0.2795, 0.2424],
                      "1024x576": [0.5880, 0.4063, 0.6730, null],
                      "576x1024": [0.5880, 0.4063, 0.6730, null],
                      "1280x720": [0.8203, 0.5724, null, null],
                      "720x1280": [0.8203, 0.5724, null, null]
                    }
                  };
                  $qualityRaw := $lookup(widgets, "model.quality");
                  $quality := ($qualityRaw != null) ? $qualityRaw : "low";
                  $sizeRaw := $lookup(widgets, "model.size");
                  $size := ($sizeRaw != null) ? $sizeRaw : "512x512";
                  $framesRaw := $lookup(widgets, "model.frame_count");
                  $frames := ($framesRaw != null) ? $framesRaw : 9;
                  $tier := ($frames <= 8) ? 0 : ($frames = 9) ? 1 : ($frames <= 12) ? 2 : 3;
                  $usd := $lookup($lookup($prices, $quality), $size)[$tier];
                  ($usd != null)
                    ? {"type": "usd", "usd": $usd, "format": {"approximate": true}}
                    : {"type": "text", "text": "Size too large for this frame count"}
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=False)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
            response_model=OpenAIImageGenerationResponse,
            asset_urls=True,
            data=OpenAIImageGenerationRequest(
                model=model["model"],
                prompt=model["prompt"],
                quality=model["quality"],
                background=model["background"],
                size=model["size"],
                moderation="low",
                output_format="gif",
                complexity=model["complexity"],
                frame_count=model["frame_count"],
                frame_duration_ms=model["frame_duration_ms"],
            ),
        )
        frames, fps = _gif_to_frames(
            await _single_image_payload(response, cls),
            "RGBA" if model["background"] == "transparent" else "RGB",
        )
        return IO.NodeOutput(frames, fps)


def _gpt_image_svg_inputs():
    return [
        IO.String.Input("prompt", default="", multiline=True),
        IO.Combo.Input("size", default="1024x1024", options=["1024x1024", "1024x1536", "1536x1024"]),
        IO.Combo.Input("background", default="auto", options=["auto", "opaque", "transparent"]),
        IO.Combo.Input(
            "quality",
            default="low",
            options=list(GPT_IMAGE_25_QUALITIES),
            tooltip="Quality of the reference image the SVG is drawn from.",
        ),
        IO.Combo.Input(
            "complexity",
            default="high",
            options=["low", "medium", "high"],
            tooltip="How much the model plans the reference image before rendering.",
        ),
        IO.Int.Input(
            "seed",
            default=42,
            min=0,
            max=2147483647,
            display_mode=IO.NumberDisplay.number,
            control_after_generate=True,
            tooltip="Seed controls whether the node should re-run; results are non-deterministic regardless of seed.",
        ),
    ]


class OpenAIGPTImageSVGNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageSVGNode",
            display_name="OpenAI GPT Image 2.6 SVG",
            category="partner/image/OpenAI",
            description="Generates an SVG via OpenAI's GPT Image endpoint.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[IO.DynamicCombo.Option(model_id, _gpt_image_svg_inputs()) for model_id in GPT_IMAGE_26_MODELS],
                ),
            ],
            outputs=[IO.SVG.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["model.quality", "model.size"]),
                expr="""
                (
                  $prices := {
                    "low": {"1024x1024": 0.0353, "1024x1536": 0.0285, "1536x1024": 0.0285},
                    "medium": {"1024x1024": 0.0622, "1024x1536": 0.0486, "1536x1024": 0.0486},
                    "high": {"1024x1024": 0.0753, "1024x1536": 0.0589, "1536x1024": 0.0589},
                    "xhigh": {"1024x1024": 0.1339, "1024x1536": 0.1055, "1536x1024": 0.1055},
                    "max": {"1024x1024": 0.3013, "1024x1536": 0.2354, "1536x1024": 0.2354}
                  };
                  $qualityRaw := $lookup(widgets, "model.quality");
                  $quality := ($qualityRaw != null) ? $qualityRaw : "low";
                  $sizeRaw := $lookup(widgets, "model.size");
                  $size := ($sizeRaw != null) ? $sizeRaw : "1024x1024";
                  $image := $lookup($lookup($prices, $quality), $size);
                  {
                    "type": "range_usd",
                    "min_usd": $image + 0.0011,
                    "max_usd": $image + 0.0084,
                    "format": {"approximate": true}
                  }
                )
                """,
            ),
        )

    @classmethod
    async def execute(cls, model: dict) -> IO.NodeOutput:
        validate_string(model["prompt"], strip_whitespace=False)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=OpenAIImageGenerationRequest(
                model=model["model"],
                prompt=model["prompt"],
                quality=model["quality"],
                background=model["background"],
                size=model["size"],
                moderation="low",
                output_format="svg",
                complexity=model["complexity"],
                svg=OpenAIImageSVGOptions(model="gpt-6-luna"),
            ),
        )
        return IO.NodeOutput(SVG([await _single_image_payload(response, cls)]))


class OpenAIChatNode(IO.ComfyNode):
    """
    Node to generate text responses from an OpenAI model.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIChatNode",
            display_name="OpenAI ChatGPT",
            category="partner/text/OpenAI",
            essentials_category="Text Generation",
            description="Generate text responses from an OpenAI model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text inputs to the model, used to generate a response.",
                ),
                IO.Boolean.Input(
                    "persist_context",
                    default=False,
                    tooltip="This parameter is deprecated and has no effect.",
                    advanced=True,
                ),
                IO.Combo.Input(
                    "model",
                    options=SupportedOpenAIModel,
                    tooltip="The model used to generate the response",
                ),
                IO.Image.Input(
                    "images",
                    tooltip="Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    optional=True,
                ),
                IO.Custom("OPENAI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Chat Input Files node.",
                ),
                IO.Custom("OPENAI_CHAT_CONFIG").Input(
                    "advanced_options",
                    optional=True,
                    tooltip="Optional configuration for the model. Accepts inputs from the OpenAI Chat Advanced Options node.",
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
                  $contains($m, "gpt-6-astra") ? {
                    "type": "list_usd",
                    "usd": [0.0143, 0.0715],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-6.1-sol") ? {
                    "type": "list_usd",
                    "usd": [0.00286, 0.0143],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-6.1-terra") ? {
                    "type": "list_usd",
                    "usd": [0.00143, 0.00715],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-6-sol") ? {
                    "type": "list_usd",
                    "usd": [0.00286, 0.0143],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-6-luna") ? {
                    "type": "list_usd",
                    "usd": [0.000143, 0.000715],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "o3") ? {
                    "type": "list_usd",
                    "usd": [0.01, 0.04],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-4.1-mini") ? {
                    "type": "list_usd",
                    "usd": [0.0004, 0.0016],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-4.1") ? {
                    "type": "list_usd",
                    "usd": [0.002, 0.008],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5.6-terra") ? {
                    "type": "list_usd",
                    "usd": [0.0025, 0.015],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5.6-luna") ? {
                    "type": "list_usd",
                    "usd": [0.001, 0.006],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5.6") ? {
                    "type": "list_usd",
                    "usd": [0.005, 0.03],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5.5-pro") ? {
                    "type": "list_usd",
                    "usd": [0.03, 0.18],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5.5") ? {
                    "type": "list_usd",
                    "usd": [0.005, 0.03],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5-nano") ? {
                    "type": "list_usd",
                    "usd": [0.00005, 0.0004],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5-mini") ? {
                    "type": "list_usd",
                    "usd": [0.00025, 0.002],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-5") ? {
                    "type": "list_usd",
                    "usd": [0.00125, 0.01],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : {"type": "text", "text": "Token-based"}
                )
                """,
            ),
        )

    @classmethod
    def get_message_content_from_response(cls, response: OpenAIResponse) -> list[OutputContent]:
        """Extract message content from the API response."""
        for output in response.output:
            if output.type == "message":
                return output.content
        raise TypeError("No output message found in response")

    @classmethod
    def get_text_from_message_content(cls, message_content: list[OutputContent]) -> str:
        """Extract text content from message content."""
        for content_item in message_content:
            if content_item.type == "output_text":
                return str(content_item.text)
        return "No text output found in response"

    @classmethod
    def tensor_to_input_image_content(cls, image: torch.Tensor, detail_level: str = "auto") -> InputImageContent:
        """Convert a tensor to an input image content object."""
        return InputImageContent(
            detail=detail_level,
            image_url=f"data:image/png;base64,{tensor_to_base64_string(image)}",
            type="input_image",
        )

    @classmethod
    def create_input_message_contents(
        cls,
        prompt: str,
        image: torch.Tensor | None = None,
        files: list[InputFileContent] | None = None,
    ) -> list[InputTextContent | InputImageContent | InputFileContent]:
        """Create a list of input message contents from prompt and optional image."""
        content_list: list[InputTextContent | InputImageContent | InputFileContent] = [
            InputTextContent(text=prompt, type="input_text"),
        ]
        if image is not None:
            for i in range(image.shape[0]):
                content_list.append(
                    InputImageContent(
                        detail="auto",
                        image_url=f"data:image/png;base64,{tensor_to_base64_string(image[i].unsqueeze(0))}",
                        type="input_image",
                    )
                )
        if files is not None:
            content_list.extend(files)
        return content_list

    @classmethod
    async def execute(
        cls,
        prompt: str,
        persist_context: bool = False,
        model: SupportedOpenAIModel = SupportedOpenAIModel.gpt_5.value,
        images: torch.Tensor | None = None,
        files: list[InputFileContent] | None = None,
        advanced_options: ModelResponseProperties | None = None,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        if advanced_options is not None and advanced_options.reasoning is not None:
            effort = advanced_options.reasoning.effort
            supported = SUPPORTED_REASONING_EFFORTS.get(model)
            if supported is not None and effort not in supported:
                if not supported:
                    raise ValueError(f"{model} is not a reasoning model; set reasoning_effort to 'default'.")
                raise ValueError(
                    f"{model} does not support reasoning_effort '{effort}'. Supported: {', '.join(supported)} or 'default'."
                )

        # Create response
        create_response = await sync_op(
            cls,
            ApiEndpoint(path=RESPONSES_ENDPOINT, method="POST"),
            response_model=OpenAIResponse,
            data=OpenAICreateResponse(
                input=[
                    InputMessage(
                        content=cls.create_input_message_contents(prompt, images, files),
                        role="user",
                    ),
                ],
                store=True,
                stream=False,
                model=model,
                previous_response_id=None,
                **(advanced_options.model_dump(exclude_none=True) if advanced_options else {}),
            ),
        )
        response_id = create_response.id

        # Get result output
        result_response = await poll_op(
            cls,
            ApiEndpoint(path=f"{RESPONSES_ENDPOINT}/{response_id}"),
            response_model=OpenAIResponse,
            status_extractor=lambda response: response.status,
            completed_statuses=["incomplete", "completed"],
        )
        return IO.NodeOutput(cls.get_text_from_message_content(cls.get_message_content_from_response(result_response)))


class OpenAIInputFiles(IO.ComfyNode):
    """
    Loads and formats input files for OpenAI API.
    """

    @classmethod
    def define_schema(cls):
        """
        For details about the supported file input types, see:
        https://platform.openai.com/docs/guides/pdf-files?api-mode=responses
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf"))
            and f.stat().st_size < 32 * 1024 * 1024
        ]
        input_files = sorted(input_files, key=lambda x: x.name)
        input_files = [f.name for f in input_files]
        return IO.Schema(
            node_id="OpenAIInputFiles",
            display_name="OpenAI ChatGPT Input Files",
            category="partner/text/OpenAI",
            description="Loads and prepares input files (text, pdf, etc.) to include as inputs for the OpenAI Chat Node. The files will be read by the OpenAI model when generating a response. 🛈 TIP: Can be chained together with other OpenAI Input File nodes.",
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=input_files,
                    default=input_files[0] if input_files else None,
                    tooltip="Input files to include as context for the model. Only accepts text (.txt) and PDF (.pdf) files for now.",
                ),
                IO.Custom("OPENAI_INPUT_FILES").Input(
                    "OPENAI_INPUT_FILES",
                    tooltip="An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Custom("OPENAI_INPUT_FILES").Output(),
            ],
        )

    @classmethod
    def create_input_file_content(cls, file_path: str) -> InputFileContent:
        return InputFileContent(
            file_data=text_filepath_to_data_uri(file_path),
            filename=os.path.basename(file_path),
            type="input_file",
        )

    @classmethod
    def execute(cls, file: str, OPENAI_INPUT_FILES: list[InputFileContent] = []) -> IO.NodeOutput:
        """
        Loads and formats input files for OpenAI API.
        """
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = cls.create_input_file_content(file_path)
        files = [input_file_content] + OPENAI_INPUT_FILES
        return IO.NodeOutput(files)


class OpenAIChatConfig(IO.ComfyNode):
    """Allows setting additional configuration for the OpenAI Chat Node."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIChatConfig",
            display_name="OpenAI ChatGPT Advanced Options",
            category="partner/text/OpenAI",
            description="Allows specifying advanced configuration options for the OpenAI Chat Nodes.",
            inputs=[
                IO.Combo.Input(
                    "truncation",
                    options=["auto", "disabled"],
                    default="auto",
                    tooltip="The truncation strategy to use for the model response. auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.disabled: If a model response will exceed the context window size for a model, the request will fail with a 400 error",
                    advanced=True,
                ),
                IO.Int.Input(
                    "max_output_tokens",
                    min=16,
                    default=4096,
                    max=16384,
                    tooltip="An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens",
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "instructions",
                    multiline=True,
                    optional=True,
                    tooltip="Instructions for the model on how to generate the response",
                ),
                IO.Combo.Input(
                    "reasoning_effort",
                    options=REASONING_EFFORT_OPTIONS,
                    default="default",
                    optional=True,
                    tooltip="Supported levels vary by model. An unsupported level stops the run and lists the valid ones. "
                    "'default' lets the model choose.",
                ),
            ],
            outputs=[
                IO.Custom("OPENAI_CHAT_CONFIG").Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        truncation: bool,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> IO.NodeOutput:
        """
        Configure advanced options for the OpenAI Chat Node.

        Note:
            While `top_p` and `temperature` are listed as properties in the
            spec, they are not supported for all models (e.g., gpt-6.1-sol).
            They are not exposed as inputs at all to avoid having to manually
            remove depending on model choice.
        """
        return IO.NodeOutput(
            ModelResponseProperties(
                instructions=instructions,
                truncation=truncation,
                max_output_tokens=max_output_tokens,
                reasoning=None if reasoning_effort in (None, "default") else Reasoning(effort=reasoning_effort),
            )
        )


class OpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            OpenAIGPTImage1,
            OpenAIGPTImageNodeV2,
            OpenAIGPTImageNodeV3,
            OpenAIGPTImageGIFNode,
            OpenAIGPTImageSVGNode,
            OpenAIChatNode,
            OpenAIInputFiles,
            OpenAIChatConfig,
        ]


async def comfy_entrypoint() -> OpenAIExtension:
    return OpenAIExtension()
