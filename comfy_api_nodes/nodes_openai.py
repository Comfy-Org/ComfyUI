import base64
import os
from enum import Enum
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import folder_paths
from comfy.utils import common_upscale
from comfy_api.latest import IO, ComfyExtension, Input
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
    OpenAIResponse,
    OutputContent,
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
    gpt_5_5_pro = "gpt-5.5-pro"
    gpt_5_5 = "gpt-5.5"
    gpt_5 = "gpt-5"
    gpt_5_mini = "gpt-5-mini"
    gpt_5_nano = "gpt-5-nano"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"
    o4_mini = "o4-mini"
    o3 = "o3"
    o1_pro = "o1-pro"
    o1 = "o1"


async def validate_and_cast_response(response, timeout: int = None) -> torch.Tensor:
    """Validates and casts a response to a torch.Tensor.

    Args:
        response: The response to validate and cast.
        timeout: Request timeout in seconds. Defaults to None (no timeout).

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
        if img_data.b64_json:
            img_io = BytesIO(base64.b64decode(img_data.b64_json))
        elif img_data.url:
            img_io = BytesIO()
            await download_url_to_bytesio(img_data.url, img_io, timeout=timeout)
        else:
            raise ValueError("Invalid image payload – neither URL nor base64 data present.")

        pil_img = Image.open(img_io).convert("RGBA")
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


class OpenAIDalle2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIDalle2",
            display_name="OpenAI DALL·E 2",
            category="partner/image/OpenAI",
            description="Generates images synchronously via OpenAI's DALL·E 2 endpoint.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for DALL·E",
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
                    "size",
                    default="1024x1024",
                    options=["256x256", "512x512", "1024x1024"],
                    tooltip="Image size",
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
                depends_on=IO.PriceBadgeDepends(widgets=["size", "n"]),
                expr="""
                (
                  $size := widgets.size;
                  $nRaw := widgets.n;
                  $n := ($nRaw != null and $nRaw != 0) ? $nRaw : 1;

                  $base :=
                    $contains($size, "256x256") ? 0.016 :
                    $contains($size, "512x512") ? 0.018 :
                    0.02;

                  {"type":"usd","usd": $round($base * $n, 3)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        seed=0,
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-2"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        img_binary = None

        if image is not None and mask is not None:
            path = "/proxy/openai/images/edits"
            content_type = "multipart/form-data"
            request_class = OpenAIImageEditRequest

            input_tensor = image.squeeze().cpu()
            height, width, channels = input_tensor.shape
            rgba_tensor = torch.ones(height, width, 4, device="cpu")
            rgba_tensor[:, :, :channels] = input_tensor

            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            rgba_tensor[:, :, 3] = 1 - mask.squeeze().cpu()

            rgba_tensor = downscale_image_tensor(rgba_tensor.unsqueeze(0)).squeeze()

            image_np = (rgba_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr  # .getvalue()
            img_binary.name = "image.png"
        elif image is not None or mask is not None:
            raise Exception("Dall-E 2 image editing requires an image AND a mask")

        response = await sync_op(
            cls,
            ApiEndpoint(path=path, method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=request_class(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed,
            ),
            files=(
                {
                    "image": ("image.png", img_binary, "image/png"),
                }
                if img_binary
                else None
            ),
            content_type=content_type,
        )

        return IO.NodeOutput(await validate_and_cast_response(response))


class OpenAIDalle3(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIDalle3",
            display_name="OpenAI DALL·E 3",
            category="partner/image/OpenAI",
            description="Generates images synchronously via OpenAI's DALL·E 3 endpoint.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for DALL·E",
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
                    default="standard",
                    options=["standard", "hd"],
                    tooltip="Image quality",
                    optional=True,
                ),
                IO.Combo.Input(
                    "style",
                    default="natural",
                    options=["natural", "vivid"],
                    tooltip="Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "size",
                    default="1024x1024",
                    options=["1024x1024", "1024x1792", "1792x1024"],
                    tooltip="Image size",
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
                depends_on=IO.PriceBadgeDepends(widgets=["size", "quality"]),
                expr="""
                (
                  $size := widgets.size;
                  $q := widgets.quality;
                  $hd := $contains($q, "hd");

                  $price :=
                    $contains($size, "1024x1024")
                      ? ($hd ? 0.08 : 0.04)
                      : (($contains($size, "1792x1024") or $contains($size, "1024x1792"))
                          ? ($hd ? 0.12 : 0.08)
                          : 0.04);

                  {"type":"usd","usd": $price}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        seed=0,
        style="natural",
        quality="standard",
        size="1024x1024",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-3"

        # build the operation
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
            response_model=OpenAIImageGenerationResponse,
            data=OpenAIImageGenerationRequest(
                model=model,
                prompt=prompt,
                quality=quality,
                size=size,
                style=style,
                seed=seed,
            ),
        )

        return IO.NodeOutput(await validate_and_cast_response(response))


def calculate_tokens_price_image_1(response: OpenAIImageGenerationResponse) -> float | None:
    # https://platform.openai.com/docs/pricing
    return ((response.usage.input_tokens * 10.0) + (response.usage.output_tokens * 40.0)) / 1_000_000.0


def calculate_tokens_price_image_1_5(response: OpenAIImageGenerationResponse) -> float | None:
    return ((response.usage.input_tokens * 8.0) + (response.usage.output_tokens * 32.0)) / 1_000_000.0


def calculate_tokens_price_image_2_0(response: OpenAIImageGenerationResponse) -> float | None:
    return ((response.usage.input_tokens * 8.0) + (response.usage.output_tokens * 30.0)) / 1_000_000.0


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
                    options=["gpt-image-1", "gpt-image-1.5", "gpt-image-2"],
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
                    "gpt-image-1": {
                      "low":    [0.011, 0.02],
                      "medium": [0.042, 0.07],
                      "high":   [0.167, 0.25]
                    },
                    "gpt-image-1.5": {
                      "low":    [0.009, 0.02],
                      "medium": [0.034, 0.062],
                      "high":   [0.133, 0.22]
                    },
                    "gpt-image-2": {
                      "low":    [0.0048, 0.019],
                      "medium": [0.041, 0.168],
                      "high":   [0.165, 0.67]
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
        model: str = "gpt-image-1",
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
        elif model in ("gpt-image-1", "gpt-image-1.5"):
            if size not in ("auto", "1024x1024", "1024x1536", "1536x1024"):
                raise ValueError(f"Resolution {size} is only supported by GPT Image 2 model")

        if model == "gpt-image-1":
            price_extractor = calculate_tokens_price_image_1
        elif model == "gpt-image-1.5":
            price_extractor = calculate_tokens_price_image_1_5
        elif model == "gpt-image-2":
            price_extractor = calculate_tokens_price_image_2_0
            if background == "transparent":
                raise ValueError("Transparent background is not supported for GPT Image 2 model")
        else:
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
                price_extractor=price_extractor,
            )
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
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
                price_extractor=price_extractor,
            )
        return IO.NodeOutput(await validate_and_cast_response(response))


def _gpt_image_shared_inputs():
    """Inputs shared by all GPT Image models (quality + reference images + mask)."""
    return [
        IO.Combo.Input(
            "quality",
            default="low",
            options=["low", "medium", "high"],
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
    """Per-model widget set for legacy gpt-image-1 / gpt-image-1.5 (4 base sizes, transparent bg allowed)."""
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


class OpenAIGPTImageNodeV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageNodeV2",
            display_name="OpenAI GPT Image 2",
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
                        IO.DynamicCombo.Option(
                            "gpt-image-2",
                            [
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
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
                                ),
                                IO.Int.Input(
                                    "custom_height",
                                    default=1024,
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip="Used only when `size` is 'Custom'. Must be a multiple of 16.",
                                ),
                                IO.Combo.Input(
                                    "background",
                                    default="auto",
                                    options=["auto", "opaque"],
                                    tooltip="Return image with or without background.",
                                ),
                                *_gpt_image_shared_inputs(),
                            ],
                        ),
                        IO.DynamicCombo.Option("gpt-image-1.5", _gpt_image_legacy_model_inputs()),
                        IO.DynamicCombo.Option("gpt-image-1", _gpt_image_legacy_model_inputs()),
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
                depends_on=IO.PriceBadgeDepends(widgets=["model", "model.quality", "n"]),
                expr="""
                (
                  $ranges := {
                    "gpt-image-1": {
                      "low":    [0.011, 0.02],
                      "medium": [0.042, 0.07],
                      "high":   [0.167, 0.25]
                    },
                    "gpt-image-1.5": {
                      "low":    [0.009, 0.02],
                      "medium": [0.034, 0.062],
                      "high":   [0.133, 0.22]
                    },
                    "gpt-image-2": {
                      "low":    [0.0048, 0.019],
                      "medium": [0.041, 0.168],
                      "high":   [0.165, 0.67]
                    }
                  };
                  $range := $lookup($lookup($ranges, widgets.model), $lookup(widgets, "model.quality"));
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

        images_dict = model.get("images") or {}
        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_images = sum(get_number_of_images(t) for t in image_tensors)
        mask = model.get("mask")

        if mask is not None and n_images == 0:
            raise ValueError("Cannot use a mask without an input image")

        if size == "Custom":
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
            size = f"{custom_width}x{custom_height}"

        if model_id == "gpt-image-1":
            price_extractor = calculate_tokens_price_image_1
        elif model_id == "gpt-image-1.5":
            price_extractor = calculate_tokens_price_image_1_5
        elif model_id == "gpt-image-2":
            price_extractor = calculate_tokens_price_image_2_0
        else:
            raise ValueError(f"Unknown model: {model_id}")

        if image_tensors:
            flat: list[torch.Tensor] = []
            for tensor in image_tensors:
                if len(tensor.shape) == 4:
                    flat.extend(tensor[i : i + 1] for i in range(tensor.shape[0]))
                else:
                    flat.append(tensor.unsqueeze(0))

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
                data=OpenAIImageEditRequest(
                    model=model_id,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    size=size,
                    moderation="low",
                ),
                content_type="multipart/form-data",
                files=files,
                price_extractor=price_extractor,
            )
        else:
            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                data=OpenAIImageGenerationRequest(
                    model=model_id,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    size=size,
                    moderation="low",
                ),
                price_extractor=price_extractor,
            )
        return IO.NodeOutput(await validate_and_cast_response(response))


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
                  $contains($m, "o4-mini") ? {
                    "type": "list_usd",
                    "usd": [0.0011, 0.0044],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "o1-pro") ? {
                    "type": "list_usd",
                    "usd": [0.15, 0.6],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "o1") ? {
                    "type": "list_usd",
                    "usd": [0.015, 0.06],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "o3-mini") ? {
                    "type": "list_usd",
                    "usd": [0.0011, 0.0044],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "o3") ? {
                    "type": "list_usd",
                    "usd": [0.01, 0.04],
                    "format": { "approximate": true, "separator": "-", "suffix": " per 1K tokens" }
                  }
                  : $contains($m, "gpt-4.1-nano") ? {
                    "type": "list_usd",
                    "usd": [0.0001, 0.0004],
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
                    tooltip="An upper bound for the number of tokens that can be generated for a response, including visible output tokens",
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "instructions",
                    multiline=True,
                    optional=True,
                    tooltip="Instructions for the model on how to generate the response",
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
    ) -> IO.NodeOutput:
        """
        Configure advanced options for the OpenAI Chat Node.

        Note:
            While `top_p` and `temperature` are listed as properties in the
            spec, they are not supported for all models (e.g., o4-mini).
            They are not exposed as inputs at all to avoid having to manually
            remove depending on model choice.
        """
        return IO.NodeOutput(
            ModelResponseProperties(
                instructions=instructions,
                truncation=truncation,
                max_output_tokens=max_output_tokens,
            )
        )


class OpenAIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            OpenAIDalle2,
            OpenAIDalle3,
            OpenAIGPTImage1,
            OpenAIGPTImageNodeV2,
            OpenAIChatNode,
            OpenAIInputFiles,
            OpenAIChatConfig,
        ]


async def comfy_entrypoint() -> OpenAIExtension:
    return OpenAIExtension()
