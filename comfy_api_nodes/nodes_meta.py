import base64
import re
from io import BytesIO

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension
from comfy_api_nodes.apis.meta import (
    MuseImageEditRequest,
    MuseImageInput,
    MuseImageRequest,
    MuseImageResponse,
    MuseImageToolEnablement,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    sync_op,
    upload_images_to_comfyapi,
    validate_string,
)

GENERATIONS_PATH = "/proxy/meta/v1/images/generations"
EDITS_PATH = "/proxy/meta/v1/images/edits"
MUSE_IMAGE_MODELS = ["muse-image-1.0"]
MAX_INPUT_IMAGES = 10
ASPECT_RATIOS = ["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "21:9", "9:21", "2:1", "1:2"]
REASONING_STRENGTHS = ["high", "low"]

_IMAGE_REF_RE = re.compile(r"@image(?P<idx>\d*)(?!\w)", re.IGNORECASE | re.ASCII)


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
        parts.append(f"image {idx}")
        pos = match.end()
        prev_end = match.end()
    parts.append(prompt[pos:])
    return "".join(parts)


def _size(aspect_ratio: str) -> str | None:
    return None if aspect_ratio == "auto" else aspect_ratio.replace(":", "x")


def _decode_images(response: MuseImageResponse) -> torch.Tensor:
    images = [
        bytesio_to_image_tensor(BytesIO(base64.b64decode(item.b64_json)))
        for item in response.data
        if item.b64_json
    ]
    if not images:
        raise Exception("The response contains no images.")
    return torch.cat(images)


def _reasoning_strength_input() -> IO.Combo.Input:
    return IO.Combo.Input(
        "reasoning_strength",
        options=REASONING_STRENGTHS,
        tooltip="How much the model thinks, plans and self-refines before rendering.",
    )


def _t2i_model_option(model_id: str) -> IO.DynamicCombo.Option:
    return IO.DynamicCombo.Option(
        model_id,
        [
            IO.String.Input(
                "prompt",
                multiline=True,
                default="",
                tooltip="Prompt describing the image. The model reasons about the prompt, and may use "
                "its built-in web and image search, before rendering.",
            ),
            IO.Combo.Input(
                "aspect_ratio",
                options=ASPECT_RATIOS,
                tooltip="Aspect ratio of the output. Images are rendered at about 2.5 megapixels "
                "(1:1 is 1600x1600, 16:9 is 2048x1152); 'auto' lets the model choose from the prompt.",
            ),
            _reasoning_strength_input(),
            *_tool_toggle_inputs(),
            _seed_input(),
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
            IO.Combo.Input(
                "aspect_ratio",
                options=ASPECT_RATIOS,
                tooltip="Aspect ratio of the output. Images are rendered at about 2.5 megapixels "
                "(1:1 is 1600x1600, 16:9 is 2048x1152); 'auto' keeps the aspect ratio of the input.",
            ),
            _reasoning_strength_input(),
            *_tool_toggle_inputs(),
            _seed_input(),
        ],
    )


def _tool_toggle_inputs() -> list[IO.Boolean.Input]:
    return [
        IO.Boolean.Input(
            "enable_web_search",
            default=True,
            advanced=True,
            tooltip="Lets the model search the web for facts and live information while planning the image.",
        ),
        IO.Boolean.Input(
            "enable_image_search",
            default=True,
            advanced=True,
            tooltip="Lets the model search for reference images while planning the image.",
        ),
        IO.Boolean.Input(
            "enable_shell",
            default=True,
            advanced=True,
            tooltip="Lets the model run code while planning, for precise layouts, charts and diagrams; "
            "when off, quantities and alignment are approximated.",
        ),
    ]


def _tool_enablement(model: dict) -> MuseImageToolEnablement | None:
    if model["enable_web_search"] and model["enable_image_search"] and model["enable_shell"]:
        return None
    return MuseImageToolEnablement(
        enable_image_search=model["enable_image_search"],
        enable_web_search=model["enable_web_search"],
        enable_shell=model["enable_shell"],
    )


def _seed_input() -> IO.Int.Input:
    return IO.Int.Input(
        "seed",
        default=42,
        min=0,
        max=2147483647,
        step=1,
        display_mode=IO.NumberDisplay.number,
        control_after_generate=True,
        tooltip="Seed to determine if node should re-run; the API has no seed, "
        "so actual results are nondeterministic regardless of this value.",
    )


def _price_badge() -> IO.PriceBadge:
    return IO.PriceBadge(expr="""{"type":"usd","usd":0.0143}""")


class MetaMuseImageTextToImageApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MetaMuseImageTextToImageApi",
            display_name="Meta Muse Image Text to Image",
            category="partner/image/Meta",
            description="Generates images from a text prompt using Meta's Muse Image model, "
            "which reasons about the prompt before rendering.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_t2i_model_option(model_id) for model_id in MUSE_IMAGE_MODELS],
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
        response = await sync_op(
            cls,
            ApiEndpoint(path=GENERATIONS_PATH, method="POST"),
            response_model=MuseImageResponse,
            data=MuseImageRequest(
                model=model["model"],
                prompt=model["prompt"],
                size=_size(model["aspect_ratio"]),
                reasoning_strength=model["reasoning_strength"],
                tool_enablement=_tool_enablement(model),
            ),
        )
        return IO.NodeOutput(_decode_images(response))


class MetaMuseImageEditApi(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MetaMuseImageEditApi",
            display_name="Meta Muse Image Edit",
            category="partner/image/Meta",
            description=f"Edits or combines up to {MAX_INPUT_IMAGES} reference images guided by a text prompt "
            "using Meta's Muse Image model.",
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[_edit_model_option(model_id) for model_id in MUSE_IMAGE_MODELS],
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
        prompt = _resolve_image_refs(model["prompt"], len(reference_images))
        urls = await upload_images_to_comfyapi(
            cls,
            [image[..., :3] for image in reference_images],
            max_images=MAX_INPUT_IMAGES,
            mime_type="image/png",
            wait_label="Uploading reference images",
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path=EDITS_PATH, method="POST"),
            response_model=MuseImageResponse,
            data=MuseImageEditRequest(
                model=model["model"],
                prompt=prompt,
                size=_size(model["aspect_ratio"]),
                reasoning_strength=model["reasoning_strength"],
                tool_enablement=_tool_enablement(model),
                images=[MuseImageInput(image_url=url) for url in urls],
            ),
        )
        return IO.NodeOutput(_decode_images(response))


class MetaApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            MetaMuseImageTextToImageApi,
            MetaMuseImageEditApi,
        ]


async def comfy_entrypoint() -> MetaApiExtension:
    return MetaApiExtension()
