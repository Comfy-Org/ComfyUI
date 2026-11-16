import json
from typing import Any

import torch
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.patina import (
    ImageSize,
    PatinaExtractRequest,
    PatinaMaterialRequest,
    PatinaPBRMapsRequest,
    PatinaResult,
    PatinaStatusResponse,
    PatinaSubmitResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    convert_mask_to_image,
    download_url_as_bytesio,
    downscale_image_tensor_by_max_side,
    poll_op,
    resize_mask_to_image,
    sync_op,
    upload_image_to_comfyapi,
    validate_image_dimensions,
    validate_string,
)


PATINA_ENDPOINT = "/proxy/fal/fal-ai/patina"
PATINA_MAPS = ["basecolor", "normal", "roughness", "metalness", "height"]
_IMAGE_SIZES = [
    ("1:1 (1024x1024)", "square_hd", 1024, 1024),
    ("1:1 (512x512)", "square", 512, 512),
    ("4:3 (1024x768)", "landscape_4_3", 1024, 768),
    ("3:4 (768x1024)", "portrait_4_3", 768, 1024),
    ("16:9 (1024x576)", "landscape_16_9", 1024, 576),
    ("9:16 (576x1024)", "portrait_16_9", 576, 1024),
]
_LABEL_TO_PRESET = {label: preset for label, preset, _, _ in _IMAGE_SIZES}
_PRESET_MP = json.dumps({label: w * h / 1048576 for label, _, w, h in _IMAGE_SIZES})
_NMAPS = (
    "(widgets.basecolor?1:0)+(widgets.normal?1:0)+(widgets.roughness?1:0)+(widgets.metalness?1:0)+(widgets.height?1:0)"
)


async def _run_patina(cls: type[IO.ComfyNode], path: str, request) -> PatinaResult:
    submit = await sync_op(
        cls,
        ApiEndpoint(path=path, method="POST"),
        response_model=PatinaSubmitResponse,
        data=request,
    )
    await poll_op(
        cls,
        ApiEndpoint(path=f"{PATINA_ENDPOINT}/requests/{submit.request_id}/status"),
        response_model=PatinaStatusResponse,
        status_extractor=lambda r: r.status,
        poll_interval=3.0,
    )
    return await sync_op(
        cls,
        ApiEndpoint(path=f"{PATINA_ENDPOINT}/requests/{submit.request_id}"),
        response_model=PatinaResult,
    )


async def _download_rgb(cls: type[IO.ComfyNode], url: str) -> torch.Tensor:
    return bytesio_to_image_tensor(await download_url_as_bytesio(url, cls=cls), mode="RGB")


async def _map_outputs(cls: type[IO.ComfyNode], result: PatinaResult) -> tuple[torch.Tensor, ...]:
    by_type = {img.map_type: img for img in result.images if img.map_type}
    outputs = []
    for name in PATINA_MAPS:
        img = by_type.get(name)
        outputs.append(await _download_rgb(cls, img.url) if img else torch.zeros(1, 1, 1, 3))
    return tuple(outputs)


async def _base_texture(cls: type[IO.ComfyNode], result: PatinaResult) -> torch.Tensor:
    texture = next((img for img in result.images if not img.map_type), None)
    if texture is None:
        return torch.zeros(1, 1, 1, 3)
    return await _download_rgb(cls, texture.url)


def _selected_maps(basecolor: bool, normal: bool, roughness: bool, metalness: bool, height: bool) -> list[str]:
    flags = {
        "basecolor": basecolor,
        "normal": normal,
        "roughness": roughness,
        "metalness": metalness,
        "height": height,
    }
    return [m for m in PATINA_MAPS if flags[m]]


def _resolve_image_size(image_size: dict[str, Any]) -> str | ImageSize:
    key = image_size.get("image_size") if isinstance(image_size, dict) else None
    if key == "custom":
        return ImageSize(width=int(image_size["width"]), height=int(image_size["height"]))
    return _LABEL_TO_PRESET.get(key, "square_hd")


def _image_size_input() -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        "image_size",
        options=[IO.DynamicCombo.Option(label, []) for label, _, _, _ in _IMAGE_SIZES]
        + [
            IO.DynamicCombo.Option(
                "custom",
                [
                    IO.Int.Input("width", default=1024, min=512, max=2048, step=16),
                    IO.Int.Input("height", default=1024, min=512, max=2048, step=16),
                ],
            )
        ],
        tooltip="Output texture size. Choose 'custom' for a width/height between 512 and 2048 "
        "(an 8K result comes from 4x upscaling the maps).",
    )


def _map_toggle_inputs() -> list[IO.Boolean.Input]:
    return [
        IO.Boolean.Input("basecolor", default=True, tooltip="Generate the basecolor (albedo) map."),
        IO.Boolean.Input("normal", default=True, tooltip="Generate the normal map."),
        IO.Boolean.Input("roughness", default=False, tooltip="Generate the roughness map."),
        IO.Boolean.Input("metalness", default=False, tooltip="Generate the metalness map."),
        IO.Boolean.Input("height", default=False, tooltip="Generate the height/displacement map."),
    ]

class PatinaPBRMapsNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PatinaPBRMapsNode",
            display_name="Patina PBR Material Maps",
            category="partner/3d/fal",
            essentials_category="3D",
            description="Generate PBR material maps (basecolor, normal, roughness, metalness, height) "
            "from a texture image.",
            inputs=[
                IO.Image.Input("image", tooltip="Input photograph or render to derive PBR maps from."),
                *_map_toggle_inputs(),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483646,
                    control_after_generate=True,
                    tooltip="Seed for reproducible denoising.",
                ),
                IO.Boolean.Input("safety_checker", default=False, advanced=True),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Automatically downscale an input image whose longest side exceeds 2048px "
                            "(the model's input limit), preserving aspect ratio; smaller images are left as-is. "
                            "Disable to raise an error on oversized images instead.",
                ),
            ],
            outputs=[
                *[IO.Image.Output(m) for m in PATINA_MAPS],
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=list(PATINA_MAPS)),
                expr=f"""
                (
                  $n := {_NMAPS};
                  {{"type":"range_usd","min_usd": 0.0143 + 0.0143*$n, "max_usd": 0.0143 + 0.0572*$n, "format":{{"approximate":true}}}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        basecolor: bool = True,
        normal: bool = True,
        roughness: bool = False,
        metalness: bool = False,
        height: bool = False,
        seed: int = 42,
        safety_checker: bool = False,
        auto_downscale: bool = True,
    ) -> IO.NodeOutput:
        maps = _selected_maps(basecolor, normal, roughness, metalness, height)
        if not maps:
            raise ValueError("Enable at least one PBR map to generate.")
        if auto_downscale:
            image = downscale_image_tensor_by_max_side(image, max_side=2048)
        else:
            validate_image_dimensions(image, max_width=2048, max_height=2048)
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")
        result = await _run_patina(
            cls,
            PATINA_ENDPOINT,
            PatinaPBRMapsRequest(
                image_url=image_url,
                maps=maps,
                seed=seed,
                enable_safety_checker=safety_checker,
            ),
        )
        basecolor_t, normal_t, roughness_t, metalness_t, height_t = await _map_outputs(cls, result)
        return IO.NodeOutput(basecolor_t, normal_t, roughness_t, metalness_t, height_t)


class PatinaMaterialNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PatinaMaterialNode",
            display_name="Patina Material",
            category="partner/3d/fal",
            essentials_category="3D",
            description="Generate a complete seamlessly tiling PBR material (texture image + PBR maps) "
            "from a text prompt, up to 8K. Optionally drive it with an input image (img2img) "
            "or an image + mask (inpaint).",
            inputs=[
                IO.String.Input("prompt", multiline=True, tooltip="Describe the material/texture to generate."),
                _image_size_input(),
                *_map_toggle_inputs(),
                IO.Int.Input(
                    "upscale_factor",
                    default=0,
                    min=0,
                    max=4,
                    step=2,
                    tooltip="Seamless SeedVR upscaling of the PBR maps (the base texture is not upscaled).",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483646,
                    control_after_generate=True,
                    tooltip="Seed for reproducible generation.",
                ),
                IO.Combo.Input(
                    "tiling_mode",
                    options=["both", "horizontal", "vertical"],
                    default="both",
                    advanced=True,
                    tooltip="Tiling direction: omnidirectional, horizontal, or vertical.",
                ),
                IO.Int.Input(
                    "num_inference_steps",
                    default=8,
                    min=1,
                    max=8,
                    advanced=True,
                    tooltip="Denoising steps for texture generation.",
                ),
                IO.Int.Input(
                    "tile_size",
                    default=128,
                    min=32,
                    max=256,
                    step=2,
                    advanced=True,
                    tooltip="Tile size in latent space (64 = 512px, 128 = 1024px).",
                ),
                IO.Int.Input(
                    "tile_stride", default=64, min=16, max=128, step=2, advanced=True, tooltip="Tile stride in latent space."
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional source image. Provided alone = img2img; with mask = inpaint.",
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional inpaint mask (requires image). White = regenerate, black = keep.",
                ),
                IO.Float.Input(
                    "strength",
                    default=0.6,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="How much to transform the input image. Only used when an image is provided.",
                ),
                IO.Boolean.Input(
                    "prompt_expansion",
                    default=False,
                    advanced=True,
                    tooltip="Expand the prompt with an LLM for richer texture detail. Off by default: "
                    "expansion reframes the prompt as a photo and tends to wash out the metalness map.",
                ),
                IO.Boolean.Input("safety_checker", default=False, advanced=True),
            ],
            outputs=[
                IO.Image.Output("texture"),
                *[IO.Image.Output(m) for m in PATINA_MAPS],
                IO.String.Output("expanded_prompt"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["image_size", "image_size.width", "image_size.height", *PATINA_MAPS, "upscale_factor"]
                ),
                expr=f"""
                (
                  $mp := $ceil(widgets.image_size = "custom"
                         ? ($lookup(widgets, "image_size.width") * $lookup(widgets, "image_size.height")) / 1048576
                         : $lookup({_PRESET_MP}, widgets.image_size));
                  $n := {_NMAPS};
                  $up := widgets.upscale_factor = 4 ? 0.02288 : widgets.upscale_factor = 2 ? 0.00572 : 0;
                  {{"type":"usd","usd": 0.0143 + 0.0286*$mp + $mp*$n*(0.0143+$up)}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        image_size: dict[str, Any],
        basecolor: bool = True,
        normal: bool = True,
        roughness: bool = False,
        metalness: bool = False,
        height: bool = False,
        upscale_factor: int = 0,
        seed: int = 42,
        tiling_mode: str = "both",
        num_inference_steps: int = 8,
        tile_size: int = 128,
        tile_stride: int = 64,
        image: Input.Image | None = None,
        mask: Input.Mask | None = None,
        strength: float = 0.6,
        prompt_expansion: bool = False,
        safety_checker: bool = False,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)
        if mask is not None and image is None:
            raise ValueError("A mask requires an input image (inpaint mode).")
        image_url = None
        mask_url = None
        if image is not None:
            image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")
            if mask is not None:
                mask_url = await upload_image_to_comfyapi(
                    cls,
                    convert_mask_to_image(resize_mask_to_image(mask, image, allow_gradient=False)),
                    mime_type="image/png",
                    wait_label="Uploading mask",
                )
        result = await _run_patina(
            cls,
            f"{PATINA_ENDPOINT}/material",
            PatinaMaterialRequest(
                prompt=prompt,
                image_size=_resolve_image_size(image_size),
                maps=_selected_maps(basecolor, normal, roughness, metalness, height),
                upscale_factor=upscale_factor,
                tiling_mode=tiling_mode,
                num_inference_steps=num_inference_steps,
                enable_prompt_expansion=prompt_expansion,
                enable_safety_checker=safety_checker,
                tile_size=tile_size,
                tile_stride=tile_stride,
                image_url=image_url,
                mask_url=mask_url,
                strength=strength,
                seed=seed,
            ),
        )
        texture = await _base_texture(cls, result)
        basecolor_t, normal_t, roughness_t, metalness_t, height_t = await _map_outputs(cls, result)
        return IO.NodeOutput(
            texture,
            basecolor_t,
            normal_t,
            roughness_t,
            metalness_t,
            height_t,
            result.prompt or prompt,
        )


class PatinaMaterialExtractNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="PatinaMaterialExtractNode",
            display_name="Patina Material Extract",
            category="partner/3d/fal",
            essentials_category="3D",
            description="Extract a seamlessly tiling PBR material (texture image + PBR maps) from a region of an "
            "input image, guided by a prompt.",
            inputs=[
                IO.Image.Input("image", tooltip="Image to extract a texture from."),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip='Describe which texture to extract from the image (e.g. "the wall").',
                ),
                _image_size_input(),
                *_map_toggle_inputs(),
                IO.Int.Input(
                    "upscale_factor",
                    default=0,
                    min=0,
                    max=4,
                    step=2,
                    tooltip="Seamless SeedVR upscaling of the PBR maps (the base texture is not upscaled).",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483646,
                    control_after_generate=True,
                    tooltip="Seed for reproducible generation.",
                ),
                IO.Float.Input(
                    "strength",
                    default=0.6,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="How much to transform the input image.",
                ),
                IO.Combo.Input(
                    "tiling_mode",
                    options=["both", "horizontal", "vertical"],
                    default="both",
                    advanced=True,
                    tooltip="Tiling direction: omnidirectional, horizontal, or vertical.",
                ),
                IO.Int.Input(
                    "num_inference_steps",
                    default=8,
                    min=1,
                    max=8,
                    advanced=True,
                    tooltip="Denoising steps for texture generation.",
                ),
                IO.Int.Input(
                    "tile_size",
                    default=128,
                    min=32,
                    max=256,
                    step=2,
                    advanced=True,
                    tooltip="Tile size in latent space (64 = 512px, 128 = 1024px).",
                ),
                IO.Int.Input(
                    "tile_stride", default=64, min=16, max=128, step=2, advanced=True, tooltip="Tile stride in latent space."
                ),
                IO.Boolean.Input(
                    "prompt_expansion",
                    default=False,
                    advanced=True,
                    tooltip="Expand the prompt with an LLM for richer texture detail. Off by default: "
                    "expansion reframes the prompt as a photo and tends to wash out the metalness map.",
                ),
                IO.Boolean.Input("safety_checker", default=False, advanced=True),
            ],
            outputs=[
                IO.Image.Output("texture"),
                *[IO.Image.Output(m) for m in PATINA_MAPS],
                IO.String.Output("expanded_prompt"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(
                    widgets=["image_size", "image_size.width", "image_size.height", *PATINA_MAPS, "upscale_factor"]
                ),
                expr=f"""
                (
                  $mp := $ceil(widgets.image_size = "custom"
                         ? ($lookup(widgets, "image_size.width") * $lookup(widgets, "image_size.height")) / 1048576
                         : $lookup({_PRESET_MP}, widgets.image_size));
                  $n := {_NMAPS};
                  $up := widgets.upscale_factor = 4 ? 0.02288 : widgets.upscale_factor = 2 ? 0.00572 : 0;
                  {{"type":"usd","usd": 0.143 + 0.0286*$mp + $mp*$n*(0.0143+$up)}}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        image_size: dict[str, Any],
        basecolor: bool = True,
        normal: bool = True,
        roughness: bool = False,
        metalness: bool = False,
        height: bool = False,
        upscale_factor: int = 0,
        seed: int = 42,
        strength: float = 0.6,
        tiling_mode: str = "both",
        num_inference_steps: int = 8,
        tile_size: int = 128,
        tile_stride: int = 64,
        prompt_expansion: bool = False,
        safety_checker: bool = False,
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=False, min_length=1)
        image_url = await upload_image_to_comfyapi(cls, image, mime_type="image/png")
        result = await _run_patina(
            cls,
            f"{PATINA_ENDPOINT}/material/extract",
            PatinaExtractRequest(
                prompt=prompt,
                image_url=image_url,
                image_size=_resolve_image_size(image_size),
                maps=_selected_maps(basecolor, normal, roughness, metalness, height),
                upscale_factor=upscale_factor,
                tiling_mode=tiling_mode,
                num_inference_steps=num_inference_steps,
                enable_prompt_expansion=prompt_expansion,
                enable_safety_checker=safety_checker,
                tile_size=tile_size,
                tile_stride=tile_stride,
                strength=strength,
                seed=seed,
            ),
        )
        texture = await _base_texture(cls, result)
        basecolor_t, normal_t, roughness_t, metalness_t, height_t = await _map_outputs(cls, result)
        return IO.NodeOutput(
            texture,
            basecolor_t,
            normal_t,
            roughness_t,
            metalness_t,
            height_t,
            result.prompt or prompt,
        )


class PatinaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PatinaPBRMapsNode,
            PatinaMaterialNode,
            PatinaMaterialExtractNode,
        ]


async def comfy_entrypoint() -> PatinaExtension:
    return PatinaExtension()
