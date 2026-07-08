from io import BytesIO

from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.reve import (
    ReveImageCreateRequest,
    ReveImageEditRequest,
    ReveImageRemixRequest,
    RevePostprocessingOperation,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    sync_op_raw,
    tensor_to_base64_string,
    validate_string,
)


def _build_postprocessing(upscale: dict, remove_background: bool) -> list[RevePostprocessingOperation] | None:
    ops = []
    if upscale["upscale"] == "enabled":
        ops.append(
            RevePostprocessingOperation(
                process="upscale",
                upscale_factor=upscale["upscale_factor"],
            )
        )
    if remove_background:
        ops.append(RevePostprocessingOperation(process="remove_background"))
    return ops or None


def _postprocessing_inputs():
    return [
        IO.DynamicCombo.Input(
            "upscale",
            options=[
                IO.DynamicCombo.Option("disabled", []),
                IO.DynamicCombo.Option(
                    "enabled",
                    [
                        IO.Int.Input(
                            "upscale_factor",
                            default=2,
                            min=2,
                            max=4,
                            step=1,
                            tooltip="Upscale factor (2x, 3x, or 4x).",
                        ),
                    ],
                ),
            ],
            tooltip="Upscale the generated image. May add additional cost.",
        ),
        IO.Boolean.Input(
            "remove_background",
            default=False,
            tooltip="Remove the background from the generated image. May add additional cost.",
        ),
    ]


def _reve_price_extractor(headers: dict) -> float | None:
    credits_used = headers.get("x-reve-credits-used")
    if credits_used is not None:
        return float(credits_used) / 524.48
    return None


def _reve_response_header_validator(headers: dict) -> None:
    error_code = headers.get("x-reve-error-code")
    if error_code:
        raise ValueError(f"Reve API error: {error_code}")
    if headers.get("x-reve-content-violation", "").lower() == "true":
        raise ValueError("The generated image was flagged for content policy violation.")


def _model_inputs(versions: list[str], aspect_ratios: list[str]):
    return [
        IO.DynamicCombo.Option(
            version,
            [
                IO.Combo.Input(
                    "aspect_ratio",
                    options=aspect_ratios,
                    tooltip="Aspect ratio of the output image.",
                ),
                IO.Int.Input(
                    "test_time_scaling",
                    default=1,
                    min=1,
                    max=5,
                    step=1,
                    tooltip="Higher values produce better images but cost more credits.",
                    advanced=True,
                ),
            ],
        )
        for version in versions
    ]


class ReveImageCreateNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ReveImageCreateNode",
            display_name="Reve Image Create",
            category="partner/image/Reve",
            description="Generate images from text descriptions using Reve.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the desired image. Maximum 2560 characters.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=_model_inputs(
                        ["reve-create@20250915"],
                        aspect_ratios=["3:2", "16:9", "9:16", "2:3", "4:3", "3:4", "1:1"],
                    ),
                    tooltip="Model version to use for generation.",
                ),
                *_postprocessing_inputs(),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
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
                    widgets=["upscale", "upscale.upscale_factor"],
                ),
                expr="""
                (
                    $factor := $lookup(widgets, "upscale.upscale_factor");
                    $fmt := {"approximate": true, "note": "(base)"};
                    widgets.upscale = "enabled" ? (
                        $factor = 4 ? {"type": "usd", "usd": 0.0762, "format": $fmt}
                        : $factor = 3 ? {"type": "usd", "usd": 0.0591, "format": $fmt}
                        : {"type": "usd", "usd": 0.0457, "format": $fmt}
                    ) : {"type": "usd", "usd": 0.03432, "format": $fmt}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        upscale: dict,
        remove_background: bool,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2560)
        response = await sync_op_raw(
            cls,
            ApiEndpoint(
                path="/proxy/reve/v1/image/create",
                method="POST",
                headers={"Accept": "image/webp"},
            ),
            as_binary=True,
            price_extractor=_reve_price_extractor,
            response_header_validator=_reve_response_header_validator,
            data=ReveImageCreateRequest(
                prompt=prompt,
                aspect_ratio=model["aspect_ratio"],
                version=model["model"],
                test_time_scaling=model["test_time_scaling"],
                postprocessing=_build_postprocessing(upscale, remove_background),
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(response)))


class ReveImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ReveImageEditNode",
            display_name="Reve Image Edit",
            category="partner/image/Reve",
            description="Edit images using natural language instructions with Reve.",
            inputs=[
                IO.Image.Input("image", tooltip="The image to edit."),
                IO.String.Input(
                    "edit_instruction",
                    multiline=True,
                    default="",
                    tooltip="Text description of how to edit the image. Maximum 2560 characters.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=_model_inputs(
                        ["reve-edit@20250915", "reve-edit-fast@20251030"],
                        aspect_ratios=["auto", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1"],
                    ),
                    tooltip="Model version to use for editing.",
                ),
                *_postprocessing_inputs(),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
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
                    widgets=["model", "upscale", "upscale.upscale_factor"],
                ),
                expr="""
                (
                    $fmt := {"approximate": true, "note": "(base)"};
                    $isFast := $contains(widgets.model, "fast");
                    $enabled := widgets.upscale = "enabled";
                    $factor := $lookup(widgets, "upscale.upscale_factor");
                    $isFast
                        ? {"type": "usd", "usd": 0.01001, "format": $fmt}
                        : $enabled ? (
                            $factor = 4 ? {"type": "usd", "usd": 0.0991, "format": $fmt}
                            : $factor = 3 ? {"type": "usd", "usd": 0.0819, "format": $fmt}
                            : {"type": "usd", "usd": 0.0686, "format": $fmt}
                        ) : {"type": "usd", "usd": 0.0572, "format": $fmt}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        edit_instruction: str,
        model: dict,
        upscale: dict,
        remove_background: bool,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(edit_instruction, min_length=1, max_length=2560)
        tts = model["test_time_scaling"]
        ar = model["aspect_ratio"]
        response = await sync_op_raw(
            cls,
            ApiEndpoint(
                path="/proxy/reve/v1/image/edit",
                method="POST",
                headers={"Accept": "image/webp"},
            ),
            as_binary=True,
            price_extractor=_reve_price_extractor,
            response_header_validator=_reve_response_header_validator,
            data=ReveImageEditRequest(
                edit_instruction=edit_instruction,
                reference_image=tensor_to_base64_string(image),
                aspect_ratio=ar if ar != "auto" else None,
                version=model["model"],
                test_time_scaling=tts if tts and tts > 1 else None,
                postprocessing=_build_postprocessing(upscale, remove_background),
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(response)))


class ReveImageRemixNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ReveImageRemixNode",
            display_name="Reve Image Remix",
            category="partner/image/Reve",
            description="Combine reference images with text prompts to create new images using Reve.",
            inputs=[
                IO.Autogrow.Input(
                    "reference_images",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image"),
                        prefix="image_",
                        min=1,
                        max=6,
                    ),
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text description of the desired image. "
                    "May include XML img tags to reference specific images by index, "
                    "e.g. <img>0</img>, <img>1</img>, etc.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=_model_inputs(
                        ["reve-remix@20250915", "reve-remix-fast@20251030"],
                        aspect_ratios=["auto", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "1:1"],
                    ),
                    tooltip="Model version to use for remixing.",
                ),
                *_postprocessing_inputs(),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
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
                    widgets=["model", "upscale", "upscale.upscale_factor"],
                ),
                expr="""
                (
                    $fmt := {"approximate": true, "note": "(base)"};
                    $isFast := $contains(widgets.model, "fast");
                    $enabled := widgets.upscale = "enabled";
                    $factor := $lookup(widgets, "upscale.upscale_factor");
                    $isFast
                        ? {"type": "usd", "usd": 0.01001, "format": $fmt}
                        : $enabled ? (
                            $factor = 4 ? {"type": "usd", "usd": 0.0991, "format": $fmt}
                            : $factor = 3 ? {"type": "usd", "usd": 0.0819, "format": $fmt}
                            : {"type": "usd", "usd": 0.0686, "format": $fmt}
                        ) : {"type": "usd", "usd": 0.0572, "format": $fmt}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        reference_images: IO.Autogrow.Type,
        prompt: str,
        model: dict,
        upscale: dict,
        remove_background: bool,
        seed: int,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1, max_length=2560)
        if not reference_images:
            raise ValueError("At least one reference image is required.")
        ref_base64_list = []
        for key in reference_images:
            ref_base64_list.append(tensor_to_base64_string(reference_images[key]))
        if len(ref_base64_list) > 6:
            raise ValueError("Maximum 6 reference images are allowed.")
        tts = model["test_time_scaling"]
        ar = model["aspect_ratio"]
        response = await sync_op_raw(
            cls,
            ApiEndpoint(
                path="/proxy/reve/v1/image/remix",
                method="POST",
                headers={"Accept": "image/webp"},
            ),
            as_binary=True,
            price_extractor=_reve_price_extractor,
            response_header_validator=_reve_response_header_validator,
            data=ReveImageRemixRequest(
                prompt=prompt,
                reference_images=ref_base64_list,
                aspect_ratio=ar if ar != "auto" else None,
                version=model["model"],
                test_time_scaling=tts if tts and tts > 1 else None,
                postprocessing=_build_postprocessing(upscale, remove_background),
            ),
        )
        return IO.NodeOutput(bytesio_to_image_tensor(BytesIO(response)))


class ReveExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ReveImageCreateNode,
            ReveImageEditNode,
            ReveImageRemixNode,
        ]


async def comfy_entrypoint() -> ReveExtension:
    return ReveExtension()
