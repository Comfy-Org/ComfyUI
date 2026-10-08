import av
import torch
from av.codec import CodecContext
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input, Types, VideoFromComponents
from comfy_api_nodes.apis.bria import (
    BriaAddObjectRequest,
    BriaEditImageRequest,
    BriaEraseByTextRequest,
    BriaEraseForegroundRequest,
    BriaEraseRequest,
    BriaExpandRequest,
    BriaExpandResponse,
    BriaFiboEditResponse,
    BriaGenFillRequest,
    BriaImageEditResponse,
    BriaImageResultResponse,
    BriaIncreaseResolutionRequest,
    BriaRelightRequest,
    BriaRemoveBackgroundRequest,
    BriaRemoveBackgroundResponse,
    BriaRemoveVideoBackgroundRequest,
    BriaRemoveVideoBackgroundResponse,
    BriaReplaceBackgroundRequest,
    BriaReplaceBackgroundResponse,
    BriaReplaceObjectRequest,
    BriaReseasonRequest,
    BriaRestoreRequest,
    BriaStatusResponse,
    BriaVideoEraseRequest,
    BriaVideoGreenScreenRequest,
    BriaVideoReplaceBackgroundRequest,
    InputModerationSettings,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    convert_mask_to_image,
    download_url_to_image_tensor,
    download_url_to_video_output,
    downscale_image_tensor_by_max_side,
    get_image_dimensions,
    poll_op,
    sync_op,
    upload_image_to_comfyapi,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_string,
    validate_video_duration,
)

BRIA_MAX_OUTPUT_SIDE = 8192
BRIA_MIN_RATIO = 0.5
BRIA_MAX_RATIO = 3.0
BRIA_MIN_SHORT_SIDE = 224


def _upscaled_output_side(height: int, width: int, multiplier: int) -> int:
    prescale = max(1.0, BRIA_MIN_SHORT_SIDE / min(height, width))
    return round(max(height, width) * prescale * multiplier)


def _smallest_output_side(height: int, width: int, multiplier: int) -> int:
    return round(max(height, width) / min(height, width) * BRIA_MIN_SHORT_SIDE * multiplier)


class BriaImageEditNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaImageEditNode",
            display_name="Bria FIBO Image Edit",
            category="partner/image/Bria",
            description="Edit images using Bria latest model",
            inputs=[
                IO.Combo.Input("model", options=["FIBO"]),
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Instruction to edit image",
                ),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                IO.String.Input(
                    "structured_prompt",
                    multiline=True,
                    default="",
                    tooltip="A string containing the structured edit prompt in JSON format. "
                    "Use this instead of usual prompt for precise, programmatic control.",
                ),
                IO.Int.Input(
                    "seed",
                    default=1,
                    min=1,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.Float.Input(
                    "guidance_scale",
                    default=3,
                    min=3,
                    max=5,
                    step=0.01,
                    display_mode=IO.NumberDisplay.number,
                    tooltip="Higher value makes the image follow the prompt more closely.",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=20,
                    max=50,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("prompt_content_moderation", default=False),
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="If omitted, the edit applies to the entire image.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(display_name="structured_prompt"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.04}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        model: str,
        image: Input.Image,
        prompt: str,
        negative_prompt: str,
        structured_prompt: str,
        seed: int,
        guidance_scale: float,
        steps: int,
        moderation: InputModerationSettings,
        mask: Input.Image | None = None,
    ) -> IO.NodeOutput:
        if not prompt and not structured_prompt:
            raise ValueError("One of prompt or structured_prompt is required to be non-empty.")
        mask_url = None
        if mask is not None:
            mask_url = await upload_image_to_comfyapi(cls, convert_mask_to_image(mask), wait_label="Uploading mask")
        response = await sync_op(
            cls,
            ApiEndpoint(path="proxy/bria/v2/image/edit", method="POST"),
            data=BriaEditImageRequest(
                instruction=prompt if prompt else None,
                structured_instruction=structured_prompt if structured_prompt else None,
                images=[await upload_image_to_comfyapi(cls, image, wait_label="Uploading image")],
                mask=mask_url,
                negative_prompt=negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale,
                seed=seed,
                model_version=model,
                steps_num=steps,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageEditResponse,
        )
        return IO.NodeOutput(
            await download_url_to_image_tensor(response.result.image_url),
            response.result.structured_prompt,
        )


class BriaRemoveImageBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveImageBackground",
            display_name="Bria Remove Image Background",
            category="partner/image/Bria",
            description="Remove the background from an image using Bria RMBG 2.0.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=True),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
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
                expr="""{"type":"usd","usd":0.018}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        moderation: dict,
        seed: int,
    ) -> IO.NodeOutput:
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/remove_background", method="POST"),
            data=BriaRemoveBackgroundRequest(
                image=await upload_image_to_comfyapi(cls, image, wait_label="Uploading image"),
                sync=False,
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


def _mask_to_binary_image(mask: Input.Image, action: str) -> torch.Tensor:
    binary = (mask > 0.5).float()
    if not binary.any():
        raise ValueError(
            f"The mask is empty, so there is nothing to {action}. Masks are binarized at 50%: "
            f"areas painted at less than half opacity are ignored."
        )
    return convert_mask_to_image(binary)


def _validate_mask_aspect_ratio(mask: Input.Image, width: int, height: int, subject: str = "image") -> None:
    mh, mw = mask.shape[-2], mask.shape[-1]
    if abs(width * mh - height * mw) > 0.01 * height * mw:
        raise ValueError(
            f"Mask must have the same aspect ratio as the {subject}: "
            f"{subject} is {width}x{height}, mask is {mw}x{mh}."
        )


class BriaGenFill(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaGenFill",
            display_name="Bria Generative Fill",
            category="partner/image/Bria",
            description="Generate objects or scenery inside a masked region of an image using Bria.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input(
                    "mask",
                    tooltip="White areas are filled with generated content, black areas are preserved. "
                    "The mask is binarized before sending, so partially painted areas count as white. "
                    "Must have the same aspect ratio as the image.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Description of what to generate inside the masked region.",
                ),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                IO.Boolean.Input(
                    "refine_prompt",
                    default=True,
                    tooltip="Automatically adjust the prompt for better results; "
                    "disable to use the prompt exactly as written.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=1,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("prompt_content_moderation", default=False),
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=False),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
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
                expr="""{"type":"usd","usd":0.0429}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        mask: Input.Image,
        prompt: str,
        negative_prompt: str,
        refine_prompt: bool,
        seed: int,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        validate_string(prompt, min_length=1)
        _validate_mask_aspect_ratio(mask, image.shape[2], image.shape[1])
        mask_image = _mask_to_binary_image(mask, "fill")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/gen_fill", method="POST"),
            data=BriaGenFillRequest(
                image=await upload_image_to_comfyapi(cls, image, total_pixels=None, wait_label="Uploading image"),
                mask=await upload_image_to_comfyapi(
                    cls, mask_image, total_pixels=None, wait_label="Uploading mask"
                ),
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                refine_prompt=refine_prompt,
                seed=seed,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageResultResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaEraser(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaEraser",
            display_name="Bria Eraser",
            category="partner/image/Bria",
            description="Remove objects or areas outlined by a mask from an image using Bria.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input(
                    "mask",
                    tooltip="White areas are erased, black areas are preserved. "
                    "The mask is binarized before sending, so partially painted areas count as white. "
                    "Must have the same aspect ratio as the image.",
                ),
                IO.Combo.Input(
                    "mask_type",
                    options=["manual", "automatic"],
                    tooltip="manual for hand-drawn or brush masks, "
                    "automatic for masks produced by segmentation models such as SAM.",
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=False),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
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
                expr="""{"type":"usd","usd":0.0286}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        mask: Input.Image,
        mask_type: str,
        moderation: dict,
    ) -> IO.NodeOutput:
        _validate_mask_aspect_ratio(mask, image.shape[2], image.shape[1])
        mask_image = _mask_to_binary_image(mask, "erase")
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/erase", method="POST"),
            data=BriaEraseRequest(
                image=await upload_image_to_comfyapi(cls, image, total_pixels=None, wait_label="Uploading image"),
                mask=await upload_image_to_comfyapi(
                    cls, mask_image, total_pixels=None, wait_label="Uploading mask"
                ),
                mask_type=mask_type,
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageResultResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaExpandImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaExpandImage",
            display_name="Bria Expand Image",
            category="partner/image/Bria",
            description="Expand an image beyond its borders with generated content using Bria.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "expand_mode",
                    options=[
                        *[IO.DynamicCombo.Option(ratio, []) for ratio in
                          ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"]],
                        IO.DynamicCombo.Option(
                            "custom_ratio",
                            [
                                IO.Int.Input(
                                    "ratio_width",
                                    default=21,
                                    min=1,
                                    max=100,
                                    tooltip="Width side of the target ratio: 21 and 9 give 21:9.",
                                ),
                                IO.Int.Input(
                                    "ratio_height",
                                    default=9,
                                    min=1,
                                    max=100,
                                    tooltip="Height side of the target ratio: 21 and 9 give 21:9. "
                                    f"Bria only accepts width/height between {BRIA_MIN_RATIO} and "
                                    f"{BRIA_MAX_RATIO}, so anything taller than 1:2 needs the manual mode.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "manual",
                            [
                                IO.Int.Input("canvas_width", default=1000, min=64, max=5000),
                                IO.Int.Input("canvas_height", default=1000, min=64, max=5000),
                                IO.Int.Input(
                                    "image_width",
                                    default=500,
                                    min=1,
                                    max=5000,
                                    tooltip="Width of the original image inside the canvas.",
                                ),
                                IO.Int.Input(
                                    "image_height",
                                    default=500,
                                    min=1,
                                    max=5000,
                                    tooltip="Height of the original image inside the canvas.",
                                ),
                                IO.Int.Input(
                                    "image_x",
                                    default=250,
                                    min=-5000,
                                    max=5000,
                                    tooltip="X position of the image's top-left corner inside the canvas; "
                                    "may fall outside the canvas, cropping the image.",
                                ),
                                IO.Int.Input(
                                    "image_y",
                                    default=250,
                                    min=-5000,
                                    max=5000,
                                    tooltip="Y position of the image's top-left corner inside the canvas; "
                                    "may fall outside the canvas, cropping the image.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Target shape of the expanded image: a preset aspect ratio, a custom ratio, "
                    "or manual placement of the original image on a canvas. "
                    "Manual is the only mode that can reach a canvas taller than 1:2.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional description of the expanded scene; "
                    "when empty, Bria generates one from the image.",
                ),
                IO.String.Input("negative_prompt", multiline=True, default=""),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=1,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("prompt_content_moderation", default=False),
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=False),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(display_name="prompt", tooltip="The prompt used for the expansion; "
                                 "auto-generated by Bria when the prompt input is empty."),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.0286}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        expand_mode: dict,
        prompt: str,
        negative_prompt: str,
        seed: int,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        mode = expand_mode["expand_mode"]
        aspect_ratio = canvas_size = original_image_size = original_image_location = None
        if mode == "manual":
            canvas_size = [expand_mode["canvas_width"], expand_mode["canvas_height"]]
            original_image_size = [expand_mode["image_width"], expand_mode["image_height"]]
            original_image_location = [expand_mode["image_x"], expand_mode["image_y"]]
        elif mode == "custom_ratio":
            ratio_width, ratio_height = expand_mode["ratio_width"], expand_mode["ratio_height"]
            aspect_ratio = ratio_width / ratio_height
            if not BRIA_MIN_RATIO <= aspect_ratio <= BRIA_MAX_RATIO:
                raise ValueError(
                    f"Bria accepts a width-to-height ratio between {BRIA_MIN_RATIO} and {BRIA_MAX_RATIO}: "
                    f"{ratio_width}:{ratio_height} is {aspect_ratio:.4f}. "
                    f"Use the manual expand mode to reach a canvas of any shape."
                )
        else:
            aspect_ratio = mode
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/expand", method="POST"),
            data=BriaExpandRequest(
                image=await upload_image_to_comfyapi(cls, image, total_pixels=None, wait_label="Uploading image"),
                aspect_ratio=aspect_ratio,
                canvas_size=canvas_size,
                original_image_size=original_image_size,
                original_image_location=original_image_location,
                prompt=prompt if prompt else None,
                negative_prompt=negative_prompt if negative_prompt else None,
                seed=seed,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaExpandResponse,
        )
        return IO.NodeOutput(
            await download_url_to_image_tensor(response.result.image_url),
            response.result.prompt or "",
        )


class BriaIncreaseResolution(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaIncreaseResolution",
            display_name="Bria Increase Resolution",
            category="partner/image/Bria",
            description="Upscale an image by 2x or 4x using Bria, preserving the original content.",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "desired_increase",
                    options=["2", "4"],
                    tooltip="Resolution multiplier. The output must fit within 8192 pixels on each side.",
                ),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=False,
                    tooltip="Automatically lower the multiplier, and downscale the input image if that is "
                    "still not enough, when the output would exceed the limit.",
                ),
                IO.DynamicCombo.Input(
                    "moderation",
                    options=[
                        IO.DynamicCombo.Option("false", []),
                        IO.DynamicCombo.Option(
                            "true",
                            [
                                IO.Boolean.Input("visual_input_moderation", default=False),
                                IO.Boolean.Input("visual_output_moderation", default=False),
                            ],
                        ),
                    ],
                    tooltip="Moderation settings",
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
                expr="""{"type":"usd","usd":0.0286}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        desired_increase: str,
        auto_downscale: bool,
        moderation: dict,
    ) -> IO.NodeOutput:
        multiplier = int(desired_increase)
        height, width = get_image_dimensions(image)
        if _upscaled_output_side(height, width, multiplier) > BRIA_MAX_OUTPUT_SIDE:
            candidates = [c for c in (4, 2) if c <= multiplier]
            if not auto_downscale:
                predicted = _upscaled_output_side(height, width, multiplier)
                raise ValueError(
                    f"Bria can upscale up to a maximum output dimension of {BRIA_MAX_OUTPUT_SIDE} pixels: "
                    f"input is {width}x{height}, x{multiplier} would be {predicted} pixels on the long side. "
                    f"Enable auto_downscale, or use a smaller input image or a lower multiplier."
                )
            fitted = next(
                (c for c in candidates if _upscaled_output_side(height, width, c) <= BRIA_MAX_OUTPUT_SIDE), None
            )
            if fitted is not None:
                multiplier = fitted
            else:
                shrinkable = next((c for c in sorted(candidates) if _smallest_output_side(height, width, c)
                                   <= BRIA_MAX_OUTPUT_SIDE), None)
                if shrinkable is None:
                    raise ValueError(
                        f"This image cannot be upscaled by Bria at any multiplier: it is {width}x{height}, and "
                        f"Bria first enlarges the short side to {BRIA_MIN_SHORT_SIDE} pixels, which pushes the "
                        f"long side past the {BRIA_MAX_OUTPUT_SIDE} pixel limit. Crop it to a squarer shape first."
                    )
                multiplier = shrinkable
                image = downscale_image_tensor_by_max_side(image, max_side=BRIA_MAX_OUTPUT_SIDE // multiplier)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/increase_resolution", method="POST"),
            data=BriaIncreaseResolutionRequest(
                image=await upload_image_to_comfyapi(cls, image, total_pixels=None, wait_label="Uploading image"),
                desired_increase=multiplier,
                visual_input_content_moderation=moderation.get("visual_input_moderation", False),
                visual_output_content_moderation=moderation.get("visual_output_moderation", False),
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaImageResultResponse,
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaRemoveVideoBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRemoveVideoBackground",
            display_name="Bria Remove Video Background",
            category="partner/video/Bria",
            description="Remove the background from a video using Bria. ",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input(
                    "background_color",
                    options=[
                        "Black",
                        "White",
                        "Gray",
                        "Red",
                        "Green",
                        "Blue",
                        "Yellow",
                        "Cyan",
                        "Magenta",
                        "Orange",
                    ],
                    tooltip="Background color for the output video.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.05,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        background_color: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/remove_background", method="POST"),
            data=BriaRemoveVideoBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_color=background_color,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaVideoGreenScreen(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaVideoGreenScreen",
            display_name="Bria Video Green Screen",
            category="partner/video/Bria",
            description="Replace a video's background with a solid chroma-key screen using Bria.",
            inputs=[
                IO.Video.Input("video"),
                IO.Combo.Input(
                    "green_shade",
                    options=["broadcast_green", "chroma_green", "blue_screen"],
                    tooltip="Solid chroma-key shade applied behind the foreground: "
                    "broadcast_green (#00B140), chroma_green (#00FF00), or blue_screen (#0000FF).",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.05,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        green_shade: str,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/green_screen", method="POST"),
            data=BriaVideoGreenScreenRequest(
                video=await upload_video_to_comfyapi(cls, video),
                green_shade=green_shade,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaVideoReplaceBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaVideoReplaceBackground",
            display_name="Bria Video Replace Background",
            category="partner/video/Bria",
            description="Replace a video's background with a supplied image or video using Bria. "
            "The output keeps the foreground's resolution and frame rate; a background with a "
            "different aspect ratio is stretched to fit, so match it for undistorted results.",
            inputs=[
                IO.Video.Input("video", tooltip="Foreground video whose background is replaced."),
                IO.Image.Input(
                    "background_image",
                    optional=True,
                    tooltip="Background image to composite behind the foreground. "
                    "Provide either a background image or a background video, not both.",
                ),
                IO.Video.Input(
                    "background_video",
                    optional=True,
                    tooltip="Background video to composite behind the foreground. "
                    "Provide either a background image or a background video, not both.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.05,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        seed: int,
        background_image: Input.Image | None = None,
        background_video: Input.Video | None = None,
    ) -> IO.NodeOutput:
        if (background_image is None) == (background_video is None):
            raise ValueError("Provide either a background image or a background video, not both.")
        validate_video_duration(video, max_duration=60.0)
        if background_video is not None:
            validate_video_duration(background_video, max_duration=60.0)
            background_url = await upload_video_to_comfyapi(cls, background_video, wait_label="Uploading background")
        else:
            # Bria's replace_background 500s on RGBA, so drop the alpha channel before upload.
            background_url = await upload_image_to_comfyapi(
                cls, background_image[:, :, :, :3], wait_label="Uploading background"
            )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/replace_background", method="POST"),
            data=BriaVideoReplaceBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_url=background_url,
                output_container_and_codec="mp4_h264",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


def _video_to_images_and_mask(video: Input.Video) -> tuple[Input.Image, Input.Mask]:
    """Decode a transparent webm (VP9 + alpha) into image frames and an alpha mask.

    VP9 keeps its alpha in a side layer that PyAV's default vp9 decoder drops, so the frames
    are decoded with libvpx-vp9. Returns RGB images [B,H,W,3] in 0..1 and a mask [B,H,W]
    following the Load Image convention (1 = transparent) for compositing or Save WEBM.
    """
    rgb_frames: list[torch.Tensor] = []
    alpha_frames: list[torch.Tensor] = []
    with av.open(video.get_stream_source(), mode="r") as container:
        stream = container.streams.video[0]
        decoder = CodecContext.create("libvpx-vp9", "r") if stream.codec_context.name == "vp9" else None
        for packet in container.demux(stream):
            for frame in (decoder.decode(packet) if decoder is not None else packet.decode()):
                rgba = torch.from_numpy(frame.to_ndarray(format="rgba")).float() / 255.0
                rgb_frames.append(rgba[..., :3])
                alpha_frames.append(rgba[..., 3])
    images = torch.stack(rgb_frames) if rgb_frames else torch.zeros(0, 0, 0, 3)
    mask = (1.0 - torch.stack(alpha_frames)) if alpha_frames else torch.zeros((images.shape[0], 64, 64))
    return images, mask


class BriaTransparentVideoBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaTransparentVideoBackground",
            display_name="Bria Remove Video Background (Transparent)",
            category="partner/video/Bria",
            description="Remove the background from a video using Bria and return the cut-out frames "
            "plus an alpha mask. Connect both to a compositing node, or feed them to Save WEBM to "
            "write a transparent video.",
            inputs=[
                IO.Video.Input("video"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Seed controls whether the node should re-run; "
                    "results are non-deterministic regardless of seed.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.Mask.Output(display_name="mask"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr="""{"type":"usd","usd":0.05,"format":{"suffix":"/second"}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, max_duration=60.0)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/remove_background", method="POST"),
            data=BriaRemoveVideoBackgroundRequest(
                video=await upload_video_to_comfyapi(cls, video),
                background_color="Transparent",
                output_container_and_codec="webm_vp9",
                seed=seed,
            ),
            response_model=BriaStatusResponse,
        )
        response = await poll_op(
            cls,
            ApiEndpoint(path=f"/proxy/bria/v2/status/{response.request_id}"),
            status_extractor=lambda r: r.status,
            response_model=BriaRemoveVideoBackgroundResponse,
        )
        video_out = await download_url_to_video_output(response.result.video_url)
        images, mask = _video_to_images_and_mask(video_out)
        return IO.NodeOutput(images, mask)


BRIA_LIGHT_TYPES = [
    "midday",
    "blue hour light",
    "low-angle sunlight",
    "sunrise light",
    "spotlight on subject",
    "overcast light",
    "soft overcast daylight lighting",
    "cloud-filtered lighting",
    "fog-diffused lighting",
    "moonlight lighting",
    "starlight nighttime",
    "soft bokeh lighting",
    "harsh studio lighting",
]
BRIA_LIGHT_DIRECTIONS = ["front", "side", "bottom", "top-down"]
BRIA_SEASONS = ["spring", "summer", "autumn", "winter"]
BRIA_FAILED_STATUSES = ["error", "unknown"]
BRIA_MAX_REFERENCE_IMAGES = 10
BRIA_VIDEO_ERASE_MAX_DURATION = 5.1
BRIA_VIDEO_ERASE_MIN_FPS = 20
BRIA_VIDEO_ERASE_MAX_FPS = 30
BRIA_FIBO_EDIT_NOTE = (
    "Bria re-renders the whole frame at about 1 megapixel, so the result is not pixel-aligned with the input."
)


def _moderation_combo(*flags: str) -> IO.DynamicCombo.Input:
    return IO.DynamicCombo.Input(
        "moderation",
        options=[
            IO.DynamicCombo.Option("false", []),
            IO.DynamicCombo.Option("true", [IO.Boolean.Input(flag, default=False) for flag in flags]),
        ],
        tooltip="Moderation settings",
    )


def _visual_moderation(moderation: InputModerationSettings) -> tuple[bool, bool]:
    return (
        moderation.get("visual_input_moderation", False),
        moderation.get("visual_output_moderation", False),
    )


def _drop_alpha(image: Input.Image) -> torch.Tensor:
    return image[..., :3]


def _structured_prompt_output() -> IO.String.Output:
    return IO.String.Output(
        id="structured_prompt",
        display_name="structured_prompt",
        tooltip="Structured description of the edited image, for a follow-up edit with Bria FIBO Image Edit.",
    )


async def _poll_bria(cls: type[IO.ComfyNode], request_id: str, response_model: type[BaseModel]):
    return await poll_op(
        cls,
        ApiEndpoint(path=f"/proxy/bria/v2/status/{request_id}"),
        status_extractor=lambda r: r.status,
        failed_statuses=BRIA_FAILED_STATUSES,
        response_model=response_model,
    )


async def _run_fibo_edit(cls: type[IO.ComfyNode], path: str, data: BaseModel) -> IO.NodeOutput:
    submitted = await sync_op(
        cls,
        ApiEndpoint(path=path, method="POST"),
        data=data,
        response_model=BriaStatusResponse,
    )
    response = await _poll_bria(cls, submitted.request_id, BriaFiboEditResponse)
    return IO.NodeOutput(
        await download_url_to_image_tensor(response.result.image_url),
        response.result.structured_prompt or "",
    )


class BriaEraseByText(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaEraseByText",
            display_name="Bria Erase by Text",
            category="partner/image/Bria",
            description="Remove an object named in plain text from an image using Bria. " + BRIA_FIBO_EDIT_NOTE,
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "object_name",
                    default="",
                    tooltip="Name of the object to remove, such as 'the lamp'. Several objects can be named "
                    "at once, such as 'the phone and the pencils'. Naming something that is not in the "
                    "picture still returns, and bills, a re-rendered image.",
                ),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        object_name: str,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        validate_string(object_name, field_name="object_name", min_length=1)
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/erase_by_text",
            BriaEraseByTextRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                object_name=object_name,
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaAddObject(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaAddObject",
            display_name="Bria Add Object",
            category="partner/image/Bria",
            description="Insert an object described in plain text into an image using Bria. " + BRIA_FIBO_EDIT_NOTE,
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "instruction",
                    multiline=True,
                    default="",
                    tooltip="What to add and where, such as 'Place a red vase with flowers on the table'.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Bria takes no seed here and re-imagines the edit on each call, so repeated runs can differ. The value is never sent: it only changes this node's cache key, so that an otherwise identical graph runs the edit again instead of returning the cached result.",
                ),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        seed: int,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        validate_string(instruction, field_name="instruction", min_length=1)
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/add_object_by_text",
            BriaAddObjectRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                instruction=instruction,
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaReplaceObject(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaReplaceObject",
            display_name="Bria Replace Object",
            category="partner/image/Bria",
            description="Swap an object in an image for another one described in plain text using Bria. "
            + BRIA_FIBO_EDIT_NOTE,
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "instruction",
                    multiline=True,
                    default="",
                    tooltip="What to replace with what, such as 'Replace the red apple with a green pear'.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Bria takes no seed here and re-imagines the edit on each call, so repeated runs can differ. The value is never sent: it only changes this node's cache key, so that an otherwise identical graph runs the edit again instead of returning the cached result.",
                ),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        instruction: str,
        seed: int,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        validate_string(instruction, field_name="instruction", min_length=1)
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/replace_object_by_text",
            BriaReplaceObjectRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                instruction=instruction,
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaRelight(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRelight",
            display_name="Bria Relight",
            category="partner/image/Bria",
            description="Change the lighting atmosphere and direction of an image using Bria. "
            + BRIA_FIBO_EDIT_NOTE,
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "light_type",
                    options=BRIA_LIGHT_TYPES,
                    tooltip="Lighting atmosphere to apply.",
                ),
                IO.Combo.Input(
                    "light_direction",
                    options=BRIA_LIGHT_DIRECTIONS,
                    tooltip="Where the light comes from. Hard-light atmospheres such as midday, "
                    "spotlight on subject and harsh studio lighting react to it the most.",
                ),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        light_type: str,
        light_direction: str,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/relight",
            BriaRelightRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                light_type=light_type,
                light_direction=light_direction,
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaRestorePhoto(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaRestorePhoto",
            display_name="Bria Restore Photo",
            category="partner/image/Bria",
            description="Repair an old or damaged photograph with Bria: grain, scratches and blur go, and "
            "age-related color casts are neutralized. Card mounts and studio borders may be cropped away and "
            "faces are re-drawn. " + BRIA_FIBO_EDIT_NOTE + " Use Bria Increase Resolution to enlarge instead.",
            inputs=[
                IO.Image.Input("image"),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(cls, image: Input.Image, moderation: InputModerationSettings) -> IO.NodeOutput:
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/restore",
            BriaRestoreRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaReseason(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaReseason",
            display_name="Bria Reseason",
            category="partner/image/Bria",
            description="Move an image to another season with Bria. The whole scene is re-rendered, so "
            "scenery can change beyond the season itself. " + BRIA_FIBO_EDIT_NOTE,
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("season", options=BRIA_SEASONS, tooltip="Season to apply."),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output(), _structured_prompt_output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(cls, image: Input.Image, season: str, moderation: InputModerationSettings) -> IO.NodeOutput:
        visual_input, visual_output = _visual_moderation(moderation)
        return await _run_fibo_edit(
            cls,
            "/proxy/bria/v2/image/edit/reseason",
            BriaReseasonRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                season=season,
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
        )


class BriaEraseForeground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaEraseForeground",
            display_name="Bria Erase Foreground",
            category="partner/image/Bria",
            description="Remove the foreground of an image with Bria and generate a background in its place. "
            "Everything Bria reads as foreground goes, not only people. The untouched pixels are preserved, "
            "but the output is re-rendered at a standard size close to 1 megapixel.",
            inputs=[
                IO.Image.Input("image"),
                _moderation_combo("visual_input_moderation", "visual_output_moderation"),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(cls, image: Input.Image, moderation: InputModerationSettings) -> IO.NodeOutput:
        visual_input, visual_output = _visual_moderation(moderation)
        submitted = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/erase_foreground", method="POST"),
            data=BriaEraseForegroundRequest(
                image=await upload_image_to_comfyapi(cls, _drop_alpha(image), wait_label="Uploading image"),
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
            response_model=BriaStatusResponse,
        )
        response = await _poll_bria(cls, submitted.request_id, BriaImageResultResponse)
        return IO.NodeOutput(await download_url_to_image_tensor(response.result.image_url))


class BriaReplaceImageBackground(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaReplaceImageBackground",
            display_name="Bria Replace Image Background",
            category="partner/image/Bria",
            description="Replace the background of an image with Bria, described by a prompt or guided by "
            "reference images. The subject's pixels are preserved; the background is generated around them.",
            inputs=[
                IO.Image.Input("image"),
                IO.DynamicCombo.Input(
                    "background",
                    options=[
                        IO.DynamicCombo.Option(
                            "prompt",
                            [
                                IO.String.Input(
                                    "prompt",
                                    multiline=True,
                                    default="",
                                    tooltip="Description of the new background. A hex color code such as "
                                    "#FF5733 produces a solid color background.",
                                ),
                                IO.Combo.Input(
                                    "mode",
                                    options=["high_control", "base", "fast"],
                                    tooltip="high_control follows the prompt most closely, base is a "
                                    "balanced default and fast trades detail for speed.",
                                ),
                                IO.Boolean.Input(
                                    "refine_prompt",
                                    default=True,
                                    tooltip="Rewrite the prompt for better results, which also translates "
                                    "non-English prompts. Turn it off to send the prompt exactly as written.",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "reference images",
                            [
                                IO.Autogrow.Input(
                                    "ref_images",
                                    template=IO.Autogrow.TemplateNames(
                                        IO.Image.Input("ref_image"),
                                        names=[f"ref_image_{i}" for i in range(1, BRIA_MAX_REFERENCE_IMAGES + 1)],
                                        min=1,
                                    ),
                                    tooltip=f"1-{BRIA_MAX_REFERENCE_IMAGES} images guiding the new background; "
                                    "they do not need to share a size. Every reference changes the result, so "
                                    "a few consistent ones beat many conflicting ones. A batched input counts "
                                    "once per image.",
                                ),
                                IO.Boolean.Input(
                                    "enhance_ref_images",
                                    default=True,
                                    tooltip="Extra processing of the reference images for better results.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="Describe the new background with a prompt, or guide it with reference images.",
                ),
                IO.Boolean.Input(
                    "original_quality",
                    default=False,
                    tooltip="Return the input's exact pixel size instead of scaling the result to about "
                    "1 megapixel. A large input then returns a large image.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="The same seed usually returns the same background; the automatic prompt "
                    "refinement can still vary it.",
                ),
                _moderation_combo(
                    "prompt_content_moderation", "visual_input_moderation", "visual_output_moderation"
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(
                    id="refined_prompt",
                    display_name="refined_prompt",
                    tooltip="The prompt Bria generated from, empty on the reference-image path.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.0572}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        background: dict,
        original_quality: bool,
        seed: int,
        moderation: InputModerationSettings,
    ) -> IO.NodeOutput:
        prompt = None
        ref_images = None
        mode = None
        refine_prompt = None
        enhance_ref_images = None
        if background["background"] == "prompt":
            prompt = background["prompt"]
            validate_string(prompt, field_name="prompt", min_length=1)
            mode = background["mode"]
            refine_prompt = background["refine_prompt"]
        else:
            references = [
                _drop_alpha(image)
                for images in background["ref_images"].values()
                if images is not None
                for image in images
            ]
            if len(references) > BRIA_MAX_REFERENCE_IMAGES:
                raise ValueError(
                    f"At most {BRIA_MAX_REFERENCE_IMAGES} reference images are supported, got {len(references)}."
                )
            ref_images = await upload_images_to_comfyapi(
                cls,
                references,
                max_images=BRIA_MAX_REFERENCE_IMAGES,
                wait_label="Uploading reference images",
            )
            enhance_ref_images = background["enhance_ref_images"]
        visual_input, visual_output = _visual_moderation(moderation)
        submitted = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/image/edit/replace_background", method="POST"),
            data=BriaReplaceBackgroundRequest(
                image=await upload_image_to_comfyapi(
                    cls,
                    _drop_alpha(image),
                    total_pixels=None if original_quality else 2048 * 2048,
                    wait_label="Uploading image",
                ),
                prompt=prompt,
                ref_images=ref_images,
                mode=mode,
                refine_prompt=refine_prompt,
                enhance_ref_images=enhance_ref_images,
                original_quality=original_quality,
                seed=seed,
                prompt_content_moderation=moderation.get("prompt_content_moderation", False),
                visual_input_content_moderation=visual_input,
                visual_output_content_moderation=visual_output,
            ),
            response_model=BriaStatusResponse,
        )
        response = await _poll_bria(cls, submitted.request_id, BriaReplaceBackgroundResponse)
        return IO.NodeOutput(
            await download_url_to_image_tensor(response.result.image_url),
            response.result.refined_prompt or "",
        )


class BriaVideoEraser(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BriaVideoEraser",
            display_name="Bria Video Eraser",
            category="partner/video/Bria",
            description="Erase whatever a per-frame mask covers from a video with Bria and fill the gap in. "
            "The mask must be white on what should go and black elsewhere. Bria accepts clips of at most "
            "5.1 seconds at 20 to 30 frames per second with even pixel dimensions; audio is kept by default. "
            "The returned clip can be a few frames shorter than the input.",
            inputs=[
                IO.Video.Input("video", tooltip="Clip to erase from."),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="One mask per frame of the video, white where the object to erase is. "
                    "Provide either a mask or a mask video, not both.",
                ),
                IO.Video.Input(
                    "mask_video",
                    optional=True,
                    tooltip="An already encoded mask video with the same dimensions and frame count as the "
                    "video. Provide either a mask or a mask video, not both.",
                ),
                IO.Boolean.Input("preserve_audio", default=True, tooltip="Keep the input's audio track."),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                expr='{"type":"usd","usd":0.06435,"format":{"suffix":"/second"}}',
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        preserve_audio: bool,
        mask: Input.Mask | None = None,
        mask_video: Input.Video | None = None,
    ) -> IO.NodeOutput:
        if mask is None and mask_video is None:
            raise ValueError("Connect either a mask or a mask video.")
        if mask is not None and mask_video is not None:
            raise ValueError("Provide either a mask or a mask video, not both.")
        validate_video_duration(video, max_duration=BRIA_VIDEO_ERASE_MAX_DURATION)
        width, height = video.get_dimensions()
        if width % 2 or height % 2:
            raise ValueError(
                f"Bria needs even pixel dimensions, but the video is {width}x{height}. Crop or scale it first."
            )
        frame_count = video.get_frame_count()
        frame_rate = video.get_frame_rate()
        if not BRIA_VIDEO_ERASE_MIN_FPS <= float(frame_rate) <= BRIA_VIDEO_ERASE_MAX_FPS:
            raise ValueError(
                f"Bria accepts {BRIA_VIDEO_ERASE_MIN_FPS} to {BRIA_VIDEO_ERASE_MAX_FPS} frames per second, "
                f"but the video runs at {float(frame_rate):.3f}. Re-time it with Get Video Components and "
                f"Create Video first."
            )
        if mask is not None:
            if mask.shape[0] != frame_count:
                raise ValueError(
                    f"The mask has {mask.shape[0]} frame(s) but the video has {frame_count}; "
                    f"Bria needs one mask frame per video frame."
                )
            _validate_mask_aspect_ratio(mask, width, height, "video")
            if mask.shape[-1] != width or mask.shape[-2] != height:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1).float(), size=(height, width), mode="nearest-exact"
                )[:, 0]
            mask_source = VideoFromComponents(
                Types.VideoComponents(images=_mask_to_binary_image(mask, "erase"), frame_rate=frame_rate)
            )
        else:
            mask_width, mask_height = mask_video.get_dimensions()
            mask_frame_count = mask_video.get_frame_count()
            if (mask_width, mask_height) != (width, height):
                raise ValueError(
                    f"The mask video must have the same dimensions as the video: video is {width}x{height}, "
                    f"mask video is {mask_width}x{mask_height}."
                )
            if mask_frame_count != frame_count:
                raise ValueError(
                    f"The mask video has {mask_frame_count} frames but the video has {frame_count}; "
                    f"Bria needs one mask frame per video frame."
                )
            mask_source = mask_video
        submitted = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bria/v2/video/edit/erase", method="POST"),
            data=BriaVideoEraseRequest(
                video=await upload_video_to_comfyapi(cls, video),
                mask=await upload_video_to_comfyapi(cls, mask_source, wait_label="Uploading mask"),
                preserve_audio=preserve_audio,
                output_container_and_codec="mp4_h264",
            ),
            response_model=BriaStatusResponse,
        )
        response = await _poll_bria(cls, submitted.request_id, BriaRemoveVideoBackgroundResponse)
        return IO.NodeOutput(await download_url_to_video_output(response.result.video_url))


class BriaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BriaImageEditNode,
            BriaRemoveImageBackground,
            BriaGenFill,
            BriaEraser,
            BriaExpandImage,
            BriaIncreaseResolution,
            BriaRemoveVideoBackground,
            BriaVideoGreenScreen,
            BriaVideoReplaceBackground,
            BriaTransparentVideoBackground,
            BriaEraseByText,
            BriaAddObject,
            BriaReplaceObject,
            BriaRelight,
            BriaRestorePhoto,
            BriaReseason,
            BriaEraseForeground,
            BriaReplaceImageBackground,
            BriaVideoEraser,
        ]


async def comfy_entrypoint() -> BriaExtension:
    return BriaExtension()
