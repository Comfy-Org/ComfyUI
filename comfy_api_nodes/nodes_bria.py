import av
import torch
from av.codec import CodecContext
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bria import (
    BriaEditImageRequest,
    BriaEraseRequest,
    BriaExpandRequest,
    BriaExpandResponse,
    BriaGenFillRequest,
    BriaImageEditResponse,
    BriaImageResultResponse,
    BriaIncreaseResolutionRequest,
    BriaRemoveBackgroundRequest,
    BriaRemoveBackgroundResponse,
    BriaRemoveVideoBackgroundRequest,
    BriaRemoveVideoBackgroundResponse,
    BriaStatusResponse,
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


def _validate_mask_aspect_ratio(image: Input.Image, mask: Input.Image) -> None:
    ih, iw = image.shape[1], image.shape[2]
    mh, mw = mask.shape[-2], mask.shape[-1]
    if abs(iw * mh - ih * mw) > 0.01 * ih * mw:
        raise ValueError(f"Mask must have the same aspect ratio as the image: image is {iw}x{ih}, mask is {mw}x{mh}.")


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
        _validate_mask_aspect_ratio(image, mask)
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
        _validate_mask_aspect_ratio(image, mask)
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
        ]


async def comfy_entrypoint() -> BriaExtension:
    return BriaExtension()
