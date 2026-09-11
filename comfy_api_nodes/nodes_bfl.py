import math

import torch
from pydantic import BaseModel
from typing_extensions import override

from comfy_api.latest import IO, ComfyExtension, Input
from comfy_api_nodes.apis.bfl import (
    BFLFluxEraseRequest,
    BFLFluxExpandImageRequest,
    BFLFluxFillImageRequest,
    BFLFluxKontextProGenerateRequest,
    BFLFluxProGenerateResponse,
    BFLFluxProUltraGenerateRequest,
    BFLFluxStatusResponse,
    BFLFluxVideoUpscaleRequest,
    BFLFluxVTORequest,
    BFLStatus,
    Flux2ProGenerateRequest,
    Flux3ImageToVideoRequest,
    Flux3TextToVideoRequest,
    Flux3VideoContinuationRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    convert_mask_to_image,
    download_url_to_image_tensor,
    downscale_video_to_max_pixels,
    download_url_to_video_output,
    get_number_of_images,
    poll_op,
    resize_mask_to_image,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    upload_video_to_comfyapi,
    validate_aspect_ratio_string,
    validate_image_dimensions,
    validate_string,
    validate_video_dimensions,
    validate_video_duration,
)


class FluxProUltraImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProUltraImageNode",
            display_name="Flux 1.1 [pro] Ultra Image",
            category="partner/image/BFL",
            description="Generates images using Flux Pro 1.1 Ultra via api based on prompt and resolution.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.String.Input(
                    "aspect_ratio",
                    default="16:9",
                    tooltip="Aspect ratio of image; must be between 1:4 and 4:1.",
                ),
                IO.Boolean.Input(
                    "raw",
                    default=False,
                    tooltip="When True, generate less processed, more natural-looking images.",
                ),
                IO.Image.Input(
                    "image_prompt",
                    optional=True,
                ),
                IO.Float.Input(
                    "image_prompt_strength",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Blend between the prompt and the image prompt.",
                    optional=True,
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
                expr="""{"type":"usd","usd":0.06}""",
            ),
        )

    @classmethod
    def validate_inputs(cls, aspect_ratio: str):
        validate_aspect_ratio_string(aspect_ratio, (1, 4), (4, 1))
        return True

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        prompt_upsampling: bool = False,
        raw: bool = False,
        seed: int = 0,
        image_prompt: Input.Image | None = None,
        image_prompt_strength: float = 0.1,
    ) -> IO.NodeOutput:
        if image_prompt is None:
            validate_string(prompt, strip_whitespace=False)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.1-ultra/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxProUltraGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                seed=seed,
                aspect_ratio=aspect_ratio,
                raw=raw,
                image_prompt=(image_prompt if image_prompt is None else tensor_to_base64_string(image_prompt)),
                image_prompt_strength=(None if image_prompt is None else round(image_prompt_strength, 2)),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxKontextProImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="partner/image/BFL",
            description="Edits images using Flux.1 Kontext [pro] via api based on prompt and aspect ratio.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation - specify what and how to edit.",
                ),
                IO.String.Input(
                    "aspect_ratio",
                    default="16:9",
                    tooltip="Aspect ratio of image; must be between 1:4 and 4:1.",
                ),
                IO.Float.Input(
                    "guidance",
                    default=3.0,
                    min=0.1,
                    max=99.0,
                    step=0.1,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=1,
                    max=150,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=1234,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. If active, automatically modifies the prompt for more creative generation, but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Image.Input(
                    "input_image",
                    optional=True,
                ),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    BFL_PATH = "/proxy/bfl/flux-kontext-pro/generate"
    NODE_ID = "FluxKontextProImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [pro] Image"

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        guidance: float,
        steps: int,
        input_image: Input.Image | None = None,
        seed=0,
        prompt_upsampling=False,
    ) -> IO.NodeOutput:
        validate_aspect_ratio_string(aspect_ratio, (1, 4), (4, 1))
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=cls.BFL_PATH, method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxKontextProGenerateRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                guidance=round(guidance, 1),
                steps=steps,
                seed=seed,
                aspect_ratio=aspect_ratio,
                input_image=(input_image if input_image is None else tensor_to_base64_string(input_image)),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxKontextMaxImageNode(FluxKontextProImageNode):

    DESCRIPTION = "Edits images using Flux.1 Kontext [max] via api based on prompt and aspect ratio."
    BFL_PATH = "/proxy/bfl/flux-kontext-max/generate"
    NODE_ID = "FluxKontextMaxImageNode"
    DISPLAY_NAME = "Flux.1 Kontext [max] Image"


class FluxProExpandNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProExpandNode",
            display_name="Flux.1 Expand Image",
            category="partner/image/BFL",
            description="Outpaints image based on prompt.",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Int.Input(
                    "top",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the top of the image",
                ),
                IO.Int.Input(
                    "bottom",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the bottom of the image",
                ),
                IO.Int.Input(
                    "left",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the left of the image",
                ),
                IO.Int.Input(
                    "right",
                    default=0,
                    min=0,
                    max=2048,
                    tooltip="Number of pixels to expand at the right of the image",
                ),
                IO.Float.Input(
                    "guidance",
                    default=60,
                    min=1.5,
                    max=100,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=15,
                    max=50,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
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
                expr="""{"type":"usd","usd":0.05}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        prompt: str,
        prompt_upsampling: bool,
        top: int,
        bottom: int,
        left: int,
        right: int,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.0-expand/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxExpandImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=tensor_to_base64_string(image),
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxProFillNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxProFillNode",
            display_name="Flux.1 Fill Image",
            category="partner/image/BFL",
            description="Inpaints image based on mask and prompt.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=False,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation, "
                    "but results are nondeterministic (same seed will not produce exactly the same result).",
                    advanced=True,
                ),
                IO.Float.Input(
                    "guidance",
                    default=60,
                    min=1.5,
                    max=100,
                    tooltip="Guidance strength for the image generation process",
                ),
                IO.Int.Input(
                    "steps",
                    default=50,
                    min=15,
                    max=50,
                    tooltip="Number of steps for the image generation process",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
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
                expr="""{"type":"usd","usd":0.05}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        mask: Input.Image,
        prompt: str,
        prompt_upsampling: bool,
        steps: int,
        guidance: float,
        seed=0,
    ) -> IO.NodeOutput:
        # prepare mask
        mask = resize_mask_to_image(mask, image)
        mask = tensor_to_base64_string(convert_mask_to_image(mask))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/flux-pro-1.0-fill/generate", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxFillImageRequest(
                prompt=prompt,
                prompt_upsampling=prompt_upsampling,
                steps=steps,
                guidance=guidance,
                seed=seed,
                image=tensor_to_base64_string(image[:, :, :, :3]),  # make sure image will have alpha channel removed
                mask=mask,
            ),
        )
        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxEraseNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxEraseNode",
            display_name="Flux Erase Image",
            category="partner/image/BFL",
            description="Removes the masked object from an image and reconstructs the background. "
            "Paint the mask over what you want to erase.",
            inputs=[
                IO.Image.Input("image"),
                IO.Mask.Input("mask", tooltip="White areas are removed; black areas are preserved."),
                IO.Int.Input(
                    "dilate_pixels",
                    default=10,
                    min=0,
                    max=25,
                    tooltip="Expands the mask boundaries to ensure clean coverage of the object's edges.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    optional=True,
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
                expr="""{"type":"range_usd","min_usd":0.03,"max_usd":0.06,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        image: Input.Image,
        mask: Input.Image,
        dilate_pixels: int = 10,
        seed: int = 0,
    ) -> IO.NodeOutput:
        validate_image_dimensions(image, min_width=256, min_height=256)
        mask = resize_mask_to_image(mask, image)
        mask = tensor_to_base64_string(convert_mask_to_image(mask))
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/v1/flux-tools/erase-v1", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxEraseRequest(
                image=tensor_to_base64_string(image[:, :, :, :3]),  # make sure image will have alpha channel removed
                mask=mask,
                dilate_pixels=dilate_pixels,
                seed=seed,
            ),
        )

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class FluxVTONode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxVTONode",
            display_name="Flux Virtual Try-On",
            category="partner/image/BFL",
            description="Virtual try-on: dresses the person in the provided garment.",
            inputs=[
                IO.Image.Input("person", tooltip="Image of the person to dress."),
                IO.Image.Input("garment", tooltip="Image of the garment to apply."),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional natural-language styling instruction (e.g. how the garment should fit).",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
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
                expr="""{"type":"range_usd","min_usd":0.0375,"max_usd":0.075,"format":{"approximate":true}}""",
            ),
        )

    @classmethod
    async def execute(
        cls,
        person: Input.Image,
        garment: Input.Image,
        prompt: str = "",
        seed: int = 0,
    ) -> IO.NodeOutput:
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/bfl/v1/flux-tools/vto-v1", method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=BFLFluxVTORequest(
                prompt=prompt,
                person=tensor_to_base64_string(person[:, :, :, :3]),
                garment=tensor_to_base64_string(garment[:, :, :, :3]),
                seed=seed,
            ),
        )

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class Flux2ProImageNode(IO.ComfyNode):

    NODE_ID = "Flux2ProImageNode"
    DISPLAY_NAME = "Flux.2 [pro] Image"
    API_ENDPOINT = "/proxy/bfl/flux-2-pro/generate"
    PRICE_BADGE_EXPR = """
    (
      $MP := 1024 * 1024;
      $outMP := $max([1, $floor(((widgets.width * widgets.height) + $MP - 1) / $MP)]);
      $outputCost := 0.03 + 0.015 * ($outMP - 1);
      inputs.images.connected
        ? {
            "type":"range_usd",
            "min_usd": $outputCost + 0.015,
            "max_usd": $outputCost + 0.12,
            "format": { "approximate": true }
          }
        : {"type":"usd","usd": $outputCost}
    )
    """

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="partner/image/BFL",
            description="Generates images synchronously based on prompt and resolution.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation or edit",
                ),
                IO.Int.Input(
                    "width",
                    default=1024,
                    min=256,
                    max=2048,
                    step=32,
                ),
                IO.Int.Input(
                    "height",
                    default=768,
                    min=256,
                    max=2048,
                    step=32,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Boolean.Input(
                    "prompt_upsampling",
                    default=True,
                    tooltip="Whether to perform upsampling on the prompt. "
                    "If active, automatically modifies the prompt for more creative generation.",
                    advanced=True,
                ),
                IO.Image.Input("images", optional=True, tooltip="Up to 9 images to be used as references."),
            ],
            outputs=[IO.Image.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["width", "height"], inputs=["images"]),
                expr=cls.PRICE_BADGE_EXPR,
            ),
            is_deprecated=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        prompt_upsampling: bool,
        images: Input.Image | None = None,
    ) -> IO.NodeOutput:
        reference_images = {}
        if images is not None:
            if get_number_of_images(images) > 9:
                raise ValueError("The current maximum number of supported images is 9.")
            for image_index in range(images.shape[0]):
                key_name = f"input_image_{image_index + 1}" if image_index else "input_image"
                reference_images[key_name] = tensor_to_base64_string(images[image_index], total_pixels=2048 * 2048)
        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=cls.API_ENDPOINT, method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=Flux2ProGenerateRequest(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                prompt_upsampling=prompt_upsampling,
                **reference_images,
            ),
        )

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


class Flux2MaxImageNode(Flux2ProImageNode):

    NODE_ID = "Flux2MaxImageNode"
    DISPLAY_NAME = "Flux.2 [max] Image"
    API_ENDPOINT = "/proxy/bfl/flux-2-max/generate"
    PRICE_BADGE_EXPR = """
    (
      $MP := 1024 * 1024;
      $outMP := $max([1, $floor(((widgets.width * widgets.height) + $MP - 1) / $MP)]);
      $outputCost := 0.07 + 0.03 * ($outMP - 1);

      inputs.images.connected
        ? {
            "type":"range_usd",
            "min_usd": $outputCost + 0.03,
            "max_usd": $outputCost + 0.24,
            "format": { "approximate": true }
          }
        : {"type":"usd","usd": $outputCost}
    )
    """


_FLUX2_MODEL_ENDPOINTS = {
    "Flux.2 [pro]": "/proxy/bfl/flux-2-pro/generate",
    "Flux.2 [max]": "/proxy/bfl/flux-2-max/generate",
}


def _flux2_model_inputs():
    return [
        IO.Int.Input(
            "width",
            default=1024,
            min=256,
            max=2048,
            step=32,
        ),
        IO.Int.Input(
            "height",
            default=768,
            min=256,
            max=2048,
            step=32,
        ),
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, 9)],
                min=0,
            ),
            tooltip="Optional reference image(s) for image-to-image generation. Up to 8 images.",
        ),
    ]


class Flux2ImageNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Flux2ImageNode",
            display_name="Flux.2 Image",
            category="partner/image/BFL",
            description="Generate images via Flux.2 [pro] or Flux.2 [max] from a prompt and optional reference images.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation or edit",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option("Flux.2 [pro]", _flux2_model_inputs()),
                        IO.DynamicCombo.Option("Flux.2 [max]", _flux2_model_inputs()),
                    ],
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
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
                    widgets=["model", "model.width", "model.height"],
                    input_groups=["model.images"],
                ),
                expr="""
                (
                  $isMax := widgets.model = "flux.2 [max]";
                  $MP := 1024 * 1024;
                  $w := $lookup(widgets, "model.width");
                  $h := $lookup(widgets, "model.height");
                  $outMP := $max([1, $floor((($w * $h) + $MP - 1) / $MP)]);
                  $outputCost := $isMax
                    ? (0.07 + 0.03 * ($outMP - 1))
                    : (0.03 + 0.015 * ($outMP - 1));
                  $refMin := $isMax ? 0.03 : 0.015;
                  $refMax := $isMax ? 0.24 : 0.12;
                  $hasRefs := $lookup(inputGroups, "model.images") > 0;
                  $hasRefs
                    ? {
                        "type": "range_usd",
                        "min_usd": $outputCost + $refMin,
                        "max_usd": $outputCost + $refMax,
                        "format": { "approximate": true }
                      }
                    : {"type": "usd", "usd": $outputCost}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: dict,
        seed: int,
    ) -> IO.NodeOutput:
        model_choice = model["model"]
        endpoint = _FLUX2_MODEL_ENDPOINTS[model_choice]
        width = model["width"]
        height = model["height"]
        images_dict = model.get("images") or {}

        image_tensors: list[Input.Image] = [t for t in images_dict.values() if t is not None]
        n_images = sum(get_number_of_images(t) for t in image_tensors)
        if n_images > 8:
            raise ValueError("The current maximum number of supported images is 8.")

        flat_tensors: list[torch.Tensor] = []
        for tensor in image_tensors:
            if len(tensor.shape) == 4:
                flat_tensors.extend(tensor[i] for i in range(tensor.shape[0]))
            else:
                flat_tensors.append(tensor)

        reference_images: dict[str, str] = {}
        for idx, tensor in enumerate(flat_tensors):
            key_name = f"input_image_{idx + 1}" if idx else "input_image"
            reference_images[key_name] = tensor_to_base64_string(tensor, total_pixels=2048 * 2048)

        initial_response = await sync_op(
            cls,
            ApiEndpoint(path=endpoint, method="POST"),
            response_model=BFLFluxProGenerateResponse,
            data=Flux2ProGenerateRequest(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
                **reference_images,
            ),
        )

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            completed_statuses=[BFLStatus.ready],
            failed_statuses=[
                BFLStatus.request_moderated,
                BFLStatus.content_moderated,
                BFLStatus.error,
                BFLStatus.task_not_found,
            ],
            queued_statuses=[],
        )
        return IO.NodeOutput(await download_url_to_image_tensor(response.result["sample"]))


_FLUX3_ASPECT_RATIOS = ["auto", "21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"]
_FLUX3_MIN_DURATION = 5
_FLUX3_MAX_DURATION = 20
_FLUX3_DURATIONS = ["auto"] + [str(i) for i in range(_FLUX3_MIN_DURATION, _FLUX3_MAX_DURATION + 1)]
_FLUX3_RESOLUTIONS = {"720p": "hd", "1080p": "fhd"}
_FLUX3_MAX_IMAGES = 10
_FLUX3_MIN_IMAGE_SIDE = 256
_FLUX3_MAX_IMAGE_ASPECT = 64


def _flux3_validate_image(image: torch.Tensor) -> None:
    validate_image_dimensions(image, min_width=_FLUX3_MIN_IMAGE_SIDE, min_height=_FLUX3_MIN_IMAGE_SIDE)
    height, width = image.shape[-3], image.shape[-2]
    if max(width, height) > _FLUX3_MAX_IMAGE_ASPECT * min(width, height):
        raise ValueError(
            f"Image aspect ratio is too extreme ({width}x{height}); "
            f"FLUX 3 accepts at most {_FLUX3_MAX_IMAGE_ASPECT}:1."
        )


def _flux3_collect_images(images: dict | None, field_name: str) -> list[torch.Tensor]:
    """Flatten Autogrow slots (each possibly batched) into single images and validate them."""
    flat: list[torch.Tensor] = []
    for tensor in (images or {}).values():
        if tensor is None:
            continue
        if tensor.ndim == 4:
            flat.extend(tensor[i] for i in range(tensor.shape[0]))
        else:
            flat.append(tensor)
    if len(flat) > _FLUX3_MAX_IMAGES:
        raise ValueError(f"FLUX 3 supports at most {_FLUX3_MAX_IMAGES} {field_name}, got {len(flat)}.")
    for tensor in flat:
        _flux3_validate_image(tensor)
    return flat


def _flux3_parse_times(value: str, image_count: int, duration: int | str) -> list[float]:
    """Parse one keyframe time in seconds per image: increasing, inside the clip."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != image_count:
        raise ValueError(
            f"Give one time per keyframe image: got {len(parts)} time(s) for {image_count} image(s)."
        )
    try:
        times = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Keyframe times must be numbers in seconds, comma-separated; got '{value}'.") from exc
    if not all(math.isfinite(time) for time in times):
        raise ValueError(f"Keyframe times must be finite numbers in seconds; got '{value}'.")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"Keyframe times must increase; got {times}.")
    if times[0] < 0:
        raise ValueError(f"Keyframe times cannot be negative; got {times[0]}.")
    cap = _FLUX3_MAX_DURATION if duration == "auto" else int(duration)
    if times[-1] > cap:
        raise ValueError(f"Keyframe time {times[-1]}s is past the end of a {cap}s clip.")
    return times


class Flux3VideoNodeBase(IO.ComfyNode):
    """Shared widgets, request plumbing and polling for the FLUX 3 generation modes."""

    RATE_HD: float
    RATE_FHD: float

    @classmethod
    def common_inputs(cls) -> list:
        return [
            IO.Combo.Input(
                "aspect_ratio",
                options=_FLUX3_ASPECT_RATIOS,
                default="auto",
                tooltip="Output aspect ratio. 'auto' picks one from the prompt and inputs.",
            ),
            IO.Combo.Input(
                "duration",
                options=_FLUX3_DURATIONS,
                default="auto",
                tooltip="Clip length in seconds. 'auto' fits the length to the content.",
            ),
            IO.Combo.Input(
                "resolution",
                options=list(_FLUX3_RESOLUTIONS),
                default="720p",
                tooltip="Output resolution.",
            ),
            IO.Boolean.Input(
                "generate_audio",
                default=True,
                tooltip="Generate synchronized audio (ambient, speech, effects). "
                "Off produces a video with no audio track.",
            ),
            IO.Int.Input(
                "safety_tolerance",
                default=2,
                min=0,
                max=4,
                advanced=True,
                tooltip="Moderation tolerance, 0 is the strictest. Requests that send images or "
                "video are capped at 2 whatever you set here.",
            ),
            IO.Int.Input(
                "seed",
                default=42,
                min=0,
                max=0xFFFFFFFF,
                control_after_generate=True,
                tooltip="Seed to determine if node should re-run; FLUX 3 picks its own seed, so "
                "actual results are nondeterministic regardless of this value.",
            ),
        ]

    @classmethod
    def common_fields(
        cls,
        prompt: str,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        generate_audio: bool,
        safety_tolerance: int,
    ) -> dict:
        validate_string(prompt, field_name="prompt", min_length=1)
        return {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration if duration == "auto" else int(duration),
            "resolution": _FLUX3_RESOLUTIONS[resolution],
            "generate_audio": generate_audio,
            "safety_tolerance": safety_tolerance,
        }

    @classmethod
    def price_badge(cls) -> IO.PriceBadge:
        return IO.PriceBadge(
            depends_on=IO.PriceBadgeDepends(widgets=["resolution", "duration"]),
            expr=f"""
                (
                  $rate := widgets.resolution = "1080p" ? {cls.RATE_FHD} : {cls.RATE_HD};
                  $type(widgets.duration) = "string" and widgets.duration != "auto"
                    ? {{"type":"usd","usd": $rate * $number(widgets.duration)}}
                    : {{"type":"usd","usd": $rate, "format": {{"suffix": "/second"}}}}
                )
                """,
        )


_FLUX3_VIDEO_ENDPOINT = ApiEndpoint(path="/proxy/bfl/v1/flux-3-video", method="POST")
_FLUX_VIDEO_UPSCALE_ENDPOINT = ApiEndpoint(path="/proxy/bfl/v1/flux-tools/video-upscale-v1", method="POST")
_BFL_POLL_PROXY_PATH = "/proxy/bfl/v1/get_result"


async def _bfl_video_execute(
    cls: type[IO.ComfyNode], endpoint: ApiEndpoint, request: BaseModel, poll_via_proxy: bool = False
) -> IO.NodeOutput:
    initial_response = await sync_op(cls, endpoint, response_model=BFLFluxProGenerateResponse, data=request)
    poll_endpoint = (
        ApiEndpoint(path=_BFL_POLL_PROXY_PATH, query_params={"polling_url": initial_response.polling_url})
        if poll_via_proxy
        else ApiEndpoint(initial_response.polling_url)
    )
    response = await poll_op(
        cls,
        poll_endpoint,
        response_model=BFLFluxStatusResponse,
        status_extractor=lambda r: r.status,
        progress_extractor=lambda r: r.progress,
        completed_statuses=[BFLStatus.ready],
        failed_statuses=[
            BFLStatus.request_moderated,
            BFLStatus.content_moderated,
            BFLStatus.error,
            BFLStatus.task_not_found,
        ],
        queued_statuses=[BFLStatus.pending],
        poll_interval=8.0,
        # a failed task answers the poll with a retryable-class HTTP 5xx (500 and 503 observed);
        # a small retry budget surfaces real failures quickly
        max_retries_per_poll=3,
    )
    return IO.NodeOutput(await download_url_to_video_output(response.result["sample"]))


class Flux3TextToVideoNode(Flux3VideoNodeBase):
    RATE_HD = 0.2431
    RATE_FHD = 0.4147

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Flux3TextToVideoNode",
            display_name="Flux 3 Text to Video",
            category="partner/video/BFL",
            description="Generates a video with synchronized audio from a text prompt via FLUX 3.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What you want, in plain language; the prompt is interpreted and expanded "
                    "before generation. Describe ambient sound, music and speech separately for layered audio.",
                ),
                *cls.common_inputs(),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=cls.price_badge(),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        generate_audio: bool,
        safety_tolerance: int,
        seed: int,
    ) -> IO.NodeOutput:
        request = Flux3TextToVideoRequest(
            **cls.common_fields(prompt, aspect_ratio, duration, resolution, generate_audio, safety_tolerance)
        )
        return await _bfl_video_execute(cls, _FLUX3_VIDEO_ENDPOINT, request)


class Flux3ImageToVideoNode(Flux3VideoNodeBase):
    RATE_HD = 0.2431
    RATE_FHD = 0.4147

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Flux3ImageToVideoNode",
            display_name="Flux 3 Image to Video",
            category="partner/video/BFL",
            description="Animates 1 to 10 images with FLUX 3. Each image becomes a frame of the clip: "
            "one image opens it, two morph from the first to the second, and more are spread across it "
            "or pinned to times you choose.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="How the scene should move and sound; the prompt is interpreted and "
                    "expanded before generation.",
                ),
                IO.Autogrow.Input(
                    "keyframes",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Image.Input("image", tooltip="Keyframe image."),
                        prefix="image_",
                        min=1,
                        max=_FLUX3_MAX_IMAGES,
                    ),
                    tooltip="1 to 10 images, in playback order. Minimum 256x256 pixels each.",
                ),
                IO.DynamicCombo.Input(
                    "placement",
                    options=[
                        IO.DynamicCombo.Option("spread across the clip", []),
                        IO.DynamicCombo.Option(
                            "at times",
                            [
                                IO.String.Input(
                                    "times",
                                    default="0",
                                    tooltip="One time in seconds per image, comma-separated and "
                                    "increasing, e.g. '0, 2.5, 5'.",
                                ),
                            ],
                        ),
                    ],
                    tooltip="'spread across the clip' lets FLUX 3 place the images (one opens the clip, "
                    "two become its start and end); 'at times' pins every image to a second you choose.",
                ),
                *cls.common_inputs(),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=cls.price_badge(),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        keyframes: IO.Autogrow.Type,
        placement: dict,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        generate_audio: bool,
        safety_tolerance: int,
        seed: int,
    ) -> IO.NodeOutput:
        fields = cls.common_fields(prompt, aspect_ratio, duration, resolution, generate_audio, safety_tolerance)
        images = _flux3_collect_images(keyframes, "keyframes")
        if not images:
            raise ValueError("Connect at least one keyframe image.")
        times = None
        if placement["placement"] == "at times":
            times = _flux3_parse_times(placement["times"], len(images), fields["duration"])
        elif len(images) >= 3 and fields["duration"] == "auto":
            # spread images land evenly between the first and last, which needs a known length
            raise ValueError(
                f"Spreading {len(images)} images across the clip needs an explicit duration: "
                "set duration, or place the images yourself with 'at times'."
            )
        urls = await upload_images_to_comfyapi(
            cls, images, max_images=_FLUX3_MAX_IMAGES, wait_label="Uploading keyframes"
        )
        request = Flux3ImageToVideoRequest(
            keyframes=list(zip(times, urls)) if times is not None else urls,
            **fields,
        )
        return await _bfl_video_execute(cls, _FLUX3_VIDEO_ENDPOINT, request)


class Flux3VideoContinuationNode(Flux3VideoNodeBase):
    RATE_HD = 0.5863
    RATE_FHD = 0.7579

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="Flux3VideoContinuationNode",
            display_name="Flux 3 Video Continuation",
            category="partner/video/BFL",
            description="Continues a video with FLUX 3: the new clip carries on from the final frames "
            "of the one you provide.",
            inputs=[
                IO.Video.Input("video", tooltip="The clip to continue."),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="What the continuation should show; the prompt is interpreted and expanded "
                    "before generation.",
                ),
                *cls.common_inputs(),
            ],
            outputs=[IO.Video.Output()],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=cls.price_badge(),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        prompt: str,
        aspect_ratio: str,
        duration: str,
        resolution: str,
        generate_audio: bool,
        safety_tolerance: int,
        seed: int,
    ) -> IO.NodeOutput:
        fields = cls.common_fields(prompt, aspect_ratio, duration, resolution, generate_audio, safety_tolerance)
        url = await upload_video_to_comfyapi(cls, video, wait_label="Uploading source video")
        request = Flux3VideoContinuationRequest(start_video=url, **fields)
        return await _bfl_video_execute(cls, _FLUX3_VIDEO_ENDPOINT, request)


_FLUX_VIDEO_UPSCALE_MODES = {"creative": 1, "precise": 0}
_FLUX_VIDEO_UPSCALE_MAX_INPUT_PIXELS = 3840 * 2160
_FLUX_VIDEO_UPSCALE_MAX_ASPECT_RATIO = 4.0


class FluxVideoUpscaleNode(IO.ComfyNode):

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FluxVideoUpscaleNode",
            display_name="Flux Video Upscale",
            category="partner/video/BFL",
            description="Upscales a video 1.5 to 3 times with FLUX super-resolution, either preserving "
            "the source precisely or creatively enhancing its detail.",
            inputs=[
                IO.Video.Input(
                    "video",
                    tooltip="Source clip of 1 to 20 seconds with an aspect ratio between 1:4 and 4:1. "
                    "The output is rendered at 24 fps and capped at about 14.4 megapixels per frame.",
                ),
                IO.Float.Input(
                    "upscale_factor",
                    default=2.0,
                    min=1.5,
                    max=3.0,
                    step=0.1,
                    tooltip="Output size relative to the source. Very large sources are upscaled by "
                    "less than the requested factor because of the per-frame cap.",
                ),
                IO.Combo.Input(
                    "mode",
                    options=list(_FLUX_VIDEO_UPSCALE_MODES),
                    default="creative",
                    tooltip="'creative' restores and invents fine detail, best for generated footage, "
                    "textures and scenery. 'precise' sharpens the source without changing it, "
                    "for faces, products and real footage.",
                ),
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Optional description of the clip that steers the enhanced detail. "
                    "Leave empty for a neutral upscale.",
                ),
                IO.Boolean.Input(
                    "auto_downscale",
                    default=True,
                    tooltip="Automatically downscale sources larger than 3840x2160 pixels in area to fit "
                    "the input limit. Aspect ratio is preserved; smaller videos are untouched.",
                ),
                IO.Int.Input(
                    "safety_tolerance",
                    default=2,
                    min=0,
                    max=4,
                    advanced=True,
                    tooltip="Moderation tolerance, 0 is the strictest.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed to determine if node should re-run; FLUX picks its own seed, so "
                    "actual results are nondeterministic regardless of this value.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["mode"]),
                expr="""
                    (
                      $precise := widgets.mode = "precise";
                      {"type":"range_usd",
                       "min_usd": $precise ? 0.212 : 0.297,
                       "max_usd": $precise ? 0.848 : 1.188,
                       "format": {"approximate": true, "suffix": "/s", "note": "(1080p-4K output)"}}
                    )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        video: Input.Video,
        upscale_factor: float,
        mode: str,
        prompt: str,
        auto_downscale: bool,
        safety_tolerance: int,
        seed: int,
    ) -> IO.NodeOutput:
        validate_video_duration(video, min_duration=1.0, max_duration=20.0)
        validate_video_dimensions(video, min_width=64, min_height=64)
        width, height = video.get_dimensions()
        if max(width, height) > _FLUX_VIDEO_UPSCALE_MAX_ASPECT_RATIO * min(width, height):
            raise ValueError(f"Video aspect ratio must be between 1:4 and 4:1, got {width}x{height}.")
        if auto_downscale:
            video = downscale_video_to_max_pixels(video, _FLUX_VIDEO_UPSCALE_MAX_INPUT_PIXELS)
        elif width * height > _FLUX_VIDEO_UPSCALE_MAX_INPUT_PIXELS:
            raise ValueError(
                f"Video must be at most 3840x2160 pixels in area, got {width}x{height}. "
                "Enable auto_downscale or use a smaller video."
            )
        url = await upload_video_to_comfyapi(cls, video, wait_label="Uploading source video")
        request = BFLFluxVideoUpscaleRequest(
            input_video=url,
            upscale_factor=round(upscale_factor, 1),
            creativity=_FLUX_VIDEO_UPSCALE_MODES[mode],
            prompt=prompt.strip() or None,
            safety_tolerance=safety_tolerance,
        )
        return await _bfl_video_execute(cls, _FLUX_VIDEO_UPSCALE_ENDPOINT, request, poll_via_proxy=True)


class BFLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FluxProUltraImageNode,
            FluxKontextProImageNode,
            FluxKontextMaxImageNode,
            FluxProExpandNode,
            FluxProFillNode,
            FluxEraseNode,
            FluxVTONode,
            Flux2ProImageNode,
            Flux2MaxImageNode,
            Flux2ImageNode,
            Flux3TextToVideoNode,
            Flux3ImageToVideoNode,
            Flux3VideoContinuationNode,
            FluxVideoUpscaleNode,
        ]


async def comfy_entrypoint() -> BFLExtension:
    return BFLExtension()
