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
    BFLFluxVTORequest,
    BFLStatus,
    Flux2ProGenerateRequest,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    convert_mask_to_image,
    download_url_to_image_tensor,
    get_number_of_images,
    poll_op,
    resize_mask_to_image,
    sync_op,
    tensor_to_base64_string,
    validate_aspect_ratio_string,
    validate_image_dimensions,
    validate_string,
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

        def price_extractor(_r: BaseModel) -> float | None:
            return None if initial_response.cost is None else initial_response.cost / 100

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            price_extractor=price_extractor,
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

        def price_extractor(_r: BaseModel) -> float | None:
            return None if initial_response.cost is None else initial_response.cost / 100

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            price_extractor=price_extractor,
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

        def price_extractor(_r: BaseModel) -> float | None:
            return None if initial_response.cost is None else initial_response.cost / 100

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            price_extractor=price_extractor,
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

        def price_extractor(_r: BaseModel) -> float | None:
            return None if initial_response.cost is None else initial_response.cost / 100

        response = await poll_op(
            cls,
            ApiEndpoint(initial_response.polling_url),
            response_model=BFLFluxStatusResponse,
            status_extractor=lambda r: r.status,
            progress_extractor=lambda r: r.progress,
            price_extractor=price_extractor,
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
        ]


async def comfy_entrypoint() -> BFLExtension:
    return BFLExtension()
