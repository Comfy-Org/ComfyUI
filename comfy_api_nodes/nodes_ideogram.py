from io import BytesIO
from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension
from PIL import Image
import numpy as np
import torch
from comfy_api_nodes.apis.ideogram import (
    IdeogramGenerateResponse,
    IdeogramPImageRequest,
    IdeogramV3Request,
    IdeogramV3EditRequest,
    IdeogramV4Request,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_as_bytesio,
    resize_mask_to_image,
    sync_op,
    validate_string,
)


V3_RATIO_MAP = {
    "1:3":"1x3",
    "3:1":"3x1",
    "1:2":"1x2",
    "2:1":"2x1",
    "9:16":"9x16",
    "16:9":"16x9",
    "10:16":"10x16",
    "16:10":"16x10",
    "2:3":"2x3",
    "3:2":"3x2",
    "3:4":"3x4",
    "4:3":"4x3",
    "4:5":"4x5",
    "5:4":"5x4",
    "1:1":"1x1",
}

V3_RESOLUTIONS= [
    "Auto",
    "512x1536",
    "576x1408",
    "576x1472",
    "576x1536",
    "640x1344",
    "640x1408",
    "640x1472",
    "640x1536",
    "704x1152",
    "704x1216",
    "704x1280",
    "704x1344",
    "704x1408",
    "704x1472",
    "736x1312",
    "768x1088",
    "768x1216",
    "768x1280",
    "768x1344",
    "800x1280",
    "832x960",
    "832x1024",
    "832x1088",
    "832x1152",
    "832x1216",
    "832x1248",
    "864x1152",
    "896x960",
    "896x1024",
    "896x1088",
    "896x1120",
    "896x1152",
    "960x832",
    "960x896",
    "960x1024",
    "960x1088",
    "1024x832",
    "1024x896",
    "1024x960",
    "1024x1024",
    "1088x768",
    "1088x832",
    "1088x896",
    "1088x960",
    "1120x896",
    "1152x704",
    "1152x832",
    "1152x864",
    "1152x896",
    "1216x704",
    "1216x768",
    "1216x832",
    "1248x832",
    "1280x704",
    "1280x768",
    "1280x800",
    "1312x736",
    "1344x640",
    "1344x704",
    "1344x768",
    "1408x576",
    "1408x640",
    "1408x704",
    "1472x576",
    "1472x640",
    "1472x704",
    "1536x512",
    "1536x576",
    "1536x640"
]

async def download_and_process_images(image_urls):
    """Helper function to download and process multiple images from URLs"""

    # Initialize list to store image tensors
    image_tensors = []

    for image_url in image_urls:
        # Using functions from apinode_utils.py to handle downloading and processing
        image_bytesio = await download_url_as_bytesio(image_url)  # Download image content to BytesIO
        img_tensor = bytesio_to_image_tensor(image_bytesio, mode="RGB")  # Convert to torch.Tensor with RGB mode
        image_tensors.append(img_tensor)

    # Stack tensors to match (N, width, height, channels)
    if image_tensors:
        stacked_tensors = torch.cat(image_tensors, dim=0)
    else:
        raise Exception("No valid images were processed")

    return stacked_tensors


class IdeogramV3(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdeogramV3",
            display_name="Ideogram V3",
            category="partner/image/Ideogram",
            description="Generates images using the Ideogram V3 model. "
                        "Supports both regular image generation from text prompts and image editing with mask.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation or editing",
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
                    "aspect_ratio",
                    options=list(V3_RATIO_MAP.keys()),
                    default="1:1",
                    tooltip="The aspect ratio for image generation. Ignored if resolution is not set to Auto.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=V3_RESOLUTIONS,
                    default="Auto",
                    tooltip="The resolution for image generation. "
                            "If not set to Auto, this overrides the aspect_ratio setting.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "magic_prompt_option",
                    options=["AUTO", "ON", "OFF"],
                    default="AUTO",
                    tooltip="Determine if MagicPrompt should be used in generation",
                    optional=True,
                    advanced=True,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Int.Input(
                    "num_images",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Combo.Input(
                    "rendering_speed",
                    options=["DEFAULT", "TURBO", "QUALITY"],
                    default="DEFAULT",
                    tooltip="Controls the trade-off between generation speed and quality",
                    optional=True,
                    advanced=True,
                ),
                IO.Image.Input(
                    "character_image",
                    tooltip="Image to use as character reference.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "character_mask",
                    tooltip="Optional mask for character reference image.",
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
                depends_on=IO.PriceBadgeDepends(widgets=["rendering_speed", "num_images"], inputs=["character_image"]),
                expr="""
                (
                  $n := widgets.num_images;
                  $speed := widgets.rendering_speed;
                  $hasChar := inputs.character_image.connected;
                  $base :=
                    $contains($speed,"quality") ? ($hasChar ? 0.286 : 0.1287) :
                    $contains($speed,"default") ? ($hasChar ? 0.2145 : 0.0858) :
                    $contains($speed,"turbo") ? ($hasChar ? 0.143 : 0.0429) :
                    0.0858;
                  {"type":"usd","usd": $round($base * $n, 2)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        image=None,
        mask=None,
        resolution="Auto",
        aspect_ratio="1:1",
        magic_prompt_option="AUTO",
        seed=0,
        num_images=1,
        rendering_speed="DEFAULT",
        character_image=None,
        character_mask=None,
    ):
        if rendering_speed == "BALANCED":  # for backward compatibility
            rendering_speed = "DEFAULT"

        character_img_binary = None
        character_mask_binary = None

        if character_image is not None:
            input_tensor = character_image.squeeze().cpu()
            if character_mask is not None:
                character_mask = resize_mask_to_image(character_mask, character_image, allow_gradient=False)
                character_mask = 1.0 - character_mask
                if character_mask.shape[1:] != character_image.shape[1:-1]:
                    raise Exception("Character mask and image must be the same size")

                mask_np = (character_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_byte_arr = BytesIO()
                mask_img.save(mask_byte_arr, format="PNG")
                mask_byte_arr.seek(0)
                character_mask_binary = mask_byte_arr
                character_mask_binary.name = "mask.png"

            img_np = (input_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            character_img_binary = img_byte_arr
            character_img_binary.name = "image.png"
        elif character_mask is not None:
            raise Exception("Character mask requires character image to be present")

        # Check if both image and mask are provided for editing mode
        if image is not None and mask is not None:
            # Process image and mask
            input_tensor = image.squeeze().cpu()
            # Resize mask to match image dimension
            mask = resize_mask_to_image(mask, image, allow_gradient=False)
            # Invert mask, as Ideogram API will edit black areas instead of white areas (opposite of convention).
            mask = 1.0 - mask

            # Validate mask dimensions match image
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")

            # Process image
            img_np = (input_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr
            img_binary.name = "image.png"

            # Process mask - white areas will be replaced
            mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_byte_arr = BytesIO()
            mask_img.save(mask_byte_arr, format="PNG")
            mask_byte_arr.seek(0)
            mask_binary = mask_byte_arr
            mask_binary.name = "mask.png"

            # Create edit request
            edit_request = IdeogramV3EditRequest(
                prompt=prompt,
                rendering_speed=rendering_speed,
            )

            # Add optional parameters
            if magic_prompt_option != "AUTO":
                edit_request.magic_prompt = magic_prompt_option
            if seed != 0:
                edit_request.seed = seed
            if num_images > 1:
                edit_request.num_images = num_images

            files = {
                "image": img_binary,
                "mask": mask_binary,
            }
            if character_img_binary:
                files["character_reference_images"] = character_img_binary
            if character_mask_binary:
                files["character_mask_binary"] = character_mask_binary

            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/ideogram/ideogram-v3/edit", method="POST"),
                response_model=IdeogramGenerateResponse,
                data=edit_request,
                files=files,
                content_type="multipart/form-data",
                max_retries=1,
            )

        elif image is not None or mask is not None:
            # If only one of image or mask is provided, raise an error
            raise Exception("Ideogram V3 image editing requires both an image AND a mask")
        else:
            # Create generation request
            gen_request = IdeogramV3Request(
                prompt=prompt,
                rendering_speed=rendering_speed,
            )

            # Handle resolution vs aspect ratio
            if resolution != "Auto":
                gen_request.resolution = resolution
            elif aspect_ratio != "1:1":
                v3_aspect = V3_RATIO_MAP.get(aspect_ratio)
                if v3_aspect:
                    gen_request.aspect_ratio = v3_aspect

            # Add optional parameters
            if magic_prompt_option != "AUTO":
                gen_request.magic_prompt = magic_prompt_option
            if seed != 0:
                gen_request.seed = seed
            if num_images > 1:
                gen_request.num_images = num_images

            files = {}
            if character_img_binary:
                files["character_reference_images"] = character_img_binary
            if character_mask_binary:
                files["character_mask_binary"] = character_mask_binary
            if files:
                gen_request.style_type = "AUTO"

            response = await sync_op(
                cls,
                endpoint=ApiEndpoint(path="/proxy/ideogram/ideogram-v3/generate", method="POST"),
                response_model=IdeogramGenerateResponse,
                data=gen_request,
                files=files if files else None,
                content_type="multipart/form-data",
                max_retries=1,
            )

        if not response.data or len(response.data) == 0:
            raise Exception("No images were generated in the response")

        image_urls = [image_data.url for image_data in response.data if image_data.url]
        if not image_urls:
            raise Exception("No image URLs were generated in the response")
        return IO.NodeOutput(await download_and_process_images(image_urls))


class IdeogramV4(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdeogramV4",
            display_name="Ideogram V4",
            category="partner/image/Ideogram",
            description="Generates images using the Ideogram 4.0 model from a text prompt.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt for the image generation.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=[
                        "Auto",
                        "2048x2048 (1:1)",
                        "1440x2880 (1:2)",
                        "2880x1440 (2:1)",
                        "1664x2496 (2:3)",
                        "2496x1664 (3:2)",
                        "1792x2240 (4:5)",
                        "2240x1792 (5:4)",
                        "1440x2560 (9:16)",
                        "2560x1440 (16:9)",
                        "1600x2560 (5:8)",
                        "2560x1600 (8:5)",
                        "1728x2304 (3:4)",
                        "2304x1728 (4:3)",
                        "1296x3168 (9:22)",
                        "3168x1296 (22:9)",
                        "1152x2944 (9:23)",
                        "2944x1152 (23:9)",
                        "1248x3328 (3:8)",
                        "3328x1248 (8:3)",
                        "1280x3072 (5:12)",
                        "3072x1280 (12:5)",
                    ],
                    default="Auto",
                ),
                IO.Combo.Input(
                    "rendering_speed",
                    options=["DEFAULT", "TURBO", "QUALITY"],
                    default="DEFAULT",
                    tooltip="Controls the trade-off between generation speed and quality.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
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
                depends_on=IO.PriceBadgeDepends(widgets=["rendering_speed"]),
                expr="""
                (
                  $speed := widgets.rendering_speed;
                  $price :=
                    $contains($speed,"turbo") ? 0.0429 :
                    $contains($speed,"quality") ? 0.143 :
                    0.0858;
                  {"type":"usd","usd": $price}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        resolution: str,
        rendering_speed: str,
        seed: int,
    ):
        validate_string(prompt, strip_whitespace=True, min_length=1)
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/ideogram/ideogram-v4/generate", method="POST"),
            response_model=IdeogramGenerateResponse,
            data=IdeogramV4Request(
                text_prompt=prompt,
                resolution=resolution.split(" ")[0] if resolution != "Auto" else None,
                rendering_speed=rendering_speed,
            ),
            max_retries=1,
        )

        if not response.data or len(response.data) == 0:
            raise Exception("No images were generated in the response")
        image_urls = [image_data.url for image_data in response.data if image_data.url]
        if not image_urls:
            raise Exception("No image URLs were generated in the response")
        return IO.NodeOutput(await download_and_process_images(image_urls))


class IdeogramPImage(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdeogramPImage",
            display_name="Ideogram & Pruna P-Image",
            category="partner/image/Ideogram",
            description="Generates images using P-Image, Ideogram's fast text-to-image model. "
                        "Strong typography and photorealism; "
                        "supports Ideogram 4.0 structured JSON captions for exact text, "
                        "colors and layout.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Text prompt. Also accepts an Ideogram 4.0 structured JSON caption "
                            "(exact colors as #RRGGBB hexes, exact text strings, bounding-box "
                            "layout) — set prompt_upsampling to OFF to use it verbatim.",
                ),
                IO.Combo.Input(
                    "quality",
                    options=["VERY_LOW", "LOW", "MEDIUM", "HIGH"],
                    default="MEDIUM",
                    tooltip="Speed/price/quality tier. MEDIUM is the everyday default; HIGH for "
                            "complex prompts, fine detail and difficult text; VERY_LOW/LOW for "
                            "drafts at scale. Difficult text renders poorly below MEDIUM.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["1K", "2K"],
                    default="1K",
                    tooltip="Output size class (exact pixels follow the aspect ratio, e.g. "
                            "16:9 gives 1280x720 at 1K and 2560x1440 at 2K). "
                            "Prefer HIGH + 2K for crisp typography.",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=list(V3_RATIO_MAP.keys()),
                    default="1:1",
                    tooltip="The aspect ratio for image generation.",
                ),
                IO.Combo.Input(
                    "prompt_upsampling",
                    options=["AUTO", "ON", "OFF"],
                    default="AUTO",
                    tooltip="Expands short prompts into a detailed structured caption before "
                            "generation (the rewritten prompt is returned as final_prompt). "
                            "Set OFF when supplying your own JSON caption or exact wording.",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=2147483647,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                    tooltip="Seed for reproducible generation. With prompt_upsampling OFF, "
                            "the same seed and settings return the same image; with ON/AUTO "
                            "the prompt rewrite varies per run — reproduce a result by reusing "
                            "its final_prompt output with prompt_upsampling OFF and the same "
                            "seed.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(
                    "final_prompt",
                    tooltip="The prompt the image was actually generated from (the rewritten "
                            "structured caption when prompt_upsampling ran, else your prompt). "
                            "Feed it back with prompt_upsampling OFF and the same seed to "
                            "reproduce this image.",
                ),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["quality", "resolution"]),
                expr="""
                (
                  $q := widgets.quality;
                  $is2k := $contains(widgets.resolution, "2k");
                  $usd :=
                    $contains($q, "very_low") ? ($is2k ? 0.00858 : 0.00429) :
                    $contains($q, "high")     ? ($is2k ? 0.0429  : 0.02145) :
                    $contains($q, "medium")   ? ($is2k ? 0.0286  : 0.0143)  :
                                                ($is2k ? 0.02145 : 0.010725);
                  {"type": "usd", "usd": $usd}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        quality: str = "MEDIUM",
        resolution: str = "1K",
        aspect_ratio: str = "1:1",
        prompt_upsampling: str = "AUTO",
        seed: int = 42,
    ):
        validate_string(prompt, strip_whitespace=True, min_length=1)
        request = IdeogramPImageRequest(
            prompt=prompt,
            quality=quality,
            resolution=resolution,
            aspect_ratio=V3_RATIO_MAP[aspect_ratio],
            prompt_upsampling=prompt_upsampling,
            seed=seed,
        )
        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/ideogram/text-to-image/p-image-ideogram", method="POST"),
            response_model=IdeogramGenerateResponse,
            data=request,
            max_retries=1,
        )
        if not response.data:
            raise Exception("No images were generated in the response")
        image_urls = [image_data.url for image_data in response.data if image_data.url]
        if not image_urls:
            if any(image_data.is_image_safe is False for image_data in response.data):
                raise Exception(
                    "The generation was blocked by Ideogram's content safety filter. "
                    "Adjust the prompt and try again."
                )
            raise Exception("No image URLs were generated in the response")
        return IO.NodeOutput(
            await download_and_process_images(image_urls),
            response.data[0].prompt or prompt,
        )


class IdeogramExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            IdeogramV3,
            IdeogramV4,
            IdeogramPImage,
        ]


async def comfy_entrypoint() -> IdeogramExtension:
    return IdeogramExtension()
