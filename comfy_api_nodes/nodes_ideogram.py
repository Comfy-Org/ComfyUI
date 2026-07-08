from io import BytesIO
from typing_extensions import override
from comfy_api.latest import IO, ComfyExtension
from PIL import Image
import numpy as np
import torch
from comfy_api_nodes.apis.ideogram import (
    IdeogramGenerateRequest,
    IdeogramGenerateResponse,
    ImageRequest,
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

V1_V1_RES_MAP = {
  "Auto":"AUTO",
  "512 x 1536":"RESOLUTION_512_1536",
  "576 x 1408":"RESOLUTION_576_1408",
  "576 x 1472":"RESOLUTION_576_1472",
  "576 x 1536":"RESOLUTION_576_1536",
  "640 x 1024":"RESOLUTION_640_1024",
  "640 x 1344":"RESOLUTION_640_1344",
  "640 x 1408":"RESOLUTION_640_1408",
  "640 x 1472":"RESOLUTION_640_1472",
  "640 x 1536":"RESOLUTION_640_1536",
  "704 x 1152":"RESOLUTION_704_1152",
  "704 x 1216":"RESOLUTION_704_1216",
  "704 x 1280":"RESOLUTION_704_1280",
  "704 x 1344":"RESOLUTION_704_1344",
  "704 x 1408":"RESOLUTION_704_1408",
  "704 x 1472":"RESOLUTION_704_1472",
  "720 x 1280":"RESOLUTION_720_1280",
  "736 x 1312":"RESOLUTION_736_1312",
  "768 x 1024":"RESOLUTION_768_1024",
  "768 x 1088":"RESOLUTION_768_1088",
  "768 x 1152":"RESOLUTION_768_1152",
  "768 x 1216":"RESOLUTION_768_1216",
  "768 x 1232":"RESOLUTION_768_1232",
  "768 x 1280":"RESOLUTION_768_1280",
  "768 x 1344":"RESOLUTION_768_1344",
  "832 x 960":"RESOLUTION_832_960",
  "832 x 1024":"RESOLUTION_832_1024",
  "832 x 1088":"RESOLUTION_832_1088",
  "832 x 1152":"RESOLUTION_832_1152",
  "832 x 1216":"RESOLUTION_832_1216",
  "832 x 1248":"RESOLUTION_832_1248",
  "864 x 1152":"RESOLUTION_864_1152",
  "896 x 960":"RESOLUTION_896_960",
  "896 x 1024":"RESOLUTION_896_1024",
  "896 x 1088":"RESOLUTION_896_1088",
  "896 x 1120":"RESOLUTION_896_1120",
  "896 x 1152":"RESOLUTION_896_1152",
  "960 x 832":"RESOLUTION_960_832",
  "960 x 896":"RESOLUTION_960_896",
  "960 x 1024":"RESOLUTION_960_1024",
  "960 x 1088":"RESOLUTION_960_1088",
  "1024 x 640":"RESOLUTION_1024_640",
  "1024 x 768":"RESOLUTION_1024_768",
  "1024 x 832":"RESOLUTION_1024_832",
  "1024 x 896":"RESOLUTION_1024_896",
  "1024 x 960":"RESOLUTION_1024_960",
  "1024 x 1024":"RESOLUTION_1024_1024",
  "1088 x 768":"RESOLUTION_1088_768",
  "1088 x 832":"RESOLUTION_1088_832",
  "1088 x 896":"RESOLUTION_1088_896",
  "1088 x 960":"RESOLUTION_1088_960",
  "1120 x 896":"RESOLUTION_1120_896",
  "1152 x 704":"RESOLUTION_1152_704",
  "1152 x 768":"RESOLUTION_1152_768",
  "1152 x 832":"RESOLUTION_1152_832",
  "1152 x 864":"RESOLUTION_1152_864",
  "1152 x 896":"RESOLUTION_1152_896",
  "1216 x 704":"RESOLUTION_1216_704",
  "1216 x 768":"RESOLUTION_1216_768",
  "1216 x 832":"RESOLUTION_1216_832",
  "1232 x 768":"RESOLUTION_1232_768",
  "1248 x 832":"RESOLUTION_1248_832",
  "1280 x 704":"RESOLUTION_1280_704",
  "1280 x 720":"RESOLUTION_1280_720",
  "1280 x 768":"RESOLUTION_1280_768",
  "1280 x 800":"RESOLUTION_1280_800",
  "1312 x 736":"RESOLUTION_1312_736",
  "1344 x 640":"RESOLUTION_1344_640",
  "1344 x 704":"RESOLUTION_1344_704",
  "1344 x 768":"RESOLUTION_1344_768",
  "1408 x 576":"RESOLUTION_1408_576",
  "1408 x 640":"RESOLUTION_1408_640",
  "1408 x 704":"RESOLUTION_1408_704",
  "1472 x 576":"RESOLUTION_1472_576",
  "1472 x 640":"RESOLUTION_1472_640",
  "1472 x 704":"RESOLUTION_1472_704",
  "1536 x 512":"RESOLUTION_1536_512",
  "1536 x 576":"RESOLUTION_1536_576",
  "1536 x 640":"RESOLUTION_1536_640",
}

V1_V2_RATIO_MAP = {
  "1:1":"ASPECT_1_1",
  "4:3":"ASPECT_4_3",
  "3:4":"ASPECT_3_4",
  "16:9":"ASPECT_16_9",
  "9:16":"ASPECT_9_16",
  "2:1":"ASPECT_2_1",
  "1:2":"ASPECT_1_2",
  "3:2":"ASPECT_3_2",
  "2:3":"ASPECT_2_3",
  "4:5":"ASPECT_4_5",
  "5:4":"ASPECT_5_4",
}

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


class IdeogramV1(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdeogramV1",
            display_name="Ideogram V1",
            category="partner/image/Ideogram",
            description="Generates images using the Ideogram V1 model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "turbo",
                    default=False,
                    tooltip="Whether to use turbo mode (faster generation, potentially lower quality)",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=list(V1_V2_RATIO_MAP.keys()),
                    default="1:1",
                    tooltip="The aspect ratio for image generation.",
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
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Description of what to exclude from the image",
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
                depends_on=IO.PriceBadgeDepends(widgets=["num_images", "turbo"]),
                expr="""
                (
                  $n := widgets.num_images;
                  $base := (widgets.turbo = true) ? 0.0286 : 0.0858;
                  {"type":"usd","usd": $round($base * $n, 2)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        turbo=False,
        aspect_ratio="1:1",
        magic_prompt_option="AUTO",
        seed=0,
        negative_prompt="",
        num_images=1,
    ):
        # Determine the model based on turbo setting
        aspect_ratio = V1_V2_RATIO_MAP.get(aspect_ratio, None)
        model = "V_1_TURBO" if turbo else "V_1"

        response = await sync_op(
            cls,
            ApiEndpoint(path="/proxy/ideogram/generate", method="POST"),
            response_model=IdeogramGenerateResponse,
            data=IdeogramGenerateRequest(
                image_request=ImageRequest(
                    prompt=prompt,
                    model=model,
                    num_images=num_images,
                    seed=seed,
                    aspect_ratio=aspect_ratio if aspect_ratio != "ASPECT_1_1" else None,
                    magic_prompt_option=(magic_prompt_option if magic_prompt_option != "AUTO" else None),
                    negative_prompt=negative_prompt if negative_prompt else None,
                )
            ),
            max_retries=1,
        )

        if not response.data or len(response.data) == 0:
            raise Exception("No images were generated in the response")

        image_urls = [image_data.url for image_data in response.data if image_data.url]
        if not image_urls:
            raise Exception("No image URLs were generated in the response")
        return IO.NodeOutput(await download_and_process_images(image_urls))


class IdeogramV2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IdeogramV2",
            display_name="Ideogram V2",
            category="partner/image/Ideogram",
            description="Generates images using the Ideogram V2 model.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Prompt for the image generation",
                ),
                IO.Boolean.Input(
                    "turbo",
                    default=False,
                    tooltip="Whether to use turbo mode (faster generation, potentially lower quality)",
                ),
                IO.Combo.Input(
                    "aspect_ratio",
                    options=list(V1_V2_RATIO_MAP.keys()),
                    default="1:1",
                    tooltip="The aspect ratio for image generation. Ignored if resolution is not set to AUTO.",
                    optional=True,
                ),
                IO.Combo.Input(
                    "resolution",
                    options=list(V1_V1_RES_MAP.keys()),
                    default="Auto",
                    tooltip="The resolution for image generation. "
                            "If not set to AUTO, this overrides the aspect_ratio setting.",
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
                IO.Combo.Input(
                    "style_type",
                    options=["AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"],
                    default="NONE",
                    tooltip="Style type for generation (V2 only)",
                    optional=True,
                    advanced=True,
                ),
                IO.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip="Description of what to exclude from the image",
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
                #"color_palette": (
                #    IO.STRING,
                #    {
                #        "multiline": False,
                #        "default": "",
                #        "tooltip": "Color palette preset name or hex colors with weights",
                #    },
                #),
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
                depends_on=IO.PriceBadgeDepends(widgets=["num_images", "turbo"]),
                expr="""
                (
                  $n := widgets.num_images;
                  $base := (widgets.turbo = true) ? 0.0715 : 0.1144;
                  {"type":"usd","usd": $round($base * $n, 2)}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt,
        turbo=False,
        aspect_ratio="1:1",
        resolution="Auto",
        magic_prompt_option="AUTO",
        seed=0,
        style_type="NONE",
        negative_prompt="",
        num_images=1,
        color_palette="",
    ):
        aspect_ratio = V1_V2_RATIO_MAP.get(aspect_ratio, None)
        resolution = V1_V1_RES_MAP.get(resolution, None)
        # Determine the model based on turbo setting
        model = "V_2_TURBO" if turbo else "V_2"

        # Handle resolution vs aspect_ratio logic
        # If resolution is not AUTO, it overrides aspect_ratio
        final_resolution = None
        final_aspect_ratio = None

        if resolution != "AUTO":
            final_resolution = resolution
        else:
            final_aspect_ratio = aspect_ratio if aspect_ratio != "ASPECT_1_1" else None

        response = await sync_op(
            cls,
            endpoint=ApiEndpoint(path="/proxy/ideogram/generate", method="POST"),
            response_model=IdeogramGenerateResponse,
            data=IdeogramGenerateRequest(
                image_request=ImageRequest(
                    prompt=prompt,
                    model=model,
                    num_images=num_images,
                    seed=seed,
                    aspect_ratio=final_aspect_ratio,
                    resolution=final_resolution,
                    magic_prompt_option=(magic_prompt_option if magic_prompt_option != "AUTO" else None),
                    style_type=style_type if style_type != "NONE" else None,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    color_palette=color_palette if color_palette else None,
                )
            ),
            max_retries=1,
        )
        if not response.data or len(response.data) == 0:
            raise Exception("No images were generated in the response")

        image_urls = [image_data.url for image_data in response.data if image_data.url]
        if not image_urls:
            raise Exception("No image URLs were generated in the response")
        return IO.NodeOutput(await download_and_process_images(image_urls))


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


class IdeogramExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            IdeogramV1,
            IdeogramV2,
            IdeogramV3,
            IdeogramV4,
        ]


async def comfy_entrypoint() -> IdeogramExtension:
    return IdeogramExtension()
