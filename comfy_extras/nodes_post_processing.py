from typing_extensions import override
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math
from enum import Enum
from typing import TypedDict, Literal
import kornia

import comfy.utils
import comfy.model_management
from comfy_extras.nodes_latent import reshape_latent_to
import node_helpers
from comfy_api.latest import ComfyExtension, io
from nodes import MAX_RESOLUTION

class Blend(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageBlend",
            search_aliases=["mix images"],
            display_name="Blend Images",
            category="image/filters",
            essentials_category="Image Tools",
            inputs=[
                io.Image.Input("image1"),
                io.Image.Input("image2"),
                io.Float.Input("blend_factor", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("blend_mode", options=["normal", "multiply", "screen", "overlay", "soft_light", "difference"]),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str) -> io.NodeOutput:
        image1, image2 = node_helpers.image_alpha_fix(image1, image2)
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = cls.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return io.NodeOutput(blended_image)

    @classmethod
    def blend_mode(cls, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (cls.g(img1) - img1))
        elif mode == "difference":
            return img1 - img2
        raise ValueError(f"Unsupported blend mode: {mode}")

    @classmethod
    def g(cls, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

def gaussian_kernel(kernel_size: int, sigma: float, device=None, dtype=torch.float32):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device), torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return (g / g.sum()).to(dtype)

class Blur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageBlur",
            display_name="Blur Image",
            category="image/filters",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("blur_radius", default=1, min=1, max=31, step=1),
                io.Float.Input("sigma", default=1.0, min=0.1, max=10.0, step=0.1),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, blur_radius: int, sigma: float) -> io.NodeOutput:
        if blur_radius == 0:
            return io.NodeOutput(image)

        image = image.to(comfy.model_management.get_torch_device())
        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device, dtype=image.dtype).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        padded_image = F.pad(image, (blur_radius,blur_radius,blur_radius,blur_radius), 'reflect')
        blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:,:,blur_radius:-blur_radius, blur_radius:-blur_radius]
        blurred = blurred.permute(0, 2, 3, 1)

        return io.NodeOutput(blurred.to(comfy.model_management.intermediate_device()))


class Quantize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageQuantize",
            display_name="Quantize Image",
            category="image/filters",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("colors", default=256, min=1, max=256, step=1),
                io.Combo.Input("dither", options=["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"]),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @staticmethod
    def bayer(im, pal_im, order):
        def normalized_bayer_matrix(n):
            if n == 0:
                return np.zeros((1,1), "float32")
            else:
                q = 4 ** n
                m = q * normalized_bayer_matrix(n - 1)
                return np.bmat(((m-1.5, m+0.5), (m+1.5, m-0.5))) / q

        num_colors = len(pal_im.getpalette()) // 3
        spread = 2 * 256 / num_colors
        bayer_n = int(math.log2(order))
        bayer_matrix = torch.from_numpy(spread * normalized_bayer_matrix(bayer_n) + 0.5)

        result = torch.from_numpy(np.array(im).astype(np.float32))
        tw = math.ceil(result.shape[0] / bayer_matrix.shape[0])
        th = math.ceil(result.shape[1] / bayer_matrix.shape[1])
        tiled_matrix = bayer_matrix.tile(tw, th).unsqueeze(-1)
        result.add_(tiled_matrix[:result.shape[0],:result.shape[1]]).clamp_(0, 255)
        result = result.to(dtype=torch.uint8)

        im = Image.fromarray(result.cpu().numpy())
        im = im.quantize(palette=pal_im, dither=Image.Dither.NONE)
        return im

    @classmethod
    def execute(cls, image: torch.Tensor, colors: int, dither: str) -> io.NodeOutput:
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            im = Image.fromarray((image[b] * 255).to(torch.uint8).numpy(), mode='RGB')

            pal_im = im.quantize(colors=colors) # Required as described in https://github.com/python-pillow/Pillow/issues/5836

            if dither == "none":
                quantized_image = im.quantize(palette=pal_im, dither=Image.Dither.NONE)
            elif dither == "floyd-steinberg":
                quantized_image = im.quantize(palette=pal_im, dither=Image.Dither.FLOYDSTEINBERG)
            elif dither.startswith("bayer"):
                order = int(dither.split('-')[-1])
                quantized_image = Quantize.bayer(im, pal_im, order)

            quantized_array = torch.tensor(np.array(quantized_image.convert("RGB"))).float() / 255
            result[b] = quantized_array

        return io.NodeOutput(result)

class Sharpen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageSharpen",
            display_name="Sharpen Image",
            category="image/filters",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("sharpen_radius", default=1, min=1, max=31, step=1, advanced=True),
                io.Float.Input("sigma", default=1.0, min=0.1, max=10.0, step=0.01, advanced=True),
                io.Float.Input("alpha", default=1.0, min=0.0, max=5.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, sharpen_radius: int, sigma:float, alpha: float) -> io.NodeOutput:
        if sharpen_radius == 0:
            return io.NodeOutput(image)

        batch_size, height, width, channels = image.shape
        image = image.to(comfy.model_management.get_torch_device())

        kernel_size = sharpen_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device, dtype=image.dtype) * -(alpha*10)
        kernel = kernel.to(dtype=image.dtype)
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        tensor_image = F.pad(tensor_image, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return io.NodeOutput(result.to(comfy.model_management.intermediate_device()))

class ImageScaleToTotalPixels(io.ComfyNode):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageScaleToTotalPixels",
            display_name="Scale Image to Total Pixels",
            category="image/upscaling",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=cls.upscale_methods),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input("resolution_steps", default=1, min=1, max=256, advanced=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, megapixels, resolution_steps) -> io.NodeOutput:
        samples = image.movedim(-1,1)
        total = megapixels * 1024 * 1024

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

        s = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled")
        s = s.movedim(1,-1)
        return io.NodeOutput(s)

class ResizeType(str, Enum):
    SCALE_BY = "scale by multiplier"
    SCALE_DIMENSIONS = "scale dimensions"
    SCALE_LONGER_DIMENSION = "scale longer dimension"
    SCALE_SHORTER_DIMENSION = "scale shorter dimension"
    SCALE_WIDTH = "scale width"
    SCALE_HEIGHT = "scale height"
    SCALE_TOTAL_PIXELS = "scale total pixels"
    MATCH_SIZE = "match size"
    SCALE_TO_MULTIPLE = "scale to multiple"

def is_image(input: torch.Tensor) -> bool:
    # images have 4 dimensions: [batch, height, width, channels]
    # masks have 3 dimensions: [batch, height, width]
    return len(input.shape) == 4

def init_image_mask_input(input: torch.Tensor, is_type_image: bool) -> torch.Tensor:
    if is_type_image:
        input = input.movedim(-1, 1)
    else:
        input = input.unsqueeze(1)
    return input

def finalize_image_mask_input(input: torch.Tensor, is_type_image: bool) -> torch.Tensor:
    if is_type_image:
        input = input.movedim(1, -1)
    else:
        input = input.squeeze(1)
    return input

def scale_by(input: torch.Tensor, multiplier: float, scale_method: str) -> torch.Tensor:
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)
    width = round(input.shape[-1] * multiplier)
    height = round(input.shape[-2] * multiplier)

    input = comfy.utils.common_upscale(input, width, height, scale_method, "disabled")
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_dimensions(input: torch.Tensor, width: int, height: int, scale_method: str, crop: str="disabled") -> torch.Tensor:
    if width == 0 and height == 0:
        return input
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)

    if width == 0:
        width = max(1, round(input.shape[-1] * height / input.shape[-2]))
    elif height == 0:
        height = max(1, round(input.shape[-2] * width / input.shape[-1]))

    input = comfy.utils.common_upscale(input, width, height, scale_method, crop)
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_longer_dimension(input: torch.Tensor, longer_size: int, scale_method: str) -> torch.Tensor:
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)
    width = input.shape[-1]
    height = input.shape[-2]

    if height > width:
        width = round((width / height) * longer_size)
        height = longer_size
    elif width > height:
        height = round((height / width) * longer_size)
        width = longer_size
    else:
        height = longer_size
        width = longer_size

    input = comfy.utils.common_upscale(input, width, height, scale_method, "disabled")
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_shorter_dimension(input: torch.Tensor, shorter_size: int, scale_method: str) -> torch.Tensor:
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)
    width = input.shape[-1]
    height = input.shape[-2]

    if height < width:
        width = round((width / height) * shorter_size)
        height = shorter_size
    elif width < height:
        height = round((height / width) * shorter_size)
        width = shorter_size
    else:
        height = shorter_size
        width = shorter_size

    input = comfy.utils.common_upscale(input, width, height, scale_method, "disabled")
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_total_pixels(input: torch.Tensor, megapixels: float, scale_method: str) -> torch.Tensor:
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)
    total = int(megapixels * 1024 * 1024)

    scale_by = math.sqrt(total / (input.shape[-1] * input.shape[-2]))
    width = round(input.shape[-1] * scale_by)
    height = round(input.shape[-2] * scale_by)

    input = comfy.utils.common_upscale(input, width, height, scale_method, "disabled")
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_match_size(input: torch.Tensor, match: torch.Tensor, scale_method: str, crop: str) -> torch.Tensor:
    is_type_image = is_image(input)
    input = init_image_mask_input(input, is_type_image)
    match = init_image_mask_input(match, is_image(match))

    width = match.shape[-1]
    height = match.shape[-2]
    input = comfy.utils.common_upscale(input, width, height, scale_method, crop)
    input = finalize_image_mask_input(input, is_type_image)
    return input

def scale_to_multiple_cover(input: torch.Tensor, multiple: int, scale_method: str) -> torch.Tensor:
    if multiple <= 1:
        return input
    is_type_image = is_image(input)
    if is_type_image:
        _, height, width, _ = input.shape
    else:
        _, height, width = input.shape
    target_w = (width // multiple) * multiple
    target_h = (height // multiple) * multiple
    if target_w == 0 or target_h == 0:
        return input
    if target_w == width and target_h == height:
        return input
    s_w = target_w / width
    s_h = target_h / height
    if s_w >= s_h:
        scaled_w = target_w
        scaled_h = int(math.ceil(height * s_w))
        if scaled_h < target_h:
            scaled_h = target_h
    else:
        scaled_h = target_h
        scaled_w = int(math.ceil(width * s_h))
        if scaled_w < target_w:
            scaled_w = target_w
    input = init_image_mask_input(input, is_type_image)
    input = comfy.utils.common_upscale(input, scaled_w, scaled_h, scale_method, "disabled")
    input = finalize_image_mask_input(input, is_type_image)
    x0 = (scaled_w - target_w) // 2
    y0 = (scaled_h - target_h) // 2
    x1 = x0 + target_w
    y1 = y0 + target_h
    if is_type_image:
        return input[:, y0:y1, x0:x1, :]
    return input[:, y0:y1, x0:x1]

class ResizeImageMaskNode(io.ComfyNode):
    scale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    class ResizeTypedDict(TypedDict):
        resize_type: ResizeType
        scale_method: Literal["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        crop: Literal["disabled", "center"]
        multiplier: float
        width: int
        height: int
        longer_size: int
        shorter_size: int
        megapixels: float
        multiple: int

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("input_type", [io.Image, io.Mask])
        crop_combo = io.Combo.Input(
            "crop",
            options=cls.crop_methods,
            default="center",
            tooltip="How to handle aspect ratio mismatch: 'disabled' stretches to fit, 'center' crops to maintain aspect ratio.",
        )
        return io.Schema(
            node_id="ResizeImageMaskNode",
            display_name="Resize Image/Mask",
            description="Resize an image or mask using various scaling methods.",
            category="image/transform",
            search_aliases=["resize", "resize image", "resize mask", "scale", "scale image", "scale mask", "image resize", "change size", "dimensions", "shrink", "enlarge"],
            inputs=[
                io.MatchType.Input("input", template=template),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Select how to resize: by exact dimensions, scale factor, matching another image, etc.",
                    options=[
                        io.DynamicCombo.Option(ResizeType.SCALE_DIMENSIONS, [
                            io.Int.Input("width", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target width in pixels. Set to 0 to auto-calculate from height while preserving aspect ratio."),
                            io.Int.Input("height", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target height in pixels. Set to 0 to auto-calculate from width while preserving aspect ratio."),
                            crop_combo,
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_BY, [
                            io.Float.Input("multiplier", default=1.00, min=0.01, max=8.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles size, 0.5 halves size)."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_LONGER_DIMENSION, [
                            io.Int.Input("longer_size", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="The longer edge will be resized to this value. Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_SHORTER_DIMENSION, [
                            io.Int.Input("shorter_size", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="The shorter edge will be resized to this value. Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_WIDTH, [
                            io.Int.Input("width", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target width in pixels. Height auto-adjusts to preserve aspect ratio."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_HEIGHT, [
                            io.Int.Input("height", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target height in pixels. Width auto-adjusts to preserve aspect ratio."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_TOTAL_PIXELS, [
                            io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01, tooltip="Target total megapixels (e.g., 1.0 ≈ 1024×1024). Aspect ratio is preserved."),
                        ]),
                        io.DynamicCombo.Option(ResizeType.MATCH_SIZE, [
                            io.MultiType.Input("match", [io.Image, io.Mask], tooltip="Resize input to match the dimensions of this reference image or mask."),
                            crop_combo,
                        ]),
                        io.DynamicCombo.Option(ResizeType.SCALE_TO_MULTIPLE, [
                            io.Int.Input("multiple", default=8, min=1, max=MAX_RESOLUTION, step=1, tooltip="Resize so width and height are divisible by this number. Useful for latent alignment (e.g., 8 or 64)."),
                        ]),
                    ],
                ),
                io.Combo.Input(
                    "scale_method",
                    options=cls.scale_methods,
                    default="area",
                    tooltip="Interpolation algorithm. 'area' is best for downscaling, 'lanczos' for upscaling, 'nearest-exact' for pixel art.",
                ),
            ],
            outputs=[io.MatchType.Output(template=template, display_name="resized")]
        )

    @classmethod
    def execute(cls, input: io.Image.Type | io.Mask.Type, scale_method: io.Combo.Type, resize_type: ResizeTypedDict) -> io.NodeOutput:
        selected_type = resize_type["resize_type"]
        if selected_type == ResizeType.SCALE_BY:
            return io.NodeOutput(scale_by(input, resize_type["multiplier"], scale_method))
        elif selected_type == ResizeType.SCALE_DIMENSIONS:
            return io.NodeOutput(scale_dimensions(input, resize_type["width"], resize_type["height"], scale_method, resize_type["crop"]))
        elif selected_type == ResizeType.SCALE_LONGER_DIMENSION:
            return io.NodeOutput(scale_longer_dimension(input, resize_type["longer_size"], scale_method))
        elif selected_type == ResizeType.SCALE_SHORTER_DIMENSION:
            return io.NodeOutput(scale_shorter_dimension(input, resize_type["shorter_size"], scale_method))
        elif selected_type == ResizeType.SCALE_WIDTH:
            return io.NodeOutput(scale_dimensions(input, resize_type["width"], 0, scale_method))
        elif selected_type == ResizeType.SCALE_HEIGHT:
            return io.NodeOutput(scale_dimensions(input, 0, resize_type["height"], scale_method))
        elif selected_type == ResizeType.SCALE_TOTAL_PIXELS:
            return io.NodeOutput(scale_total_pixels(input, resize_type["megapixels"], scale_method))
        elif selected_type == ResizeType.MATCH_SIZE:
            return io.NodeOutput(scale_match_size(input, resize_type["match"], scale_method, resize_type["crop"]))
        elif selected_type == ResizeType.SCALE_TO_MULTIPLE:
            return io.NodeOutput(scale_to_multiple_cover(input, resize_type["multiple"], scale_method))
        raise ValueError(f"Unsupported resize type: {selected_type}")

def batch_images(images: list[torch.Tensor]) -> torch.Tensor | None:
    if len(images) == 0:
        return None
    # first, get the max channels count
    max_channels = max(image.shape[-1] for image in images)
    # then, pad all images to have the same channels count
    padded_images: list[torch.Tensor] = []
    for image in images:
        if image.shape[-1] < max_channels:
            padded_images.append(torch.nn.functional.pad(image, (0,1), mode='constant', value=1.0))
        else:
            padded_images.append(image)
    # resize all images to be the same size as the first image
    resized_images: list[torch.Tensor] = []
    first_image_shape = padded_images[0].shape
    for image in padded_images:
        if image.shape[1:] != first_image_shape[1:]:
            resized_images.append(comfy.utils.common_upscale(image.movedim(-1,1), first_image_shape[2], first_image_shape[1], "bilinear", "center").movedim(1,-1))
        else:
            resized_images.append(image)
    # batch the images in the format [b, h, w, c]
    return torch.cat(resized_images, dim=0)

def batch_masks(masks: list[torch.Tensor]) -> torch.Tensor | None:
    if len(masks) == 0:
        return None
    # resize all masks to be the same size as the first mask
    resized_masks: list[torch.Tensor] = []
    first_mask_shape = masks[0].shape
    for mask in masks:
        if mask.shape[1:] != first_mask_shape[1:]:
            mask = init_image_mask_input(mask, is_type_image=False)
            mask = comfy.utils.common_upscale(mask, first_mask_shape[2], first_mask_shape[1], "bilinear", "center")
            resized_masks.append(finalize_image_mask_input(mask, is_type_image=False))
        else:
            resized_masks.append(mask)
    # batch the masks in the format [b, h, w]
    return torch.cat(resized_masks, dim=0)

def batch_latents(latents: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    if len(latents) == 0:
        return None
    samples_out = latents[0].copy()
    samples_out["batch_index"] = []
    first_samples = latents[0]["samples"]
    tensors: list[torch.Tensor] = []
    for latent in latents:
        # first, deal with latent tensors
        tensors.append(reshape_latent_to(first_samples.shape, latent["samples"], repeat_batch=False))
        # next, deal with batch_index
        samples_out["batch_index"].extend(latent.get("batch_index", [x for x in range(0, latent["samples"].shape[0])]))
    samples_out["samples"] = torch.cat(tensors, dim=0)
    return samples_out

class BatchImagesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(io.Image.Input("image"), prefix="image", min=1, max=50)
        return io.Schema(
            node_id="BatchImagesNode",
            display_name="Batch Images",
            category="image/batch",
            essentials_category="Image Tools",
            search_aliases=["batch", "image batch", "batch images", "combine images", "merge images", "stack images"],
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template)
            ],
            outputs=[
                io.Image.Output()
            ]
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(batch_images(list(images.values())))

class BatchMasksNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(io.Mask.Input("mask"), prefix="mask", min=1, max=50)
        return io.Schema(
            node_id="BatchMasksNode",
            search_aliases=["combine masks", "stack masks", "merge masks"],
            display_name="Batch Masks",
            category="image/mask",
            inputs=[
                io.Autogrow.Input("masks", template=autogrow_template)
            ],
            outputs=[
                io.Mask.Output()
            ]
        )

    @classmethod
    def execute(cls, masks: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(batch_masks(list(masks.values())))

class BatchLatentsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(io.Latent.Input("latent"), prefix="latent", min=1, max=50)
        return io.Schema(
            node_id="BatchLatentsNode",
            search_aliases=["combine latents", "stack latents", "merge latents"],
            display_name="Batch Latents",
            category="model/latent/batch",
            inputs=[
                io.Autogrow.Input("latents", template=autogrow_template)
            ],
            outputs=[
                io.Latent.Output()
            ]
        )

    @classmethod
    def execute(cls, latents: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(batch_latents(list(latents.values())))

class BatchImagesMasksLatentsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        matchtype_template = io.MatchType.Template("input", allowed_types=[io.Image, io.Mask, io.Latent])
        autogrow_template = io.Autogrow.TemplatePrefix(
                io.MatchType.Input("input", matchtype_template),
                prefix="input", min=1, max=50)
        return io.Schema(
            node_id="BatchImagesMasksLatentsNode",
            search_aliases=["combine batch", "merge batch", "stack inputs"],
            display_name="Batch Images/Masks/Latents",
            category="util",
            inputs=[
                io.Autogrow.Input("inputs", template=autogrow_template)
            ],
            outputs=[
                io.MatchType.Output(id=None, template=matchtype_template)
            ]
        )

    @classmethod
    def execute(cls, inputs: io.Autogrow.Type) -> io.NodeOutput:
        batched = None
        values = list(inputs.values())
        # latents
        if isinstance(values[0], dict):
            batched = batch_latents(values)
        # images
        elif is_image(values[0]):
            batched = batch_images(values)
        # masks
        else:
            batched = batch_masks(values)
        return io.NodeOutput(batched)


class ColorTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ColorTransfer",
            display_name="Transfer Color",
            category="image/filters",
            description="Match the colors of one image to another using various algorithms.",
            search_aliases=["color match", "color grading", "color correction", "match colors", "color transform", "mkl", "reinhard", "histogram"],
            inputs=[
                io.Image.Input("image_target", tooltip="Image(s) to apply the color transform to."),
                io.Image.Input("image_ref", tooltip="Reference image(s) to match colors to."),
                io.Combo.Input("method", options=['reinhard_lab', 'mkl_lab', 'histogram'],),
                io.DynamicCombo.Input("source_stats",
                    tooltip="per_frame: each frame matched to image_ref individually. uniform: pool stats across all source frames as baseline, match to image_ref. target_frame: use one chosen frame as the baseline for the transform to image_ref, applied uniformly to all frames (preserves relative differences)",
                    options=[
                        io.DynamicCombo.Option("per_frame", []),
                        io.DynamicCombo.Option("uniform", []),
                        io.DynamicCombo.Option("target_frame", [
                            io.Int.Input("target_index", default=0, min=0, max=10000,
                                tooltip="Frame index used as the source baseline for computing the transform to image_ref"),
                        ]),
                    ]),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @staticmethod
    def _to_lab(images, i, device):
        return kornia.color.rgb_to_lab(
            images[i:i+1].to(device, dtype=torch.float32).permute(0, 3, 1, 2))

    @staticmethod
    def _pool_stats(images, device, is_reinhard, eps):
        """Two-pass pooled mean + std/cov across all frames."""
        N, C = images.shape[0], images.shape[3]
        HW = images.shape[1] * images.shape[2]
        mean = torch.zeros(C, 1, device=device, dtype=torch.float32)
        for i in range(N):
            mean += ColorTransfer._to_lab(images, i, device).view(C, -1).mean(dim=-1, keepdim=True)
        mean /= N
        acc = torch.zeros(C, 1 if is_reinhard else C, device=device, dtype=torch.float32)
        for i in range(N):
            centered = ColorTransfer._to_lab(images, i, device).view(C, -1) - mean
            if is_reinhard:
                acc += (centered * centered).mean(dim=-1, keepdim=True)
            else:
                acc += centered @ centered.T / HW
        if is_reinhard:
            return mean, torch.sqrt(acc / N).clamp_min_(eps)
        return mean, acc / N

    @staticmethod
    def _frame_stats(lab_flat, hw, is_reinhard, eps):
        """Per-frame mean + std/cov."""
        mean = lab_flat.mean(dim=-1, keepdim=True)
        if is_reinhard:
            return mean, lab_flat.std(dim=-1, keepdim=True, unbiased=False).clamp_min_(eps)
        centered = lab_flat - mean
        return mean, centered @ centered.T / hw

    @staticmethod
    def _mkl_matrix(cov_s, cov_r, eps):
        """Compute MKL 3x3 transform matrix from source and ref covariances."""
        eig_val_s, eig_vec_s = torch.linalg.eigh(cov_s)
        sqrt_val_s = torch.sqrt(eig_val_s.clamp_min(0)).clamp_min_(eps)

        scaled_V = eig_vec_s * sqrt_val_s.unsqueeze(0)
        mid = scaled_V.T @ cov_r @ scaled_V
        eig_val_m, eig_vec_m = torch.linalg.eigh(mid)
        sqrt_m = torch.sqrt(eig_val_m.clamp_min(0))

        inv_sqrt_s = 1.0 / sqrt_val_s
        inv_scaled_V = eig_vec_s * inv_sqrt_s.unsqueeze(0)
        M_half = (eig_vec_m * sqrt_m.unsqueeze(0)) @ eig_vec_m.T
        return inv_scaled_V @ M_half @ inv_scaled_V.T

    @staticmethod
    def _histogram_lut(src, ref, bins=256):
        """Build per-channel LUT from source and ref histograms. src/ref: (C, HW) in [0,1]."""
        s_bins = (src * (bins - 1)).long().clamp(0, bins - 1)
        r_bins = (ref * (bins - 1)).long().clamp(0, bins - 1)
        s_hist = torch.zeros(src.shape[0], bins, device=src.device, dtype=src.dtype)
        r_hist = torch.zeros(src.shape[0], bins, device=src.device, dtype=src.dtype)
        ones_s = torch.ones_like(src)
        ones_r = torch.ones_like(ref)
        s_hist.scatter_add_(1, s_bins, ones_s)
        r_hist.scatter_add_(1, r_bins, ones_r)
        s_cdf = s_hist.cumsum(1)
        s_cdf = s_cdf / s_cdf[:, -1:]
        r_cdf = r_hist.cumsum(1)
        r_cdf = r_cdf / r_cdf[:, -1:]
        return torch.searchsorted(r_cdf, s_cdf).clamp_max_(bins - 1).float() / (bins - 1)

    @classmethod
    def _pooled_cdf(cls, images, device, num_bins=256):
        """Build pooled CDF across all frames, one frame at a time."""
        C = images.shape[3]
        hist = torch.zeros(C, num_bins, device=device, dtype=torch.float32)
        for i in range(images.shape[0]):
            frame = images[i].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(C, -1)
            bins = (frame * (num_bins - 1)).long().clamp(0, num_bins - 1)
            hist.scatter_add_(1, bins, torch.ones_like(frame))
        cdf = hist.cumsum(1)
        return cdf / cdf[:, -1:]

    @classmethod
    def _build_histogram_transform(cls, image_target, image_ref, device, stats_mode, target_index, B):
        """Build per-frame or uniform LUT transform for histogram mode."""
        if stats_mode == 'per_frame':
            return None  # LUT computed per-frame in the apply loop

        r_cdf = cls._pooled_cdf(image_ref, device)
        if stats_mode == 'target_frame':
            ti = min(target_index, B - 1)
            s_cdf = cls._pooled_cdf(image_target[ti:ti+1], device)
        else:
            s_cdf = cls._pooled_cdf(image_target, device)
        return torch.searchsorted(r_cdf, s_cdf).clamp_max_(255).float() / 255.0

    @classmethod
    def _build_lab_transform(cls, image_target, image_ref, device, stats_mode, target_index, is_reinhard):
        """Build transform parameters for Lab-based methods. Returns a transform function."""
        eps = 1e-6
        B, H, W, C = image_target.shape
        B_ref = image_ref.shape[0]
        single_ref = B_ref == 1
        HW = H * W
        HW_ref = image_ref.shape[1] * image_ref.shape[2]

        # Precompute ref stats
        if single_ref or stats_mode in ('uniform', 'target_frame'):
            ref_mean, ref_sc = cls._pool_stats(image_ref, device, is_reinhard, eps)

        # Uniform/target_frame: precompute single affine transform
        if stats_mode in ('uniform', 'target_frame'):
            if stats_mode == 'target_frame':
                ti = min(target_index, B - 1)
                s_lab = cls._to_lab(image_target, ti, device).view(C, -1)
                s_mean, s_sc = cls._frame_stats(s_lab, HW, is_reinhard, eps)
            else:
                s_mean, s_sc = cls._pool_stats(image_target, device, is_reinhard, eps)

            if is_reinhard:
                scale = ref_sc / s_sc
                offset = ref_mean - scale * s_mean
                return lambda src_flat, **_: src_flat * scale + offset
            T = cls._mkl_matrix(s_sc, ref_sc, eps)
            offset = ref_mean - T @ s_mean
            return lambda src_flat, **_: T @ src_flat + offset

        # per_frame
        def per_frame_transform(src_flat, frame_idx):
            s_mean, s_sc = cls._frame_stats(src_flat, HW, is_reinhard, eps)

            if single_ref:
                r_mean, r_sc = ref_mean, ref_sc
            else:
                ri = min(frame_idx, B_ref - 1)
                r_mean, r_sc = cls._frame_stats(cls._to_lab(image_ref, ri, device).view(C, -1), HW_ref, is_reinhard, eps)

            centered = src_flat - s_mean
            if is_reinhard:
                return centered * (r_sc / s_sc) + r_mean
            T = cls._mkl_matrix(centered @ centered.T / HW, r_sc, eps)
            return T @ centered + r_mean

        return per_frame_transform

    @classmethod
    def execute(cls, image_target, image_ref, method, source_stats, strength=1.0) -> io.NodeOutput:
        stats_mode = source_stats["source_stats"]
        target_index = source_stats.get("target_index", 0)

        if strength == 0 or image_ref is None:
            return io.NodeOutput(image_target)

        device = comfy.model_management.get_torch_device()
        intermediate_device = comfy.model_management.intermediate_device()
        intermediate_dtype = comfy.model_management.intermediate_dtype()

        B, H, W, C = image_target.shape
        B_ref = image_ref.shape[0]
        pbar = comfy.utils.ProgressBar(B)
        out = torch.empty(B, H, W, C, device=intermediate_device, dtype=intermediate_dtype)

        if method == 'histogram':
            uniform_lut = cls._build_histogram_transform(
                image_target, image_ref, device, stats_mode, target_index, B)

            for i in range(B):
                src = image_target[i].to(device, dtype=torch.float32).permute(2, 0, 1)
                src_flat = src.reshape(C, -1)
                if uniform_lut is not None:
                    lut = uniform_lut
                else:
                    ri = min(i, B_ref - 1)
                    ref = image_ref[ri].to(device, dtype=torch.float32).permute(2, 0, 1).reshape(C, -1)
                    lut = cls._histogram_lut(src_flat, ref)
                bin_idx = (src_flat * 255).long().clamp(0, 255)
                matched = lut.gather(1, bin_idx).view(C, H, W)
                result = matched if strength == 1.0 else torch.lerp(src, matched, strength)
                out[i] = result.permute(1, 2, 0).clamp_(0, 1).to(device=intermediate_device, dtype=intermediate_dtype)
                pbar.update(1)
        else:
            transform = cls._build_lab_transform(image_target, image_ref, device, stats_mode, target_index, is_reinhard=method == "reinhard_lab")

            for i in range(B):
                src_frame = cls._to_lab(image_target, i, device)
                corrected = transform(src_frame.view(C, -1), frame_idx=i)
                if strength == 1.0:
                    result = kornia.color.lab_to_rgb(corrected.view(1, C, H, W))
                else:
                    result = kornia.color.lab_to_rgb(torch.lerp(src_frame, corrected.view(1, C, H, W), strength))
                out[i] = result.squeeze(0).permute(1, 2, 0).clamp_(0, 1).to(device=intermediate_device, dtype=intermediate_dtype)
                pbar.update(1)

        return io.NodeOutput(out)


class PostProcessingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Blend,
            Blur,
            Quantize,
            Sharpen,
            ImageScaleToTotalPixels,
            ResizeImageMaskNode,
            BatchImagesNode,
            BatchMasksNode,
            BatchLatentsNode,
            ColorTransfer,
            # BatchImagesMasksLatentsNode,
        ]

async def comfy_entrypoint() -> PostProcessingExtension:
    return PostProcessingExtension()
