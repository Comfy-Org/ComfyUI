import nodes
import folder_paths

import av
import json

import os
import re
import math
import numpy as np
import struct
import torch

import zlib
import comfy.utils
from fractions import Fraction

from server import PromptServer
from comfy_api.latest import ComfyExtension, IO, UI
from comfy.cli_args import args
from typing_extensions import override

SVG = IO.SVG.Type  # TODO: temporary solution for backward compatibility, will be removed later.

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class ImageCrop(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCrop",
            search_aliases=["trim"],
            display_name="Crop Image (DEPRECATED)",
            category="image/transform",
            is_deprecated=True,
            essentials_category="Image Tools",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("x", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("y", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, x, y) -> IO.NodeOutput:
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        return IO.NodeOutput(img)

    crop = execute  # TODO: remove


class ImageCropV2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageCropV2",
            search_aliases=["crop", "cut", "trim"],
            display_name="Crop Image",
            category="image/transform",
            description = "Crop an image to the specified dimensions.",
            essentials_category="Image Tools",
            has_intermediate_output=True,
            inputs=[
                IO.Image.Input("image"),
                IO.BoundingBox.Input("crop_region", component="ImageCrop"),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, crop_region) -> IO.NodeOutput:
        x = crop_region.get("x", 0)
        y = crop_region.get("y", 0)
        width = crop_region.get("width", 512)
        height = crop_region.get("height", 512)

        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        return IO.NodeOutput(img, ui=UI.PreviewImage(img))


class BoundingBox(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrimitiveBoundingBox",
            display_name="Bounding Box",
            category="utilities/primitive",
            inputs=[
                IO.Int.Input("x", default=0, min=0, max=MAX_RESOLUTION),
                IO.Int.Input("y", default=0, min=0, max=MAX_RESOLUTION),
                IO.Int.Input("width", default=512, min=1, max=MAX_RESOLUTION),
                IO.Int.Input("height", default=512, min=1, max=MAX_RESOLUTION),
            ],
            outputs=[IO.BoundingBox.Output()],
        )

    @classmethod
    def execute(cls, x, y, width, height) -> IO.NodeOutput:
        return IO.NodeOutput({"x": x, "y": y, "width": width, "height": height})


class RepeatImageBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RepeatImageBatch",
            search_aliases=["duplicate image", "clone image"],
            display_name="Repeat Image Batch",
            category="image/batch",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("amount", default=1, min=1, max=4096),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, amount) -> IO.NodeOutput:
        s = image.repeat((amount, 1,1,1))
        return IO.NodeOutput(s)

    repeat = execute  # TODO: remove


class ImageFromBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageFromBatch",
            search_aliases=["select image", "pick from batch", "extract image"],
            display_name="Get Image from Batch",
            category="image/batch",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("batch_index", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION),
                IO.Int.Input("length", default=1, min=1, max=4096),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, batch_index, length) -> IO.NodeOutput:
        s_in = image
        if batch_index < 0:
            batch_index += s_in.shape[0]
        batch_index = max(0, min(s_in.shape[0] - 1, batch_index))
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return IO.NodeOutput(s)

    frombatch = execute  # TODO: remove


class ImageAddNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAddNoise",
            search_aliases=["film grain"],
            display_name="Add Noise to Image",
            category="image/filters",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                ),
                IO.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, seed, strength) -> IO.NodeOutput:
        generator = torch.manual_seed(seed)
        s = torch.clip((image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image)), min=0.0, max=1.0)
        return IO.NodeOutput(s)

    repeat = execute  # TODO: remove


class SaveAnimatedWEBP(IO.ComfyNode):
    COMPRESS_METHODS = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAnimatedWEBP",
            display_name="Save Animated WEBP",
            category="image",
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                IO.Boolean.Input("lossless", default=True),
                IO.Int.Input("quality", default=80, min=0, max=100),
                IO.Combo.Input("method", options=list(cls.COMPRESS_METHODS.keys())),
                # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images")]
        )

    @classmethod
    def execute(cls, images, fps, filename_prefix, lossless, quality, method, num_frames=0) -> IO.NodeOutput:
        return IO.NodeOutput(
            images,
            ui=UI.ImageSaveHelper.get_save_animated_webp_ui(
                images=images,
                filename_prefix=filename_prefix,
                cls=cls,
                fps=fps,
                lossless=lossless,
                quality=quality,
                method=cls.COMPRESS_METHODS.get(method)
            )
        )


class SaveAnimatedPNG(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveAnimatedPNG",
            display_name="Save Animated PNG",
            category="image",
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                IO.Int.Input("compress_level", default=4, min=0, max=9, advanced=True),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images")]
        )

    @classmethod
    def execute(cls, images, fps, compress_level, filename_prefix="ComfyUI") -> IO.NodeOutput:
        return IO.NodeOutput(
            images,
            ui=UI.ImageSaveHelper.get_save_animated_png_ui(
                images=images,
                filename_prefix=filename_prefix,
                cls=cls,
                fps=fps,
                compress_level=compress_level,
            )
        )


class ImageStitch(IO.ComfyNode):
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageStitch",
            search_aliases=["combine images", "join images", "concatenate images", "side by side"],
            display_name="Stitch Images",
            description="Stitches image2 to image1 in the specified direction.\n"
            "If image2 is not provided, returns image1 unchanged.\n"
            "Optional spacing can be added between images.",
            category="image/transform",
            inputs=[
                IO.Image.Input("image1"),
                IO.Combo.Input("direction", options=["right", "down", "left", "up"], default="right"),
                IO.Boolean.Input("match_image_size", default=True),
                IO.Int.Input("spacing_width", default=0, min=0, max=1024, step=2, advanced=True),
                IO.Combo.Input("spacing_color", options=["white", "black", "red", "green", "blue"], default="white", advanced=True),
                IO.Image.Input("image2", optional=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        image1,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        image2=None,
    ) -> IO.NodeOutput:
        if image2 is None:
            return IO.NodeOutput(image1)

        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )

        # Match image sizes if requested
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:  # up, down
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = comfy.utils.common_upscale(
                image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            ).movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        # When not matching sizes, pad to align non-concat dimensions
        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                # For horizontal concat, pad heights to match
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

        # Ensure same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        # Add spacing if specified
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:  # Add alpha
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        # Concatenate images
        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return IO.NodeOutput(torch.cat(images, dim=concat_dim))

    stitch = execute  # TODO: remove


class ResizeAndPadImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ResizeAndPadImage",
            search_aliases=["fit to size"],
            display_name="Resize And Pad Image",
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("target_width", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Int.Input("target_height", default=512, min=1, max=nodes.MAX_RESOLUTION, step=1),
                IO.Combo.Input("padding_color", options=["white", "black"], advanced=True),
                IO.Combo.Input("interpolation", options=["area", "bicubic", "nearest-exact", "bilinear", "lanczos"], advanced=True),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, target_width, target_height, padding_color, interpolation) -> IO.NodeOutput:
        batch_size, orig_height, orig_width, channels = image.shape

        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_permuted = image.permute(0, 3, 1, 2)

        resized = comfy.utils.common_upscale(image_permuted, new_width, new_height, interpolation, "disabled")

        pad_value = 0.0 if padding_color == "black" else 1.0
        padded = torch.full(
            (batch_size, channels, target_height, target_width),
            pad_value,
            dtype=image.dtype,
            device=image.device
        )

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        padded[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        output = padded.permute(0, 2, 3, 1)
        return IO.NodeOutput(output)

    resize_and_pad = execute  # TODO: remove


class SaveSVGNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveSVGNode",
            search_aliases=["export vector", "save vector graphics"],
            display_name="Save SVG",
            description="Save SVG files on disk.",
            category="image",
            inputs=[
                IO.SVG.Input("svg"),
                IO.String.Input(
                    "filename_prefix",
                    default="svg/ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.SVG.Output("svg")],
        )

    @classmethod
    def execute(cls, svg: IO.SVG.Type, filename_prefix="svg/ComfyUI") -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results: list[UI.SavedResult] = []

        # Prepare metadata JSON
        metadata_dict = {}
        if cls.hidden.prompt is not None:
            metadata_dict["prompt"] = cls.hidden.prompt
        if cls.hidden.extra_pnginfo is not None:
            metadata_dict.update(cls.hidden.extra_pnginfo)

        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata_dict, indent=2) if metadata_dict else None


        for batch_number, svg_bytes in enumerate(svg.data):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.svg"

            # Read SVG content
            svg_bytes.seek(0)
            svg_content = svg_bytes.read().decode('utf-8')

            # Inject metadata if available
            if metadata_json:
                # Create metadata element with CDATA section
                metadata_element = f"""  <metadata>
                <![CDATA[
            {metadata_json}
                ]]>
            </metadata>
            """
                # Insert metadata after opening svg tag using regex with a replacement function
                def replacement(match):
                    # match.group(1) contains the captured <svg> tag
                    return match.group(1) + '\n' + metadata_element

                # Apply the substitution
                svg_content = re.sub(r'(<svg[^>]*>)', replacement, svg_content, flags=re.UNICODE)

            # Write the modified SVG to file
            with open(os.path.join(full_output_folder, file), 'wb') as svg_file:
                svg_file.write(svg_content.encode('utf-8'))

            results.append(UI.SavedResult(filename=file, subfolder=subfolder, type=IO.FolderType.output))
            counter += 1
        return IO.NodeOutput(svg, ui={"images": results})


class GetImageSize(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetImageSize",
            search_aliases=["dimensions", "resolution", "image info"],
            display_name="Get Image Size",
            description="Returns width and height of the image, and passes it through unchanged.",
            category="image",
            inputs=[
                IO.Image.Input("image"),
            ],
            outputs=[
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Int.Output(display_name="batch_size"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        height = image.shape[1]
        width = image.shape[2]
        batch_size = image.shape[0]

        # Send progress text to display size on the node
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(f"width: {width}, height: {height}\n batch size: {batch_size}", cls.hidden.unique_id)

        return IO.NodeOutput(width, height, batch_size)

    get_size = execute  # TODO: remove


class ImageRotate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageRotate",
            display_name="Rotate Image",
            search_aliases=["turn", "flip orientation"],
            category="image/transform",
            essentials_category="Image Tools",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("rotation", options=["none", "90 degrees", "180 degrees", "270 degrees"]),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, rotation) -> IO.NodeOutput:
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        image = torch.rot90(image, k=rotate_by, dims=[2, 1])
        return IO.NodeOutput(image)

    rotate = execute  # TODO: remove


class ImageFlip(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageFlip",
            search_aliases=["mirror", "reflect"],
            display_name="Flip Image",
            category="image/transform",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("flip_method", options=["x-axis: vertically", "y-axis: horizontally"]),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, flip_method) -> IO.NodeOutput:
        if flip_method.startswith("x"):
            image = torch.flip(image, dims=[1])
        elif flip_method.startswith("y"):
            image = torch.flip(image, dims=[2])

        return IO.NodeOutput(image)

    flip = execute  # TODO: remove


class ImageScaleToMaxDimension(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageScaleToMaxDimension",
            display_name="Scale Image to Max Dimension",
            category="image/upscaling",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input(
                    "upscale_method",
                    options=["area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"],
                ),
                IO.Int.Input("largest_size", default=512, min=0, max=MAX_RESOLUTION, step=1),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, upscale_method, largest_size) -> IO.NodeOutput:
        height = image.shape[1]
        width = image.shape[2]

        if height > width:
            width = round((width / height) * largest_size)
            height = largest_size
        elif width > height:
            height = round((height / width) * largest_size)
            width = largest_size
        else:
            height = largest_size
            width = largest_size

        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return IO.NodeOutput(s)

    upscale = execute    # TODO: remove


class SplitImageToTileList(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplitImageToTileList",
            category="image/batch",
            search_aliases=["split image", "tile image", "slice image"],
            display_name="Split Image into List of Tiles",
            description="Splits an image into a batched list of tiles with a specified overlap.",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("tile_width", default=1024, min=64, max=MAX_RESOLUTION),
                IO.Int.Input("tile_height", default=1024, min=64, max=MAX_RESOLUTION),
                IO.Int.Input("overlap", default=128, min=0, max=4096),
            ],
            outputs=[
                IO.Image.Output(is_output_list=True),
            ],
        )

    @staticmethod
    def get_grid_coords(width, height, tile_width, tile_height, overlap):
        coords = []
        stride_x = round(max(tile_width * 0.25, tile_width - overlap))
        stride_y = round(max(tile_height * 0.25, tile_height - overlap))

        y = 0
        while y < height:
            x = 0
            y_end = min(y + tile_height, height)
            y_start = max(0, y_end - tile_height)

            while x < width:
                x_end = min(x + tile_width, width)
                x_start = max(0, x_end - tile_width)

                coords.append((x_start, y_start, x_end, y_end))

                if x_end >= width:
                    break
                x += stride_x

            if y_end >= height:
                break
            y += stride_y

        return coords

    @classmethod
    def execute(cls, image, tile_width, tile_height, overlap):
        b, h, w, c = image.shape
        coords = cls.get_grid_coords(w, h, tile_width, tile_height, overlap)

        output_list = []
        for (x_start, y_start, x_end, y_end) in coords:
            tile = image[:, y_start:y_end, x_start:x_end, :]
            output_list.append(tile)

        return IO.NodeOutput(output_list)


class ImageMergeTileList(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageMergeTileList",
            display_name="Merge List of Tiles to Image",
            category="image/batch",
            search_aliases=["split image", "tile image", "slice image"],
            is_input_list=True,
            inputs=[
                IO.Image.Input("image_list"),
                IO.Int.Input("final_width", default=1024, min=64, max=32768),
                IO.Int.Input("final_height", default=1024, min=64, max=32768),
                IO.Int.Input("overlap", default=128, min=0, max=4096),
            ],
            outputs=[
                IO.Image.Output(is_output_list=False),
            ],
        )

    @classmethod
    def execute(cls, image_list, final_width, final_height, overlap):
        w = final_width[0]
        h = final_height[0]
        ovlp = overlap[0]
        feather_str = 1.0

        first_tile = image_list[0]
        b, t_h, t_w, c = first_tile.shape
        device = first_tile.device
        dtype = first_tile.dtype

        coords = SplitImageToTileList.get_grid_coords(w, h, t_w, t_h, ovlp)

        canvas = torch.zeros((b, h, w, c), device=device, dtype=dtype)
        weights = torch.zeros((b, h, w, 1), device=device, dtype=dtype)

        if ovlp > 0:
            y_w = torch.sin(math.pi * torch.linspace(0, 1, t_h, device=device, dtype=dtype))
            x_w = torch.sin(math.pi * torch.linspace(0, 1, t_w, device=device, dtype=dtype))
            y_w = torch.clamp(y_w, min=1e-5)
            x_w = torch.clamp(x_w, min=1e-5)

            sine_mask = (y_w.unsqueeze(1) * x_w.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)
            flat_mask = torch.ones_like(sine_mask)

            weight_mask = torch.lerp(flat_mask, sine_mask, feather_str)
        else:
            weight_mask = torch.ones((1, t_h, t_w, 1), device=device, dtype=dtype)

        for i, (x_start, y_start, x_end, y_end) in enumerate(coords):
            if i >= len(image_list):
                break

            tile = image_list[i]

            region_h = y_end - y_start
            region_w = x_end - x_start

            real_h = min(region_h, tile.shape[1])
            real_w = min(region_w, tile.shape[2])

            y_end_actual = y_start + real_h
            x_end_actual = x_start + real_w

            tile_crop = tile[:, :real_h, :real_w, :]
            mask_crop = weight_mask[:, :real_h, :real_w, :]

            canvas[:, y_start:y_end_actual, x_start:x_end_actual, :] += tile_crop * mask_crop
            weights[:, y_start:y_end_actual, x_start:x_end_actual, :] += mask_crop

        weights[weights == 0] = 1.0
        merged_image = canvas / weights

        return IO.NodeOutput(merged_image)


# ---------------------------------------------------------------------------
# Format specifications
# ---------------------------------------------------------------------------

# Maps (file_format, bit_depth, has_alpha) -> (numpy dtype scale, av pixel format,
# stream pix_fmt). Keeps the encode path declarative instead of branchy.
_FORMAT_SPECS = {
    ("png", "8-bit", False):  {"scale": 255.0,   "dtype": np.uint8,   "frame_fmt": "rgb24",     "stream_fmt": "rgb24"},
    ("png", "8-bit", True):   {"scale": 255.0,   "dtype": np.uint8,   "frame_fmt": "rgba",      "stream_fmt": "rgba"},
    ("png", "16-bit", False): {"scale": 65535.0, "dtype": np.uint16,  "frame_fmt": "rgb48le",   "stream_fmt": "rgb48be"},
    ("png", "16-bit", True):  {"scale": 65535.0, "dtype": np.uint16,  "frame_fmt": "rgba64le",  "stream_fmt": "rgba64be"},
    ("exr", "32-bit float", False): {"scale": 1.0, "dtype": np.float32, "frame_fmt": "gbrpf32le",  "stream_fmt": "gbrpf32le"},
    ("exr", "32-bit float", True):  {"scale": 1.0, "dtype": np.float32, "frame_fmt": "gbrapf32le", "stream_fmt": "gbrapf32le"},
}


# ---------------------------------------------------------------------------
# Color transforms
# ---------------------------------------------------------------------------

def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF (IEC 61966-2-1). Operates on RGB channels only;
    alpha (if present as the 4th channel) is passed through unchanged."""
    if t.shape[-1] == 4:
        rgb, alpha = t[..., :3], t[..., 3:]
        return torch.cat([srgb_to_linear(rgb), alpha], dim=-1)

    # Piecewise: linear toe below 0.04045, gamma curve above.
    low = t / 12.92
    high = ((t.clamp(min=0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(t <= 0.04045, low, high)


# HLG OETF constants from BT.2100 Table 5.
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991072928   # = 0.5 - a*ln(4*a)


def hlg_to_linear(t: torch.Tensor) -> torch.Tensor:
    """Inverse HLG OETF (BT.2100). Maps a non-linear HLG signal in [0, 1] to
    *scene*-linear light in [0, 1]. Per BT.2100 Note 5a, this is the correct
    transform when converting HLG to a linear scene-light representation
    (rather than display-light, which would also involve the HLG OOTF).

    Operates on RGB channels only; alpha is passed through unchanged."""
    if t.shape[-1] == 4:
        rgb, alpha = t[..., :3], t[..., 3:]
        return torch.cat([hlg_to_linear(rgb), alpha], dim=-1)

    # Piecewise: sqrt branch below 0.5, log branch above.
    # Clamp inside the log branch so negative / out-of-range values don't blow up;
    # values above 1.0 are allowed and extrapolate naturally.
    low = (t ** 2) / 3.0
    high = (torch.exp((t.clamp(min=_HLG_C) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    return torch.where(t <= 0.5, low, high)


# ---------------------------------------------------------------------------
# Metadata injection
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk: length | type | data | CRC32(type+data)."""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _png_text_chunk(keyword: str, text: str) -> bytes:
    """tEXt chunk: latin-1 keyword + NUL + latin-1 text."""
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1", errors="replace")
    return _png_chunk(b"tEXt", payload)


def inject_png_metadata(png_bytes: bytes, prompt: dict | None, extra_pnginfo: dict | None) -> bytes:
    """Insert ComfyUI prompt/workflow as tEXt chunks right after IHDR."""
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return png_bytes

    chunks: list[bytes] = []
    if prompt is not None:
        chunks.append(_png_text_chunk("prompt", json.dumps(prompt)))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            chunks.append(_png_text_chunk(key, json.dumps(value)))
    if not chunks:
        return png_bytes

    # IHDR is always the first chunk; insert ours immediately after it.
    ihdr_length = struct.unpack(">I", png_bytes[8:12])[0]
    ihdr_end = 8 + 8 + ihdr_length + 4  # signature + (len+type) + data + crc
    return png_bytes[:ihdr_end] + b"".join(chunks) + png_bytes[ihdr_end:]


# Standard chromaticities (CIE 1931 xy) for the colorspaces this node writes.
# Each tuple is (Rx, Ry, Gx, Gy, Bx, By, Wx, Wy). All share D65 white point.
_CHROMATICITIES = {
    # ITU-R BT.709 / sRGB primaries
    "Rec.709":  (0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290),
    # ITU-R BT.2020 (UHDTV / wide-gamut HDR) primaries
    "Rec.2020": (0.7080, 0.2920, 0.1700, 0.7970, 0.1310, 0.0460, 0.3127, 0.3290),
}


def _pack_chromaticities(primaries: tuple) -> bytes:
    """Serialize 8 chromaticity floats into the EXR `chromaticities` payload."""
    return struct.pack("<8f", *primaries)


def _exr_attribute(name: str, attr_type: str, value: bytes) -> bytes:
    """Serialize one EXR header attribute: name\\0 type\\0 size:int32 value."""
    return (
        name.encode("utf-8") + b"\x00"
        + attr_type.encode("utf-8") + b"\x00"
        + struct.pack("<i", len(value))
        + value
    )


def inject_exr_metadata(
    exr_bytes: bytes,
    prompt: dict | None,
    extra_pnginfo: dict | None,
    colorspace: str | None = None,
) -> bytes:
    """Insert ComfyUI metadata and color-space info into an EXR header.

    Color: EXR pixels are linear by convention. The standard way to describe
    their RGB→XYZ relationship is the `chromaticities` attribute. We pick the
    primaries that match what the user told us their input was:

      colorspace="sRGB" → Rec. 709 / sRGB primaries (D65)
      colorspace="HDR"  → Rec. 2020 / BT.2100 primaries (D65)

    Pixels are always converted to linear scene light upstream (sRGB EOTF
    inverse for sRGB; HLG OETF inverse for HDR), so the file content is
    scene-linear in the indicated gamut. OpenEXR has no standard transfer-
    function attribute (the OpenEXR TSC has discussed adding one but it
    doesn't exist), so we don't invent one — `chromaticities` plus the EXR
    linear-by-convention rule fully specifies the color.

    Prompt/workflow: written as plain `string` attributes using the same keys
    (`prompt`, `workflow`, ...) that Comfy uses for PNG tEXt chunks, so the
    same readers can pull them out symmetrically.

    Implementation note: the chunk-offset table that follows the header stores
    *absolute* byte offsets into the file. Inserting N bytes into the header
    means every offset must be incremented by N or the file becomes unreadable.
    """
    if len(exr_bytes) < 8 or exr_bytes[:4] != b"\x76\x2f\x31\x01":
        return exr_bytes

    new_blob = b""
    if prompt is not None:
        new_blob += _exr_attribute("prompt", "string", json.dumps(prompt).encode("utf-8"))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            new_blob += _exr_attribute(key, "string", json.dumps(value).encode("utf-8"))
    if colorspace is not None:
        # Map each colorspace option to the RGB primaries the linear pixels
        # are now in. "sRGB" and "linear" both produce Rec. 709 linear; "HDR"
        # (HLG-encoded Rec. 2020 input) produces Rec. 2020 linear.
        primaries_name = {
            "sRGB":   "Rec.709",
            "linear": "Rec.709",
            "HDR":    "Rec.2020",
        }.get(colorspace, "Rec.709")
        new_blob += _exr_attribute(
            "chromaticities",
            "chromaticities",
            _pack_chromaticities(_CHROMATICITIES[primaries_name]),
        )
    if not new_blob:
        return exr_bytes

    # Walk header attributes to find the terminating null byte, and pick up
    # dataWindow + compression so we know how many chunks the offset table has.
    pos = 8  # past magic (4) + version (4)
    data_window = None
    compression = 0
    while pos < len(exr_bytes) and exr_bytes[pos] != 0:
        name_end = exr_bytes.index(b"\x00", pos)
        attr_name = exr_bytes[pos:name_end].decode("latin-1", errors="replace")
        type_end = exr_bytes.index(b"\x00", name_end + 1)
        attr_type = exr_bytes[name_end + 1:type_end].decode("latin-1", errors="replace")
        size = struct.unpack("<i", exr_bytes[type_end + 1:type_end + 5])[0]
        value_start = type_end + 5
        value = exr_bytes[value_start:value_start + size]

        if attr_name == "dataWindow" and attr_type == "box2i":
            data_window = struct.unpack("<iiii", value)  # xMin, yMin, xMax, yMax
        elif attr_name == "compression" and attr_type == "compression":
            compression = value[0]

        pos = value_start + size

    if data_window is None:
        return exr_bytes  # required attribute missing — don't risk corrupting

    # Scanlines per chunk by compression, from the OpenEXR spec.
    scanlines_per_block = {
        0: 1,   # NO_COMPRESSION
        1: 1,   # RLE
        2: 1,   # ZIPS
        3: 16,  # ZIP
        4: 32,  # PIZ
        5: 16,  # PXR24
        6: 32,  # B44
        7: 32,  # B44A
        8: 256, # DWAA
        9: 256, # DWAB
    }.get(compression, 1)

    _, y_min, _, y_max = data_window
    height = y_max - y_min + 1
    num_chunks = (height + scanlines_per_block - 1) // scanlines_per_block

    header_end = pos  # position of the terminating null byte
    table_start = header_end + 1
    pixel_start = table_start + num_chunks * 8
    delta = len(new_blob)

    old_offsets = struct.unpack(f"<{num_chunks}Q", exr_bytes[table_start:pixel_start])
    new_table = struct.pack(f"<{num_chunks}Q", *(o + delta for o in old_offsets))

    return (
        exr_bytes[:header_end]                # header attributes
        + new_blob                            # our new attributes
        + exr_bytes[header_end:table_start]   # terminating null byte
        + new_table                           # shifted offset table
        + exr_bytes[pixel_start:]             # pixel data, untouched
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_image(
    img_tensor: torch.Tensor,
    file_format: str,
    bit_depth: str,
    colorspace: str,
) -> bytes:
    """Encode a single HxWxC tensor to PNG or EXR bytes in memory.

    For EXR the input is interpreted according to `colorspace` and converted
    to scene-linear (EXR's convention) before writing:

      "sRGB"   → input is sRGB-encoded Rec. 709; apply inverse sRGB EOTF.
      "HDR"    → input is HLG-encoded Rec. 2020 (BT.2100); apply inverse HLG
                 OETF to get scene-linear, per BT.2100 Note 5a.
      "linear" → input is already scene-linear (Rec. 709 primaries); write
                 through unchanged. Use this for renderer/compositor output.

    For PNG, colorspace selection does not modify pixels — PNG is delivered
    sRGB-encoded and there is no PNG path for wide-gamut HDR in this node.
    """
    height, width, num_channels = img_tensor.shape
    has_alpha = num_channels == 4

    spec = _FORMAT_SPECS[(file_format, bit_depth, has_alpha)]

    if spec["dtype"] == np.float32:
        # EXR path: preserve full range, no clamp.
        if colorspace == "sRGB":
            img_tensor = srgb_to_linear(img_tensor)
        elif colorspace == "HDR":
            img_tensor = hlg_to_linear(img_tensor)
        img_np = img_tensor.cpu().numpy().astype(np.float32)
    else:
        # PNG path: quantize to integer range.
        scaled = (img_tensor * spec["scale"]).clamp(0, spec["scale"])
        img_np = scaled.to(torch.int32).cpu().numpy().astype(spec["dtype"])

    # Encode directly via CodecContext. PyAV's `image2` muxer does NOT write to
    # BytesIO (it expects a real file path), so we bypass the container entirely.
    # For single-frame PNG/EXR the raw codec output IS the file.
    codec = av.CodecContext.create(file_format, "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = spec["stream_fmt"]
    codec.time_base = Fraction(1, 1)

    frame = av.VideoFrame.from_ndarray(img_np, format=spec["frame_fmt"])
    if spec["frame_fmt"] != spec["stream_fmt"]:
        frame = frame.reformat(format=spec["stream_fmt"])
    frame.pts = 0
    frame.time_base = codec.time_base

    packets = list(codec.encode(frame)) + list(codec.encode(None))  # flush with None
    return b"".join(bytes(p) for p in packets)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SaveImageAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveImageAdvanced",
            search_aliases=["save", "save image", "export image", "output image", "write image"],
            display_name="Save Image (Advanced)",
            description="Saves the input images to your ComfyUI output directory.",
            category="image",
            essentials_category="Basics",
            inputs=[
                IO.Image.Input("images", tooltip="The images to save."),
                IO.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=("The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd% or %Empty Latent Image.width%."),
                ),
                IO.DynamicCombo.Input(
                    "format",
                    options=[
                        IO.DynamicCombo.Option("png", [
                            IO.Combo.Input("bit_depth", options=["8-bit", "16-bit"], default="8-bit", advanced=True),
                            IO.Combo.Input("input_color_space", options=["sRGB"], default="sRGB", advanced=True),
                        ]),
                        IO.DynamicCombo.Option("exr", [
                            IO.Combo.Input("bit_depth", options=["32-bit float"], default="32-bit float", advanced=True),
                            IO.Combo.Input(
                                "input_color_space",
                                options=["sRGB", "HDR", "linear"],
                                default="sRGB",
                                advanced=True,
                                tooltip=(
                                    "Colorspace of the input tensor. The EXR is always written as scene-linear in the matching gamut.\n"
                                    "sRGB — input is sRGB-encoded Rec.709; the inverse sRGB EOTF is applied.\n"
                                    "HDR — input is HLG-encoded Rec.2020 (BT.2100); the inverse HLG OETF is applied to get scene-linear light.\n"
                                    "linear — input is already scene-linear (Rec.709 primaries); written through unchanged. Use this for renderer/compositor output."
                                ),
                            ),
                        ]),
                    ],
                    tooltip="The file format in which to save the image.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images")]
        )

    @classmethod
    def execute(cls, images, filename_prefix: str, format: dict) -> IO.NodeOutput:
        file_format = format["format"]
        bit_depth = format["bit_depth"]
        colorspace = format.get("input_color_space", "sRGB")

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        write_metadata = not args.disable_metadata

        results = []
        for batch_number, image in enumerate(images):
            encoded = _encode_image(image, file_format, bit_depth, colorspace)

            if write_metadata:
                if file_format == "png":
                    encoded = inject_png_metadata(encoded, prompt, extra_pnginfo)
                elif file_format == "exr":
                    encoded = inject_exr_metadata(encoded, prompt, extra_pnginfo, colorspace)

            name = filename.replace("%batch_num%", str(batch_number))
            file = f"{name}_{counter:05}.{file_format}"
            with open(os.path.join(full_output_folder, file), "wb") as f:
                f.write(encoded)

            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1

        return IO.NodeOutput(images, ui={"images": results})


class ImagesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            ImageCrop,
            ImageCropV2,
            BoundingBox,
            RepeatImageBatch,
            ImageFromBatch,
            ImageAddNoise,
            SaveAnimatedWEBP,
            SaveAnimatedPNG,
            SaveImageAdvanced,
            SaveSVGNode,
            ImageStitch,
            ResizeAndPadImage,
            GetImageSize,
            ImageRotate,
            ImageFlip,
            ImageScaleToMaxDimension,
            SplitImageToTileList,
            ImageMergeTileList,
        ]


async def comfy_entrypoint() -> ImagesExtension:
    return ImagesExtension()
