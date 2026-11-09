import hashlib
import os

import numpy as np
import torch
from PIL import Image

import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io, UI
from typing_extensions import override


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0.0, 0.0, 0.0)
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


class PainterNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Painter",
            display_name="Painter",
            category="image",
            has_intermediate_output=True,
            inputs=[
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional base image to paint over",
                ),
                io.String.Input(
                    "mask",
                    default="",
                    socketless=True,
                    extra_dict={"widgetType": "PAINTER", "image_upload": True},
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=64,
                    socketless=True,
                    extra_dict={"hidden": True},
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=64,
                    socketless=True,
                    extra_dict={"hidden": True},
                ),
                io.Color.Input("bg_color", default="#000000"),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
            ],
        )

    @classmethod
    def execute(cls, mask, width, height, bg_color="#000000", image=None) -> io.NodeOutput:
        if image is not None:
            base_image = image[:1]
            h, w = base_image.shape[1], base_image.shape[2]
        else:
            h, w = height, width
            r, g, b = hex_to_rgb(bg_color)
            base_image = torch.zeros((1, h, w, 3), dtype=torch.float32)
            base_image[0, :, :, 0] = r
            base_image[0, :, :, 1] = g
            base_image[0, :, :, 2] = b

        if mask and mask.strip():
            mask_path = folder_paths.get_annotated_filepath(mask)
            painter_img = node_helpers.pillow(Image.open, mask_path)
            painter_img = painter_img.convert("RGBA")

            if painter_img.size != (w, h):
                painter_img = painter_img.resize((w, h), Image.LANCZOS)

            painter_np = np.array(painter_img).astype(np.float32) / 255.0
            painter_rgb = painter_np[:, :, :3]
            painter_alpha = painter_np[:, :, 3:4]

            mask_tensor = torch.from_numpy(painter_np[:, :, 3]).unsqueeze(0)

            base_np = base_image[0].cpu().numpy()
            composited = painter_rgb * painter_alpha + base_np * (1.0 - painter_alpha)
            out_image = torch.from_numpy(composited).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, h, w), dtype=torch.float32)
            out_image = base_image

        return io.NodeOutput(out_image, mask_tensor, ui=UI.PreviewImage(out_image))

    @classmethod
    def fingerprint_inputs(cls, mask, width, height, bg_color="#000000", image=None):
        if mask and mask.strip():
            mask_path = folder_paths.get_annotated_filepath(mask)
            if os.path.exists(mask_path):
                m = hashlib.sha256()
                with open(mask_path, "rb") as f:
                    m.update(f.read())
                return m.digest().hex()
        return ""



class PainterExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [PainterNode]


async def comfy_entrypoint():
    return PainterExtension()
