import torch
import torch.nn.functional as F
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import kornia.color


def _max_pool_dilate(tensor, kernel_size):
    pad = kernel_size // 2
    return F.max_pool2d(tensor, kernel_size, stride=1, padding=pad)


def _max_pool_erode(tensor, kernel_size):
    pad = kernel_size // 2
    return -F.max_pool2d(-tensor, kernel_size, stride=1, padding=pad)


class Morphology(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Morphology",
            search_aliases=["erode", "dilate"],
            display_name="Apply Morphology",
            category="image/filters",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "operation",
                    options=["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"],
                ),
                io.Int.Input("kernel_size", default=3, min=3, max=999, step=1),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, operation, kernel_size) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        image_k = image.to(device).movedim(-1, 1)
        if operation == "erode":
            output = _max_pool_erode(image_k, kernel_size)
        elif operation == "dilate":
            output = _max_pool_dilate(image_k, kernel_size)
        elif operation == "open":
            output = _max_pool_dilate(_max_pool_erode(image_k, kernel_size), kernel_size)
        elif operation == "close":
            output = _max_pool_erode(_max_pool_dilate(image_k, kernel_size), kernel_size)
        elif operation == "gradient":
            output = _max_pool_dilate(image_k, kernel_size) - _max_pool_erode(image_k, kernel_size)
        elif operation == "top_hat":
            output = image_k - _max_pool_dilate(_max_pool_erode(image_k, kernel_size), kernel_size)
        elif operation == "bottom_hat":
            output = _max_pool_erode(_max_pool_dilate(image_k, kernel_size), kernel_size) - image_k
        else:
            raise ValueError(f"Invalid operation {operation} for morphology. Must be one of 'erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'bottomhat'")
        img_out = output.to(comfy.model_management.intermediate_device()).movedim(1, -1)
        return io.NodeOutput(img_out)


class ImageRGBToYUV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageRGBToYUV",
            search_aliases=["color space conversion"],
            display_name="Image RGB to YUV",
            category="image/color",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(display_name="Y"),
                io.Image.Output(display_name="U"),
                io.Image.Output(display_name="V"),
            ],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        image = image[..., :3]
        out = kornia.color.rgb_to_ycbcr(image.movedim(-1, 1)).movedim(1, -1)
        return io.NodeOutput(out[..., 0:1].expand_as(image), out[..., 1:2].expand_as(image), out[..., 2:3].expand_as(image))

class ImageYUVToRGB(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageYUVToRGB",
            search_aliases=["color space conversion"],
            display_name="Image YUV to RGB",
            category="image/color",
            inputs=[
                io.Image.Input("Y"),
                io.Image.Input("U"),
                io.Image.Input("V"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, Y, U, V) -> io.NodeOutput:
        image = torch.cat([torch.mean(Y[..., :3], dim=-1, keepdim=True), torch.mean(U[..., :3], dim=-1, keepdim=True), torch.mean(V[..., :3], dim=-1, keepdim=True)], dim=-1)
        out = kornia.color.ycbcr_to_rgb(image.movedim(-1, 1)).movedim(1, -1)
        return io.NodeOutput(out)


class MorphologyExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Morphology,
            ImageRGBToYUV,
            ImageYUVToRGB,
        ]


async def comfy_entrypoint() -> MorphologyExtension:
    return MorphologyExtension()

