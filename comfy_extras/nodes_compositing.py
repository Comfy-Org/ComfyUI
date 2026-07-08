import torch
import comfy.utils
from enum import Enum
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

class PorterDuffMode(Enum):
    ADD = 0
    CLEAR = 1
    DARKEN = 2
    DST = 3
    DST_ATOP = 4
    DST_IN = 5
    DST_OUT = 6
    DST_OVER = 7
    LIGHTEN = 8
    MULTIPLY = 9
    OVERLAY = 10
    SCREEN = 11
    SRC = 12
    SRC_ATOP = 13
    SRC_IN = 14
    SRC_OUT = 15
    SRC_OVER = 16
    XOR = 17


def porter_duff_composite(src_image: torch.Tensor, src_alpha: torch.Tensor, dst_image: torch.Tensor, dst_alpha: torch.Tensor, mode: PorterDuffMode):
    # convert mask to alpha
    src_alpha = 1 - src_alpha
    dst_alpha = 1 - dst_alpha
    # premultiply alpha
    src_image = src_image * src_alpha
    dst_image = dst_image * dst_alpha

    # composite ops below assume alpha-premultiplied images
    if mode == PorterDuffMode.ADD:
        out_alpha = torch.clamp(src_alpha + dst_alpha, 0, 1)
        out_image = torch.clamp(src_image + dst_image, 0, 1)
    elif mode == PorterDuffMode.CLEAR:
        out_alpha = torch.zeros_like(dst_alpha)
        out_image = torch.zeros_like(dst_image)
    elif mode == PorterDuffMode.DARKEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image + torch.min(src_image, dst_image)
    elif mode == PorterDuffMode.DST:
        out_alpha = dst_alpha
        out_image = dst_image
    elif mode == PorterDuffMode.DST_ATOP:
        out_alpha = src_alpha
        out_image = src_alpha * dst_image + (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.DST_IN:
        out_alpha = src_alpha * dst_alpha
        out_image = dst_image * src_alpha
    elif mode == PorterDuffMode.DST_OUT:
        out_alpha = (1 - src_alpha) * dst_alpha
        out_image = (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.DST_OVER:
        out_alpha = dst_alpha + (1 - dst_alpha) * src_alpha
        out_image = dst_image + (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.LIGHTEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image + torch.max(src_image, dst_image)
    elif mode == PorterDuffMode.MULTIPLY:
        out_alpha = src_alpha * dst_alpha
        out_image = src_image * dst_image
    elif mode == PorterDuffMode.OVERLAY:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = torch.where(2 * dst_image < dst_alpha, 2 * src_image * dst_image,
            src_alpha * dst_alpha - 2 * (dst_alpha - src_image) * (src_alpha - dst_image))
    elif mode == PorterDuffMode.SCREEN:
        out_alpha = src_alpha + dst_alpha - src_alpha * dst_alpha
        out_image = src_image + dst_image - src_image * dst_image
    elif mode == PorterDuffMode.SRC:
        out_alpha = src_alpha
        out_image = src_image
    elif mode == PorterDuffMode.SRC_ATOP:
        out_alpha = dst_alpha
        out_image = dst_alpha * src_image + (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.SRC_IN:
        out_alpha = src_alpha * dst_alpha
        out_image = src_image * dst_alpha
    elif mode == PorterDuffMode.SRC_OUT:
        out_alpha = (1 - dst_alpha) * src_alpha
        out_image = (1 - dst_alpha) * src_image
    elif mode == PorterDuffMode.SRC_OVER:
        out_alpha = src_alpha + (1 - src_alpha) * dst_alpha
        out_image = src_image + (1 - src_alpha) * dst_image
    elif mode == PorterDuffMode.XOR:
        out_alpha = (1 - dst_alpha) * src_alpha + (1 - src_alpha) * dst_alpha
        out_image = (1 - dst_alpha) * src_image + (1 - src_alpha) * dst_image
    else:
        return None, None

    # back to non-premultiplied alpha
    out_image = torch.where(out_alpha > 1e-5, out_image / out_alpha, torch.zeros_like(out_image))
    out_image = torch.clamp(out_image, 0, 1)
    # convert alpha to mask
    out_alpha = 1 - out_alpha
    return out_image, out_alpha


class PorterDuffImageComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PorterDuffImageComposite",
            search_aliases=["alpha composite", "blend modes", "layer blend", "transparency blend"],
            display_name="Porter-Duff Image Composite",
            category="image/compositing",
            inputs=[
                io.Image.Input("source"),
                io.Mask.Input("source_alpha"),
                io.Image.Input("destination"),
                io.Mask.Input("destination_alpha"),
                io.Combo.Input("mode", options=[mode.name for mode in PorterDuffMode], default=PorterDuffMode.DST.name),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def execute(cls, source: torch.Tensor, source_alpha: torch.Tensor, destination: torch.Tensor, destination_alpha: torch.Tensor, mode) -> io.NodeOutput:
        batch_size = min(len(source), len(source_alpha), len(destination), len(destination_alpha))
        out_images = []
        out_alphas = []

        for i in range(batch_size):
            src_image = source[i]
            dst_image = destination[i]

            assert src_image.shape[2] == dst_image.shape[2] # inputs need to have same number of channels

            src_alpha = source_alpha[i].unsqueeze(2)
            dst_alpha = destination_alpha[i].unsqueeze(2)

            if dst_alpha.shape[:2] != dst_image.shape[:2]:
                upscale_input = dst_alpha.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_image.shape[1], dst_image.shape[0], upscale_method='bicubic', crop='center')
                dst_alpha = upscale_output.permute(0, 2, 3, 1).squeeze(0)
            if src_image.shape != dst_image.shape:
                upscale_input = src_image.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_image.shape[1], dst_image.shape[0], upscale_method='bicubic', crop='center')
                src_image = upscale_output.permute(0, 2, 3, 1).squeeze(0)
            if src_alpha.shape != dst_alpha.shape:
                upscale_input = src_alpha.unsqueeze(0).permute(0, 3, 1, 2)
                upscale_output = comfy.utils.common_upscale(upscale_input, dst_alpha.shape[1], dst_alpha.shape[0], upscale_method='bicubic', crop='center')
                src_alpha = upscale_output.permute(0, 2, 3, 1).squeeze(0)

            out_image, out_alpha = porter_duff_composite(src_image, src_alpha, dst_image, dst_alpha, PorterDuffMode[mode])

            out_images.append(out_image)
            out_alphas.append(out_alpha.squeeze(2))

        return io.NodeOutput(torch.stack(out_images), torch.stack(out_alphas))


class SplitImageWithAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SplitImageWithAlpha",
            search_aliases=["extract alpha", "separate transparency", "remove alpha"],
            display_name="Split Image with Alpha",
            category="image/compositing",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        out_images = [i[:,:,:3] for i in image]
        out_alphas = [i[:,:,3] if i.shape[2] > 3 else torch.ones_like(i[:,:,0]) for i in image]
        return io.NodeOutput(torch.stack(out_images), 1.0 - torch.stack(out_alphas))


class JoinImageWithAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JoinImageWithAlpha",
            search_aliases=["add transparency", "apply alpha", "composite alpha", "RGBA"],
            display_name="Join Image with Alpha",
            category="image/compositing",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("alpha"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, alpha: torch.Tensor) -> io.NodeOutput:
        batch_size = max(len(image), len(alpha))
        alpha = 1.0 - resize_mask(alpha.to(image), image.shape[1:])
        alpha = comfy.utils.repeat_to_batch_size(alpha, batch_size)
        image = comfy.utils.repeat_to_batch_size(image, batch_size)
        return io.NodeOutput(torch.cat((image[..., :3], alpha.unsqueeze(-1)), dim=-1))


class CompositingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PorterDuffImageComposite,
            SplitImageWithAlpha,
            JoinImageWithAlpha,
        ]


async def comfy_entrypoint() -> CompositingExtension:
    return CompositingExtension()
