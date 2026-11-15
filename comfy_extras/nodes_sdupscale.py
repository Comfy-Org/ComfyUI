from typing_extensions import override

import torch
import comfy.utils
from comfy_api.latest import ComfyExtension, io

class SD_4XUpscale_Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SD_4XUpscale_Conditioning",
            category="model/conditioning/stable diffusion upscaler",
            inputs=[
                io.Image.Input("images"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("scale_ratio", default=4.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("noise_augmentation", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, images, positive, negative, scale_ratio, noise_augmentation):
        width = max(1, round(images.shape[-2] * scale_ratio))
        height = max(1, round(images.shape[-3] * scale_ratio))

        pixels = comfy.utils.common_upscale((images.movedim(-1,1) * 2.0) - 1.0, width // 4, height // 4, "bilinear", "center")

        out_cp = []
        out_cn = []

        for t in positive:
            n = [t[0], t[1].copy()]
            n[1]['concat_image'] = pixels
            n[1]['noise_augmentation'] = noise_augmentation
            out_cp.append(n)

        for t in negative:
            n = [t[0], t[1].copy()]
            n[1]['concat_image'] = pixels
            n[1]['noise_augmentation'] = noise_augmentation
            out_cn.append(n)

        latent = torch.zeros([images.shape[0], 4, height // 4, width // 4])
        return io.NodeOutput(out_cp, out_cn, {"samples":latent})


class SdUpscaleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SD_4XUpscale_Conditioning,
        ]


async def comfy_entrypoint() -> SdUpscaleExtension:
    return SdUpscaleExtension()
