from typing_extensions import override
import nodes
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats

from comfy_api.latest import ComfyExtension, io


class EmptyCosmosLatentVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyCosmosLatentVideo",
            category="model/latent/cosmos",
            inputs=[
                io.Int.Input("width", default=1280, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=704, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=121, min=1, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, length, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return io.NodeOutput({"samples": latent})


def vae_encode_with_padding(vae, image, width, height, length, padding=0):
    pixels = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    pixel_len = min(pixels.shape[0], length)
    padded_length = min(length, (((pixel_len - 1) // 8) + 1 + padding) * 8 - 7)
    padded_pixels = torch.ones((padded_length, height, width, 3)) * 0.5
    padded_pixels[:pixel_len] = pixels[:pixel_len]
    latent_len = ((pixel_len - 1) // 8) + 1
    latent_temp = vae.encode(padded_pixels)
    return latent_temp[:, :, :latent_len]


class CosmosImageToVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CosmosImageToVideoLatent",
            category="model/conditioning/cosmos",
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input("width", default=1280, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=704, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=121, min=1, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, width, height, length, batch_size, start_image=None, end_image=None) -> io.NodeOutput:
        latent = torch.zeros([1, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is None and end_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return io.NodeOutput(out_latent)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 8) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            latent_temp = vae_encode_with_padding(vae, start_image, width, height, length, padding=1)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        if end_image is not None:
            latent_temp = vae_encode_with_padding(vae, end_image, width, height, length, padding=0)
            latent[:, :, -latent_temp.shape[-3]:] = latent_temp
            mask[:, :, -latent_temp.shape[-3]:] *= 0.0

        out_latent = {}
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return io.NodeOutput(out_latent)

class CosmosPredict2ImageToVideoLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CosmosPredict2ImageToVideoLatent",
            category="model/conditioning/cosmos",
            inputs=[
                io.Vae.Input("vae"),
                io.Int.Input("width", default=848, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=93, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("end_image", optional=True),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, width, height, length, batch_size, start_image=None, end_image=None) -> io.NodeOutput:
        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is None and end_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return io.NodeOutput(out_latent)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            latent_temp = vae_encode_with_padding(vae, start_image, width, height, length, padding=1)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        if end_image is not None:
            latent_temp = vae_encode_with_padding(vae, end_image, width, height, length, padding=0)
            latent[:, :, -latent_temp.shape[-3]:] = latent_temp
            mask[:, :, -latent_temp.shape[-3]:] *= 0.0

        out_latent = {}
        latent_format = comfy.latent_formats.Wan21()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return io.NodeOutput(out_latent)


class CosmosExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EmptyCosmosLatentVideo,
            CosmosImageToVideoLatent,
            CosmosPredict2ImageToVideoLatent,
        ]


async def comfy_entrypoint() -> CosmosExtension:
    return CosmosExtension()
