from typing_extensions import override

import comfy.utils
import node_helpers
import torch
import comfy.model_management
from comfy_api.latest import ComfyExtension, io


class TextEncodeMageFlowEdit(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeMageFlowEdit",
            category="model/conditioning/mage",
            description="Encode an edit instruction with one or more reference images for Mage-Flow-Edit. Reference latents are resized to the output resolution (width/height, or the first image's size when 0). Use the latent output for sampling so the sizes always match.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, advanced=True),
                io.Vae.Input("vae", optional=True),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference image(s) to edit. All references are resized to the output resolution before encoding.",
                ),
                io.Int.Input("width", default=0, min=0, max=8192, step=16, tooltip="Output width. 0 = use the first reference image's size."),
                io.Int.Input("height", default=0, min=0, max=8192, step=16, tooltip="Output height. 0 = use the first reference image's size."),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt="", vae=None, images: io.Autogrow.Type = None, width=0, height=0, batch_size=1) -> io.NodeOutput:
        ref_latents = []
        images = images or {}
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]
        images_vl = []

        # Output resolution: explicit width/height, else the primary reference's own size, floored to /16.
        # Each dimension falls back independently so a 0 on one axis keeps an explicit value on the other.
        if width == 0 or height == 0:
            if len(images) > 0:
                ref_h, ref_w = images[0].shape[1], images[0].shape[2]
            else:
                ref_h, ref_w = 1024, 1024
            height = height or ref_h
            width = width or ref_w
        width = max(16, (width // 16) * 16)
        height = max(16, (height // 16) * 16)

        for image in images:
            samples = image.movedim(-1, 1)

            # VL conditioning copy: cap the long edge at 384 (training preprocessing).
            long_edge = max(samples.shape[3], samples.shape[2])
            if long_edge > 384:
                scale_by = 384 / long_edge
                s = comfy.utils.common_upscale(samples, max(1, round(samples.shape[3] * scale_by)), max(1, round(samples.shape[2] * scale_by)), "bicubic", "disabled")
                images_vl.append(s.movedim(1, -1))
            else:
                images_vl.append(image)

            if vae is not None:
                # All references are resized to the output resolution before encoding, because Mage's RoPE aligns reference and target content by position
                if samples.shape[3] != width or samples.shape[2] != height:
                    s = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
                else:
                    s = samples
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        # Negative branch keeps the same reference images (VL tokens + ref latents), only the instruction differs.
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt if negative_prompt else " ", images=images_vl))

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        latent = torch.zeros([batch_size, 128, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})


class MageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeMageFlowEdit,
        ]


async def comfy_entrypoint() -> MageExtension:
    return MageExtension()
