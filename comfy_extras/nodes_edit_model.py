import math

import comfy.utils
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from nodes import EmptyLatentImage


class ReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReferenceLatent",
            display_name="Set Reference Latent",
            category="model/conditioning",
            description="This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, latent=None) -> io.NodeOutput:
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        return io.NodeOutput(conditioning)


class ReferenceMultiLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReferenceMultiLatent",
            display_name="Set Reference Multi Latent",
            category="model/conditioning",
            description="Encodes reference images and appends their latents to positive and optional negative conditioning for edit models.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative", optional=True),
                io.Vae.Input("vae", optional=True, tooltip="Reference images are ignored when no VAE is connected."),
                io.DynamicCombo.Input("resolution", options=[
                    io.DynamicCombo.Option("keep resolution", []),
                    io.DynamicCombo.Option("resize", [
                        io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01,
                                       tooltip="Target size of each reference image, preserving aspect ratio. 1.0 is about 1024 x 1024 pixels."),
                    ]),
                ]),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image", optional=True), prefix="image", min=1, max=100,
                ), optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative", tooltip="Returns the positive output when negative conditioning is not connected."),
                io.Latent.Output(display_name="empty_latent", tooltip="First reference latent multiplied by 0.0, or a square empty latent at the selected resolution."),
            ],
        )

    @classmethod
    def execute(cls, positive, resolution: io.DynamicCombo.Type, negative=None, vae=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        ref_latents = []
        if vae is not None:
            for name in sorted(images or {}, key=lambda name: int(name.removeprefix("image"))):
                image = images[name]
                if image is None:
                    continue
                if resolution["resolution"] == "resize":
                    height, width = image.shape[1:3]
                    scale = math.sqrt(resolution["megapixels"] * 1024 * 1024 / (width * height))
                    width = max(1, round(width * scale))
                    height = max(1, round(height * scale))
                    image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
                ref_latents.append(vae.encode(image))

        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            if negative is not None:
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
            empty_latent = {"samples": ref_latents[0] * 0.0}
        else:
            size = 1024
            if resolution["resolution"] == "resize":
                size = round(math.sqrt(resolution["megapixels"] * 1024 * 1024))
            empty_latent = EmptyLatentImage().generate(size, size)[0]
        return io.NodeOutput(positive, negative if negative is not None else positive, empty_latent)


class EditModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ReferenceLatent,
            ReferenceMultiLatent,
        ]


def comfy_entrypoint() -> EditModelExtension:
    return EditModelExtension()
