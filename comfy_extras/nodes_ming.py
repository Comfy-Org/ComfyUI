import comfy.utils
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class TextEncodeMingImageEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeMingImageEdit",
            display_name="Text Encode Ming Image Edit",
            category="model/conditioning/ming image",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae", optional=True, tooltip="Without a VAE the images only condition the text encoder through the vision tower."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 9)],
                        min=0,
                    ),
                    tooltip="Optional reference images, seen by the text encoder and appended to the latent sequence as clean frames. Later images are resized to the first one, and the sampled latent should match its size.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, images: io.Autogrow.Type, vae=None) -> io.NodeOutput:
        images = [images[name] for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])) if images[name] is not None]
        tokens = clip.tokenize(prompt, images=[image[:, :, :, :3] for image in images])
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if vae is not None and len(images) > 0:
            height, width = images[0].shape[1:3]  # every reference frame shares the canvas of the first one, as the vendor processor does
            ref_latents = [vae.encode(comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "disabled").movedim(1, -1)) for image in images]
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)


class MingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeMingImageEdit,
        ]


async def comfy_entrypoint() -> MingExtension:
    return MingExtension()
