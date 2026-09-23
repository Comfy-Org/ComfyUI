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
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae, image) -> io.NodeOutput:
        tokens = clip.tokenize(prompt, images=[image[:, :, :, :3]])
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [vae.encode(image)]}, append=True)
        return io.NodeOutput(conditioning)


class MingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeMingImageEdit,
        ]


async def comfy_entrypoint() -> MingExtension:
    return MingExtension()
