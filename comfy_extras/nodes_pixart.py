from typing_extensions import override
import nodes
from comfy_api.latest import ComfyExtension, io

class CLIPTextEncodePixArtAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodePixArtAlpha",
            display_name="CLIP Text Encode (PixArt Alpha)",
            search_aliases=["pixart prompt"],
            category="model/conditioning/pixart",
            description="Encodes text and sets the resolution conditioning for PixArt Alpha. Does not apply to PixArt Sigma.",
            inputs=[
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                # "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Clip.Input("clip"),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, width, height, text):
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height}))


class PixArtExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodePixArtAlpha,
        ]

async def comfy_entrypoint() -> PixArtExtension:
    return PixArtExtension()
