from typing_extensions import override

import nodes
from comfy_api.latest import ComfyExtension, io


class CLIPTextEncodeSDXLRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeSDXLRefiner",
            display_name="CLIP Text Encode (SDXL Refiner)",
            category="model/conditioning/stable diffusion",
            inputs=[
                io.Float.Input("ascore", default=6.0, min=0.0, max=1000.0, step=0.01),
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
                io.Clip.Input("clip"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, ascore, width, height, text) -> io.NodeOutput:
        tokens = clip.tokenize(text)
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"aesthetic_score": ascore, "width": width, "height": height}))

class CLIPTextEncodeSDXL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncodeSDXL",
            display_name="CLIP Text Encode (SDXL)",
            category="model/conditioning/stable diffusion",
            inputs=[
                io.Clip.Input("clip"),
                io.Int.Input("width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("crop_w", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True),
                io.Int.Input("crop_h", default=0, min=0, max=nodes.MAX_RESOLUTION, advanced=True),
                io.Int.Input("target_width", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.Int.Input("target_height", default=1024, min=0, max=nodes.MAX_RESOLUTION),
                io.String.Input("text_g", multiline=True, dynamic_prompts=True),
                io.String.Input("text_l", multiline=True, dynamic_prompts=True),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l) -> io.NodeOutput:
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}))


class ClipSdxlExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CLIPTextEncodeSDXLRefiner,
            CLIPTextEncodeSDXL,
        ]


async def comfy_entrypoint() -> ClipSdxlExtension:
    return ClipSdxlExtension()
