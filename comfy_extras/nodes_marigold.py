from typing_extensions import override

import torch
import torch.nn.functional as F

from comfy_api.latest import ComfyExtension, io


class MarigoldV2PostProcess(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MarigoldV2PostProcess",
            display_name="Marigold V2 Post-Process",
            category="image/geometry estimation",
            description="Turns a decoded Marigold V2 prediction into an image: normalized depth with near as bright, unit surface normals, or sRGB albedo.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("prediction", options=["depth", "normals", "albedo"]),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, prediction) -> io.NodeOutput:
        if prediction == "normals":
            return io.NodeOutput((F.normalize(image * 2.0 - 1.0, dim=-1) + 1.0) * 0.5)
        if prediction == "albedo":
            return io.NodeOutput(torch.where(image <= 0.0031308, image * 12.92, 1.055 * image.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055))
        d = image.mean(dim=-1, keepdim=True)
        lo = d.amin(dim=(1, 2, 3), keepdim=True)
        hi = d.amax(dim=(1, 2, 3), keepdim=True)
        return io.NodeOutput(((hi - d) / (hi - lo).clamp(min=1e-6)).repeat(1, 1, 1, 3))


class MarigoldExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MarigoldV2PostProcess,
        ]


async def comfy_entrypoint() -> MarigoldExtension:
    return MarigoldExtension()
