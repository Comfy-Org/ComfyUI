from typing_extensions import override

from comfy.ldm.sensenova.sampling import SenseNovaModelSampling
from comfy_api.latest import ComfyExtension, io


class SenseNovaSamplingOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SenseNovaSamplingOptions",
            display_name="SenseNova Sampling Options",
            category="model/patch/sensenova",
            description="Set the SenseNova flow shift.",
            inputs=[
                io.Model.Input(id="model"),
                io.Float.Input(id="shift", default=3.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, *, model, shift: float) -> io.NodeOutput:
        patched = model.clone()
        model_sampling = SenseNovaModelSampling(patched.model.model_config)
        model_sampling.set_parameters(shift=shift)
        patched.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(patched)


class SenseNovaExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNovaSamplingOptions,
        ]


async def comfy_entrypoint() -> SenseNovaExtension:
    return SenseNovaExtension()
