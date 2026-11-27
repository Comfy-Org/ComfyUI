from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_tweak_wrapper


class TorchTweakModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TorchTweakModel",
            category="experimental",
            description="Measures OpenVINO, Torch Inductor, graphs and eager on the first step of each image size and keeps the fastest. Later runs with the same model reuse the choice. Needs the torch-tweak package.",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone(disable_dynamic=True)
        set_torch_tweak_wrapper(model=m)
        return io.NodeOutput(m)


class TorchTweakExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchTweakModel,
        ]


async def comfy_entrypoint() -> TorchTweakExtension:
    return TorchTweakExtension()
