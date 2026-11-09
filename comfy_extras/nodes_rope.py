from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class ScaleROPE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ScaleROPE",
            category="model/patch",
            description="Scale and shift the ROPE of the model.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale_x", default=1.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("shift_x", default=0.0, min=-256.0, max=256.0, step=0.1, advanced=True),

                io.Float.Input("scale_y", default=1.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("shift_y", default=0.0, min=-256.0, max=256.0, step=0.1, advanced=True),

                io.Float.Input("scale_t", default=1.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("shift_t", default=0.0, min=-256.0, max=256.0, step=0.1, advanced=True),


            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, scale_x, shift_x, scale_y, shift_y, scale_t, shift_t) -> io.NodeOutput:
        m = model.clone()
        m.set_model_rope_options(scale_x, shift_x, scale_y, shift_y, scale_t, shift_t)
        return io.NodeOutput(m)


class RopeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ScaleROPE
        ]


async def comfy_entrypoint() -> RopeExtension:
    return RopeExtension()
