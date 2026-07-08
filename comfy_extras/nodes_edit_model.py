import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


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


class EditModelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ReferenceLatent,
        ]


def comfy_entrypoint() -> EditModelExtension:
    return EditModelExtension()
