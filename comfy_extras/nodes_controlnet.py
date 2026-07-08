from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import nodes
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class SetUnionControlNetType(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SetUnionControlNetType",
            search_aliases=["set controlnet type", "union controlnet type"],
            display_name="Set Union ControlNet Type",
            category="model/conditioning/controlnet",
            inputs=[
                io.ControlNet.Input("control_net"),
                io.Combo.Input("type", options=["auto"] + list(UNION_CONTROLNET_TYPES.keys())),
            ],
            outputs=[
                io.ControlNet.Output(),
            ],
        )

    @classmethod
    def execute(cls, control_net, type) -> io.NodeOutput:
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return io.NodeOutput(control_net)

    set_controlnet_type = execute  # TODO: remove


class ControlNetInpaintingAliMamaApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ControlNetInpaintingAliMamaApply",
            search_aliases=["masked controlnet"],
            display_name="Apply ControlNet Inpainting (AliMama)",
            category="model/conditioning/controlnet",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.ControlNet.Input("control_net"),
                io.Vae.Input("vae"),
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, control_net, vae, image, mask, strength, start_percent, end_percent) -> io.NodeOutput:
        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
            image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
            extra_concat = [mask]

        result = nodes.ControlNetApplyAdvanced().apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent, vae=vae, extra_concat=extra_concat)
        return io.NodeOutput(result[0], result[1])

    apply_inpaint_controlnet = execute  # TODO: remove


class ControlNetExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SetUnionControlNetType,
            ControlNetInpaintingAliMamaApply,
        ]


async def comfy_entrypoint() -> ControlNetExtension:
    return ControlNetExtension()
