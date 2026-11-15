#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

from typing_extensions import override

import comfy.model_patcher
import comfy.samplers
from comfy_api.latest import ComfyExtension, io


class PerturbedAttentionGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PerturbedAttentionGuidance",
            category="model/patch/unet",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("scale", default=3.0, min=0.0, max=100.0, step=0.01, round=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, scale) -> io.NodeOutput:
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            return v

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            # Replace Self-attention with PAG
            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention, "attn1", unet_block, unet_block_id)
            (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - pag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)


class PAGExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PerturbedAttentionGuidance,
        ]


async def comfy_entrypoint() -> PAGExtension:
    return PAGExtension()
