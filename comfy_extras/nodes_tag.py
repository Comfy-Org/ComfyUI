# TAG: Tangential Amplifying Guidance - (arXiv: https://arxiv.org/pdf/2510.04533)

from typing_extensions import override
import torch

from comfy_api.latest import ComfyExtension, io

class TAGGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TAG-Guidance",
            display_name="Tangential Amplifying Guidance",
            category="advanced/guidance",
            description="TAG - Tangential Amplifying Guidance (2510.04533)\n\nLeverages an intermediate sample as a projection basis and amplifies the tangential components of the estimated scores with respect to this basis to correct the sampling trajectory. Improves diffusion sampling fidelity with minimal computational addition",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("t_guidance_scale", default=1.0, min=0.0, max=20.0, step=0.05),
                io.Float.Input("r_guidance_scale", default=1.0, min=0.0, max=20.0, step=0.05),
            ],
            outputs=[
                io.Model.Output(display_name="patched_model"),
            ],
        )

    @classmethod
    def execute(cls, model, t_guidance_scale, r_guidance_scale):
        m = model.clone()

        def tag_guidance(args):

            post_latents = args['input'] 
            v_t_2d       = post_latents / (post_latents.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8)

            latents = args['denoised']

            delta_latents = latents - post_latents
            delta_unit    = (delta_latents * v_t_2d).sum(dim=(1,2,3), keepdim=True)

            normal_update_vector     = delta_unit * v_t_2d
            tangential_update_vector = delta_latents - normal_update_vector

            eta_v = t_guidance_scale
            eta_n = r_guidance_scale

            latents = post_latents + \
                eta_v * tangential_update_vector + \
                eta_n * normal_update_vector
            
            return latents

        m.set_model_sampler_post_cfg_function(tag_guidance)
        return io.NodeOutput(m)


class TagExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TAGGuidance,
        ]


async def comfy_entrypoint() -> TagExtension:
    return TagExtension()
