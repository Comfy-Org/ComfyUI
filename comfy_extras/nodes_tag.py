# References:
# https://hyeon-cho.github.io/TAG/
# https://arxiv.org/abs/2510.04533
# https://github.com/hyeon-cho/Tangential-Amplifying-Guidance
from typing_extensions import override
from comfy_api.latest import io, ComfyExtension


# Algorithm 2: Tangential Amplifying Guidance
def tangential_amplify(x, prev_x, t_scale, r_scale):
    dims = tuple(dim + 1 for dim, size in enumerate(x.shape[1:]) if size > 1)
    v_r = prev_x / (prev_x.norm(p=2, dim=dims, keepdim=True) + 1e-8)
    dx = x - prev_x
    radial_update = (dx * v_r).sum(dim=dims, keepdim=True) * v_r
    tangential_update = dx - radial_update
    return prev_x + t_scale * tangential_update + r_scale * radial_update


class TangentialAmplifyingGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TangentialAmplifyingGuidance",
            display_name="Tangential Amplifying Guidance",
            category="experimental",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("t_scale", default=1.05, min=0.0, max=10.0, step=0.001, tooltip="scale for the tangential (orthogonal) component"),
                io.Float.Input("r_scale", default=1.0, min=0.0, max=10.0, step=0.001, tooltip="scale for the radial (parallel) component"),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.Model.Output()
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, t_scale, r_scale, start_percent, end_percent) -> io.NodeOutput:
        ms = model.get_model_object("model_sampling")
        sigma_hi = ms.percent_to_sigma(start_percent)
        sigma_lo = ms.percent_to_sigma(end_percent)

        def post_cfg_function(args):
            if not (sigma_lo <= args["sigma"].flatten()[0] <= sigma_hi):
                return args["denoised"]
            return tangential_amplify(args["denoised"], args["input"], t_scale, r_scale)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)
        return io.NodeOutput(m)


class TangentialAmplifyingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TangentialAmplifyingGuidance,
        ]


async def comfy_entrypoint() -> TangentialAmplifyingExtension:
    return TangentialAmplifyingExtension()
