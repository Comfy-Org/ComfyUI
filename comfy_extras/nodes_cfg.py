from typing_extensions import override

import torch

from comfy_api.latest import ComfyExtension, io


# https://github.com/WeichenFan/CFG-Zero-star
def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))

class CFGZeroStar(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CFGZeroStar",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
            ],
            outputs=[io.Model.Output(display_name="patched_model")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        def cfg_zero_star(args):
            guidance_scale = args['cond_scale']
            x = args['input']
            cond_p = args['cond_denoised']
            uncond_p = args['uncond_denoised']
            out = args["denoised"]
            alpha = optimized_scale(x - cond_p, x - uncond_p)

            return out + uncond_p * (alpha - 1.0)  + guidance_scale * uncond_p * (1.0 - alpha)
        m.set_model_sampler_post_cfg_function(cfg_zero_star)
        return io.NodeOutput(m)

class CFGNorm(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CFGNorm",
            category="advanced/guidance",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=1.0, min=0.0, max=100.0, step=0.01),
                io.Boolean.Input(
                    "pre_cfg",
                    default=False,
                    optional=True,
                    tooltip=(
                        "If true, rescale the combined noise BEFORE the sampler's CFG combine, "
                        "without clamping (can amplify). Matches the norm-scaled CFG used by "
                        "models like Lens. Default false keeps the original post-CFG x0-space "
                        "attenuate-only behavior."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="patched_model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, strength, pre_cfg=False) -> io.NodeOutput:
        m = model.clone()
        if pre_cfg:
            def cfg_norm_pre(args):
                cond = args["cond"]
                uncond = args["uncond"]
                cond_scale = args["cond_scale"]
                comb = uncond + cond_scale * (cond - uncond)
                cond_norm = torch.linalg.vector_norm(cond, dim=1, keepdim=True)
                comb_norm = torch.linalg.vector_norm(comb, dim=1, keepdim=True)
                rescale = torch.where(
                    comb_norm > 0,
                    cond_norm / comb_norm.clamp_min(1e-12),
                    torch.ones_like(comb_norm),
                )
                rescaled = comb * rescale
                # strength blends back toward standard linear CFG (1.0 = full rescale).
                if strength != 1.0:
                    rescaled = strength * rescaled + (1.0 - strength) * comb
                return rescaled
            m.set_model_sampler_cfg_function(cfg_norm_pre)
        else:
            def cfg_norm(args):
                cond_p = args['cond_denoised']
                pred_text_ = args["denoised"]

                norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
                norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
                scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
                return pred_text_ * scale * strength

            m.set_model_sampler_post_cfg_function(cfg_norm)
        return io.NodeOutput(m)


class CfgExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CFGZeroStar,
            CFGNorm,
        ]


async def comfy_entrypoint() -> CfgExtension:
    return CfgExtension()
