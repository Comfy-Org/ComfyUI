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
                out = pred_text_ * scale * strength
                # `scale` is clamped to <= 1, so while strength <= 1 this branch can
                # only attenuate and nothing can overflow. Above 1 the multiply is an
                # amplification of the x0 prediction applied on every sampling step,
                # it feeds back through the latent, and well before the declared
                # max=100 it runs off the end of the dtype (around strength 4.5 on
                # SD1.5 fp16). The inf used to be returned as-is: the sampler turned
                # it into NaN and the render finished as a black image with no error
                # anywhere. Check here, where we can still name the input.
                if strength > 1.0 and not torch.isfinite(out).all():
                    raise RuntimeError(
                        "CFGNorm: strength={} made the denoised prediction non-finite. "
                        "strength multiplies the prediction on every sampling step and "
                        "feeds back through the latent, so it compounds until it runs out "
                        "of numeric range somewhere in the sampling loop; the render would "
                        "have decoded to an all-NaN (black) image. Values above 1.0 "
                        "amplify, which is what this node's clamp(max=1.0) exists to "
                        "prevent -- strength <= 1.0 can never overflow.".format(strength)
                    )
                return out

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
