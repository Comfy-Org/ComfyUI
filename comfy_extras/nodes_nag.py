import torch
from comfy_api.v3 import ComfyExtension, io
from typing_extensions import override


class NAGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NAGuidance",
            display_name="Normalized Attention Guidance",
            description="Applies Normalized Attention Guidance to models, enabling negative prompts on distilled/schnell models.",
            category="advanced/guidance",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to apply NAG to."),
                io.Float.Input("nag_scale", min=0.0, default=5.0, max=50.0, step=0.1, tooltip="The guidance scale factor. Higher values push further from the negative prompt."),
                io.Float.Input("nag_alpha", min=0.0, default=0.5, max=1.0, step=0.01, tooltip="Blending factor for the normalized attention. 1.0 is full replacement, 0.0 is no effect."),
                io.Float.Input("nag_tau", min=1.0, default=1.5, max=10.0, step=0.01),
                # io.Float.Input("start_percent", min=0.0, default=0.0, max=1.0, step=0.01, tooltip="The relative sampling step to begin applying NAG."),
                # io.Float.Input("end_percent", min=0.0, default=1.0, max=1.0, step=0.01, tooltip="The relative sampling step to stop applying NAG."),
            ],
            outputs=[
                io.Model.Output(tooltip="The patched model with NAG enabled."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, nag_scale: float, nag_alpha: float, nag_tau: float) -> io.NodeOutput:
        m = model.clone()

        # sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_percent)
        # sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_percent)

        def nag_attention_output_patch(out, extra_options):
            cond_or_uncond = extra_options.get("cond_or_uncond", None)
            if cond_or_uncond is None:
                return out

            if not (1 in cond_or_uncond and 0 in cond_or_uncond):
                return out

            # sigma = extra_options.get("sigmas", None)
            # if sigma is not None and len(sigma) > 0:
            #     sigma = sigma[0].item()
            #     if sigma > sigma_start or sigma < sigma_end:
            #         return out

            img_slice = extra_options.get("img_slice", None)

            if img_slice is not None:
                orig_out = out
                out = out[:, img_slice[0]:img_slice[1]]  # only apply on img part

            batch_size = out.shape[0]
            half_size = batch_size // len(cond_or_uncond)

            ind_neg = cond_or_uncond.index(1)
            ind_pos = cond_or_uncond.index(0)
            z_pos = out[half_size * ind_pos:half_size * (ind_pos + 1)]
            z_neg = out[half_size * ind_neg:half_size * (ind_neg + 1)]

            guided = z_pos * nag_scale - z_neg * (nag_scale - 1.0)

            eps = 1e-6
            norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True).clamp_min(eps)
            norm_guided = torch.norm(guided, p=1, dim=-1, keepdim=True).clamp_min(eps)

            ratio = norm_guided / norm_pos
            scale_factor = torch.minimum(ratio, torch.full_like(ratio, nag_tau)) / ratio

            guided_normalized = guided * scale_factor

            z_final = guided_normalized * nag_alpha + z_pos * (1.0 - nag_alpha)

            if img_slice is not None:
                orig_out[half_size * ind_neg:half_size * (ind_neg + 1), img_slice[0]:img_slice[1]] = z_final
                orig_out[half_size * ind_pos:half_size * (ind_pos + 1), img_slice[0]:img_slice[1]] = z_final
                return orig_out
            else:
                out[half_size * ind_pos:half_size * (ind_pos + 1)] = z_final
            return out

        m.set_model_attn1_output_patch(nag_attention_output_patch)
        m.disable_model_cfg1_optimization()

        return io.NodeOutput(m)


class NagExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            NAGuidance,
        ]


async def comfy_entrypoint() -> NagExtension:
    return NagExtension()
