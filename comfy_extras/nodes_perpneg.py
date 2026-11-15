import torch
import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
import node_helpers
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale):
    pos = noise_pred_pos - noise_pred_nocond
    neg = noise_pred_neg - noise_pred_nocond

    perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
    perp_neg = perp * neg_scale
    cfg_result = noise_pred_nocond + cond_scale*(pos - perp_neg)
    return cfg_result

#TODO: This node should be removed, it has been replaced with PerpNegGuider
class PerpNeg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PerpNeg",
            display_name="Perp-Neg (DEPRECATED by Perp-Neg Guider)",
            category="experimental",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("empty_conditioning"),
                io.Float.Input("neg_scale", default=1.0, min=0.0, max=100.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=True,
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, model, empty_conditioning, neg_scale) -> io.NodeOutput:
        m = model.clone()
        nocond = comfy.sampler_helpers.convert_cond(empty_conditioning)

        def cfg_function(args):
            model = args["model"]
            noise_pred_pos = args["cond_denoised"]
            noise_pred_neg = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            nocond_processed = comfy.samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

            (noise_pred_nocond,) = comfy.samplers.calc_cond_batch(model, [nocond_processed], x, sigma, model_options)

            cfg_result = x - perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale)
            return cfg_result

        m.set_model_sampler_cfg_function(cfg_function)

        return io.NodeOutput(m)


class Guider_PerpNeg(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative, empty_negative_prompt):
        empty_negative_prompt = node_helpers.conditioning_set_values(empty_negative_prompt, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "empty_negative_prompt": empty_negative_prompt, "negative": negative})

    def set_cfg(self, cfg, neg_scale):
        self.cfg = cfg
        self.neg_scale = neg_scale

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and empty, so we call calc_cond_batch([pos,neg,empty]) directly

        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)
        empty_cond = self.conds.get("empty_negative_prompt", None)

        if model_options.get("disable_cfg1_optimization", False) == False:
            if math.isclose(self.neg_scale, 0.0):
                negative_cond = None
                if math.isclose(self.cfg, 1.0):
                    empty_cond = None

        conds = [positive_cond, negative_cond, empty_cond]

        out = comfy.samplers.calc_cond_batch(self.inner_model, conds, x, timestep, model_options)

        # Apply pre_cfg_functions since sampling_function() is skipped
        for fn in model_options.get("sampler_pre_cfg_function", []):
            args = {"conds":conds, "conds_out": out, "cond_scale": self.cfg, "timestep": timestep,
                    "input": x, "sigma": timestep, "model": self.inner_model, "model_options": model_options}
            out = fn(args)

        noise_pred_pos, noise_pred_neg, noise_pred_empty = out
        cfg_result = perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_empty, self.neg_scale, self.cfg)

        # normally this would be done in cfg_function, but we skipped
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": cfg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "cond_scale": self.cfg,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                # not in the original call in samplers.py:cfg_function, but made available for future hooks
                "empty_cond": empty_cond,
                "empty_cond_denoised": noise_pred_empty,}
            cfg_result = fn(args)

        return cfg_result

class PerpNegGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PerpNegGuider",
            display_name="Perp-Neg Guider",
            category="experimental",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Conditioning.Input("empty_conditioning"),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("neg_scale", default=1.0, min=0.0, max=100.0, step=0.01, advanced=True),
            ],
            outputs=[
                io.Guider.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, positive, negative, empty_conditioning, cfg, neg_scale) -> io.NodeOutput:
        guider = Guider_PerpNeg(model)
        guider.set_conds(positive, negative, empty_conditioning)
        guider.set_cfg(cfg, neg_scale)
        return io.NodeOutput(guider)


class PerpNegExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PerpNeg,
            PerpNegGuider,
        ]


async def comfy_entrypoint() -> PerpNegExtension:
    return PerpNegExtension()
