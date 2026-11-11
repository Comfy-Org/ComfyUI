import numpy as np
import torch
from tqdm.auto import trange
from typing_extensions import override

import comfy.model_patcher
import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d
from comfy_api.latest import ComfyExtension, io


@torch.no_grad()
def sample_lcm_upscale(model, x, sigmas, extra_args=None, callback=None, disable=None, total_upscale=2.0, upscale_method="bislerp", upscale_steps=None):
    extra_args = {} if extra_args is None else extra_args

    if upscale_steps is None:
        upscale_steps = max(len(sigmas) // 2 + 1, 2)
    else:
        upscale_steps += 1
        upscale_steps = min(upscale_steps, len(sigmas) + 1)

    upscales = np.linspace(1.0, total_upscale, upscale_steps)[1:]

    orig_shape = x.size()
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if i < len(upscales):
            x = comfy.utils.common_upscale(x, round(orig_shape[-1] * upscales[i]), round(orig_shape[-2] * upscales[i]), upscale_method, "disabled")

        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x


class SamplerLCMUpscale(io.ComfyNode):
    UPSCALE_METHODS = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerLCMUpscale",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("scale_ratio", default=1.0, min=0.1, max=20.0, step=0.01, advanced=True),
                io.Int.Input("scale_steps", default=-1, min=-1, max=1000, step=1, advanced=True),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, scale_ratio, scale_steps, upscale_method) -> io.NodeOutput:
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscale, extra_options={"total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return io.NodeOutput(sampler)


@torch.no_grad()
def sample_euler_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x - denoised + temp[0], sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


class SamplerLCM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerLCM",
            category="model/sampling/samplers",
            description=("LCM sampler with tunable per-step noise. s_noise is a multiplier on the model's training noise scale"),
            inputs=[
                io.Float.Input("s_noise", default=1.0, min=0.0, max=64.0, step=0.01,
                               tooltip="Per-step noise multiplier at the first step (1.0 = match training)."),
                io.Float.Input("s_noise_end", default=1.0, min=0.0, max=64.0, step=0.01,
                               tooltip="Per-step noise multiplier at the last step. Set equal to s_noise for a constant schedule."),
                io.Float.Input("noise_clip_std", default=0.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Clamp per-step noise to +/- N*std. 0 disables."),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, s_noise, s_noise_end, noise_clip_std) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler(
            "lcm",
            {
                "s_noise": float(s_noise),
                "s_noise_end": float(s_noise_end),
                "noise_clip_std": float(noise_clip_std),
            },
        )
        return io.NodeOutput(sampler)


class SamplerEulerCFGpp(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerEulerCFGpp",
            display_name="SamplerEulerCFG++",
            category="experimental",  # "sampling/samplers"
            inputs=[
                io.Combo.Input("version", options=["regular", "alternative"], advanced=True),
            ],
            outputs=[io.Sampler.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, version) -> io.NodeOutput:
        if version == "alternative":
            sampler = comfy.samplers.KSAMPLER(sample_euler_pp)
        else:
            sampler = comfy.samplers.ksampler("euler_cfg_pp")
        return io.NodeOutput(sampler)


class AdvancedSamplersExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SamplerLCMUpscale,
            SamplerLCM,
            SamplerEulerCFGpp,
        ]

async def comfy_entrypoint() -> AdvancedSamplersExtension:
    return AdvancedSamplersExtension()
