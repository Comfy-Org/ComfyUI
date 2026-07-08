import math
import comfy.samplers
import comfy.sampler_helpers
import comfy.patcher_extension
import comfy.sample
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion import sa_solver
import latent_preview
import torch
import comfy.utils
import node_helpers
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import re


class BasicScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BasicScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, model, scheduler, steps, denoise) -> io.NodeOutput:
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return io.NodeOutput(torch.FloatTensor([]))
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class KarrasScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KarrasScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("sigma_max", default=14.614642, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sigma_min", default=0.0291675, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("rho", default=7.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, sigma_max, sigma_min, rho) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class ExponentialScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExponentialScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("sigma_max", default=14.614642, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sigma_min", default=0.0291675, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, sigma_max, sigma_min) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class PolyexponentialScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PolyexponentialScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("sigma_max", default=14.614642, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sigma_min", default=0.0291675, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("rho", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, sigma_max, sigma_min, rho) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class LaplaceScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LaplaceScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("sigma_max", default=14.614642, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sigma_min", default=0.0291675, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("mu", default=0.0, min=-10.0, max=10.0, step=0.1, round=False, advanced=True),
                io.Float.Input("beta", default=0.5, min=0.0, max=10.0, step=0.1, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, sigma_max, sigma_min, mu, beta) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_laplace(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, mu=mu, beta=beta)
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class SDTurboScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SDTurboScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=1, min=1, max=10),
                io.Float.Input("denoise", default=1.0, min=0, max=1.0, step=0.01),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, model, steps, denoise) -> io.NodeOutput:
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
        sigmas = model.get_model_object("model_sampling").sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class BetaSamplingScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BetaSamplingScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("alpha", default=0.6, min=0.0, max=50.0, step=0.01, round=False, advanced=True),
                io.Float.Input("beta", default=0.6, min=0.0, max=50.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, model, steps, alpha, beta) -> io.NodeOutput:
        sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta)
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class VPScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VPScheduler",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("beta_d", default=19.9, min=0.0, max=5000.0, step=0.01, round=False, advanced=True), #TODO: fix default values
                io.Float.Input("beta_min", default=0.1, min=0.0, max=5000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("eps_s", default=0.001, min=0.0, max=1.0, step=0.0001, round=False, advanced=True),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, steps, beta_d, beta_min, eps_s) -> io.NodeOutput:
        sigmas = k_diffusion_sampling.get_sigmas_vp(n=steps, beta_d=beta_d, beta_min=beta_min, eps_s=eps_s)
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class SplitSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SplitSigmas",
            category="model/sampling/sigmas",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Int.Input("step", default=0, min=0, max=10000),
            ],
            outputs=[
                io.Sigmas.Output(display_name="high_sigmas"),
                io.Sigmas.Output(display_name="low_sigmas"),
            ]
        )

    @classmethod
    def execute(cls, sigmas, step) -> io.NodeOutput:
        sigmas1 = sigmas[:step + 1]
        sigmas2 = sigmas[step:]
        return io.NodeOutput(sigmas1, sigmas2)

    get_sigmas = execute

class SplitSigmasDenoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SplitSigmasDenoise",
            category="model/sampling/sigmas",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Sigmas.Output(display_name="high_sigmas"),
                io.Sigmas.Output(display_name="low_sigmas"),
            ]
        )

    @classmethod
    def execute(cls, sigmas, denoise) -> io.NodeOutput:
        steps = max(sigmas.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        sigmas1 = sigmas[:-(total_steps)]
        sigmas2 = sigmas[-(total_steps + 1):]
        return io.NodeOutput(sigmas1, sigmas2)

    get_sigmas = execute

class FlipSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlipSigmas",
            category="model/sampling/sigmas",
            inputs=[io.Sigmas.Input("sigmas")],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        if len(sigmas) == 0:
            return io.NodeOutput(sigmas)

        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        return io.NodeOutput(sigmas)

    get_sigmas = execute

class SetFirstSigma(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SetFirstSigma",
            category="model/sampling/sigmas",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Float.Input("sigma", default=136.0, min=0.0, max=20000.0, step=0.001, round=False),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, sigmas, sigma) -> io.NodeOutput:
        sigmas = sigmas.clone()
        sigmas[0] = sigma
        return io.NodeOutput(sigmas)

    set_first_sigma = execute

class ExtendIntermediateSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtendIntermediateSigmas",
            search_aliases=["interpolate sigmas"],
            category="model/sampling/sigmas",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Int.Input("steps", default=2, min=1, max=100),
                io.Float.Input("start_at_sigma", default=-1.0, min=-1.0, max=20000.0, step=0.01, round=False),
                io.Float.Input("end_at_sigma", default=12.0, min=0.0, max=20000.0, step=0.01, round=False),
                io.Combo.Input("spacing", options=['linear', 'cosine', 'sine']),
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, sigmas: torch.Tensor, steps: int, start_at_sigma: float, end_at_sigma: float, spacing: str) -> io.NodeOutput:
        if start_at_sigma < 0:
            start_at_sigma = float("inf")

        interpolator = {
            'linear': lambda x: x,
            'cosine': lambda x: torch.sin(x*math.pi/2),
            'sine':   lambda x: 1 - torch.cos(x*math.pi/2)
        }[spacing]

        # linear space for our interpolation function
        x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
        computed_spacing = interpolator(x)

        extended_sigmas = []
        for i in range(len(sigmas) - 1):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i+1]

            extended_sigmas.append(sigma_current)

            if end_at_sigma <= sigma_current <= start_at_sigma:
                interpolated_steps = computed_spacing * (sigma_next - sigma_current) + sigma_current
                extended_sigmas.extend(interpolated_steps.tolist())

        # Add the last sigma value
        if len(sigmas) > 0:
            extended_sigmas.append(sigmas[-1])

        extended_sigmas = torch.FloatTensor(extended_sigmas)

        return io.NodeOutput(extended_sigmas)

    extend = execute


class SamplingPercentToSigma(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplingPercentToSigma",
            category="model/sampling/sigmas",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("sampling_percent", default=0.0, min=0.0, max=1.0, step=0.0001),
                io.Boolean.Input("return_actual_sigma", default=False, tooltip="Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0."),
            ],
            outputs=[io.Float.Output(display_name="sigma_value")]
        )

    @classmethod
    def execute(cls, model, sampling_percent, return_actual_sigma) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        sigma_val = model_sampling.percent_to_sigma(sampling_percent)
        if return_actual_sigma:
            if sampling_percent == 0.0:
                sigma_val = model_sampling.sigma_max.item()
            elif sampling_percent == 1.0:
                sigma_val = model_sampling.sigma_min.item()
        return io.NodeOutput(sigma_val)

    get_sigma = execute


class KSamplerSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KSamplerSelect",
            category="model/sampling/samplers",
            inputs=[io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES)],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, sampler_name) -> io.NodeOutput:
        sampler = comfy.samplers.sampler_object(sampler_name)
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerDPMPP_3M_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_3M_SDE",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("noise_device", options=['gpu', 'cpu'], advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, eta, s_noise, noise_device) -> io.NodeOutput:
        if noise_device == 'cpu':
            sampler_name = "dpmpp_3m_sde"
        else:
            sampler_name = "dpmpp_3m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerDPMPP_2M_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_2M_SDE",
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver_type", options=['midpoint', 'heun']),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("noise_device", options=['gpu', 'cpu'], advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, solver_type, eta, s_noise, noise_device) -> io.NodeOutput:
        if noise_device == 'cpu':
            sampler_name = "dpmpp_2m_sde"
        else:
            sampler_name = "dpmpp_2m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type})
        return io.NodeOutput(sampler)

    get_sampler = execute


class SamplerDPMPP_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_SDE",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("r", default=0.5, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Combo.Input("noise_device", options=['gpu', 'cpu'], advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, eta, s_noise, r, noise_device) -> io.NodeOutput:
        if noise_device == 'cpu':
            sampler_name = "dpmpp_sde"
        else:
            sampler_name = "dpmpp_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerDPMPP_2S_Ancestral(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMPP_2S_Ancestral",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, eta, s_noise) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler("dpmpp_2s_ancestral", {"eta": eta, "s_noise": s_noise})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerEulerAncestral(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerEulerAncestral",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, eta, s_noise) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler("euler_ancestral", {"eta": eta, "s_noise": s_noise})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerEulerAncestralCFGPP(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerEulerAncestralCFGPP",
            display_name="SamplerEulerAncestralCFG++",
            category="model/sampling/samplers",
            inputs=[
                io.Float.Input("eta", default=1.0, min=0.0, max=1.0, step=0.01, round=False),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=10.0, step=0.01, round=False),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, eta, s_noise) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler(
            "euler_ancestral_cfg_pp",
            {"eta": eta, "s_noise": s_noise})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerLMS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerLMS",
            category="model/sampling/samplers",
            inputs=[io.Int.Input("order", default=4, min=1, max=100, advanced=True)],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, order) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler("lms", {"order": order})
        return io.NodeOutput(sampler)

    get_sampler = execute

class SamplerDPMAdaptative(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerDPMAdaptative",
            category="model/sampling/samplers",
            inputs=[
                io.Int.Input("order", default=3, min=2, max=3, advanced=True),
                io.Float.Input("rtol", default=0.05, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("atol", default=0.0078, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("h_init", default=0.05, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("pcoeff", default=0.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("icoeff", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("dcoeff", default=0.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("accept_safety", default=0.81, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("eta", default=0.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise) -> io.NodeOutput:
        sampler = comfy.samplers.ksampler("dpm_adaptive", {"order": order, "rtol": rtol, "atol": atol, "h_init": h_init, "pcoeff": pcoeff,
                                                              "icoeff": icoeff, "dcoeff": dcoeff, "accept_safety": accept_safety, "eta": eta,
                                                              "s_noise":s_noise })
        return io.NodeOutput(sampler)

    get_sampler = execute


class SamplerER_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerER_SDE",
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver_type", options=["ER-SDE", "Reverse-time SDE", "ODE"]),
                io.Int.Input("max_stage", default=3, min=1, max=3, advanced=True),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, tooltip="Stochastic strength of reverse-time SDE.\nWhen eta=0, it reduces to deterministic ODE. This setting doesn't apply to ER-SDE solver type.", advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, solver_type, max_stage, eta, s_noise) -> io.NodeOutput:
        if solver_type == "ODE" or (solver_type == "Reverse-time SDE" and eta == 0):
            eta = 0
            s_noise = 0

        def reverse_time_sde_noise_scaler(x):
            return x ** (eta + 1)

        if solver_type == "ER-SDE":
            # Use the default one in sample_er_sde()
            noise_scaler = None
        else:
            noise_scaler = reverse_time_sde_noise_scaler

        sampler_name = "er_sde"
        sampler = comfy.samplers.ksampler(sampler_name, {"s_noise": s_noise, "noise_scaler": noise_scaler, "max_stage": max_stage})
        return io.NodeOutput(sampler)

    get_sampler = execute


class SamplerSASolver(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSASolver",
            search_aliases=["sde"],
            category="model/sampling/samplers",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("eta", default=1.0, min=0.0, max=10.0, step=0.01, round=False, advanced=True),
                io.Float.Input("sde_start_percent", default=0.2, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("sde_end_percent", default=0.8, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Int.Input("predictor_order", default=3, min=1, max=6, advanced=True),
                io.Int.Input("corrector_order", default=4, min=0, max=6, advanced=True),
                io.Boolean.Input("use_pece", advanced=True),
                io.Boolean.Input("simple_order_2", advanced=True),
            ],
            outputs=[io.Sampler.Output()]
        )

    @classmethod
    def execute(cls, model, eta, sde_start_percent, sde_end_percent, s_noise, predictor_order, corrector_order, use_pece, simple_order_2) -> io.NodeOutput:
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=eta)

        sampler_name = "sa_solver"
        sampler = comfy.samplers.ksampler(
            sampler_name,
            {
                "tau_func": tau_func,
                "s_noise": s_noise,
                "predictor_order": predictor_order,
                "corrector_order": corrector_order,
                "use_pece": use_pece,
                "simple_order_2": simple_order_2,
            },
        )
        return io.NodeOutput(sampler)

    get_sampler = execute


class SamplerSEEDS2(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSEEDS2",
            search_aliases=["sde", "exp heun"],
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver_type", options=["phi_1", "phi_2"]),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, round=False, tooltip="Stochastic strength", advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, tooltip="SDE noise multiplier", advanced=True),
                io.Float.Input("r", default=0.5, min=0.01, max=1.0, step=0.01, round=False, tooltip="Relative step size for the intermediate stage (c2 node)", advanced=True),
            ],
            outputs=[io.Sampler.Output()],
            description=(
                "This sampler node can represent multiple samplers:\n\n"
                "seeds_2\n"
                "- default setting\n\n"
                "exp_heun_2_x0\n"
                "- solver_type=phi_2, r=1.0, eta=0.0\n\n"
                "exp_heun_2_x0_sde\n"
                "- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0"
            )
        )

    @classmethod
    def execute(cls, solver_type, eta, s_noise, r) -> io.NodeOutput:
        sampler_name = "seeds_2"
        sampler = comfy.samplers.ksampler(
            sampler_name,
            {"eta": eta, "s_noise": s_noise, "r": r, "solver_type": solver_type},
        )
        return io.NodeOutput(sampler)


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        if latent_image.is_nested:
            tensors = latent_image.unbind()
            zeros = []
            for t in tensors:
                zeros.append(torch.zeros(t.shape, dtype=t.dtype, layout=t.layout, device="cpu"))
            return comfy.nested_tensor.NestedTensor(zeros)
        else:
            return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

class SamplerCustom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerCustom",
            category="model/sampling/custom",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("add_noise", default=True, advanced=True),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ]
        )

    @classmethod
    def execute(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image) -> io.NodeOutput:
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0_out = model.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out
        return io.NodeOutput(out, out_denoised)

    sample = execute

class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class BasicGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BasicGuider",
            display_name="Basic Guider",
            category="model/sampling/guiders",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[io.Guider.Output()]
        )

    @classmethod
    def execute(cls, model, conditioning) -> io.NodeOutput:
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return io.NodeOutput(guider)

    get_guider = execute

class CFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CFGGuider",
            display_name="CFG Guider",
            category="model/sampling/guiders",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
            ],
            outputs=[io.Guider.Output()]
        )

    @classmethod
    def execute(cls, model, positive, negative, cfg) -> io.NodeOutput:
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return io.NodeOutput(guider)

    get_guider = execute

class Guider_DualCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, cfg2, nested=False):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.nested = nested

    def set_conds(self, positive, middle, negative):
        middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        middle_cond = self.conds.get("middle", None)
        positive_cond = self.conds.get("positive", None)

        if self.nested:
            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            pred_text = comfy.samplers.cfg_function(self.inner_model, out[2], out[1], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=middle_cond)
            return out[0] + self.cfg2 * (pred_text - out[0])
        else:
            if model_options.get("disable_cfg1_optimization", False) == False:
                if math.isclose(self.cfg2, 1.0):
                    negative_cond = None
                    if math.isclose(self.cfg1, 1.0):
                        middle_cond = None

            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            return comfy.samplers.cfg_function(self.inner_model, out[1], out[0], self.cfg2, x, timestep, model_options=model_options, cond=middle_cond, uncond=negative_cond) + (out[2] - out[1]) * self.cfg1

class DualCFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DualCFGGuider",
            search_aliases=["dual prompt guidance"],
            display_name="Dual CFG Guider",
            category="model/sampling/guiders",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("cond1"),
                io.Conditioning.Input("cond2"),
                io.Conditioning.Input("negative"),
                io.Float.Input("cfg_conds", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("cfg_cond2_negative", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("style", options=["regular", "nested"]),
            ],
            outputs=[io.Guider.Output()]
        )

    @classmethod
    def execute(cls, model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative, style) -> io.NodeOutput:
        guider = Guider_DualCFG(model)
        guider.set_conds(cond1, cond2, negative)
        guider.set_cfg(cfg_conds, cfg_cond2_negative, nested=(style == "nested"))
        return io.NodeOutput(guider)

    get_guider = execute

class Guider_DualModel(comfy.samplers.CFGGuider):
    # Runs the positive (cond) pass on the main model and the negative (uncond) pass on a separate model
    def __init__(self, model_patcher, uncond_model_patcher):
        super().__init__(model_patcher)
        self.uncond_model_patcher = uncond_model_patcher
        self.uncond_inner = None

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        self.uncond_inner = None
        self.uncond_loaded = []
        self._uncond_neg = None
        # skip at cfg 1.0
        if not math.isclose(self.cfg, 1.0):
            uc = {"negative": list(map(lambda a: a.copy(), self.conds["negative"]))}
            self.uncond_inner, uc, self.uncond_loaded = comfy.sampler_helpers.prepare_sampling(
                self.uncond_model_patcher, noise.shape, uc, self.uncond_model_patcher.model_options)
            self._uncond_neg = uc["negative"]
            self.uncond_model_patcher.pre_run()
        try:
            return super().outer_sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            if self.uncond_inner is not None:
                self.uncond_model_patcher.cleanup()
                comfy.sampler_helpers.cleanup_models({"negative": self._uncond_neg}, self.uncond_loaded)
                self.uncond_inner = None

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if self.uncond_inner is not None:
            li = latent_image
            if li is not None and torch.count_nonzero(li) > 0:
                li = self.uncond_inner.process_latent_in(li)
            self._uncond_conds = comfy.samplers.process_conds(
                self.uncond_inner, noise, {"negative": self._uncond_neg}, device, li, denoise_mask, seed, latent_shapes=latent_shapes)["negative"]
        return super().inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        cond = comfy.samplers.calc_cond_batch(self.inner_model, [positive], x, timestep, model_options)[0]
        # uncond model not loaded (base cfg==1/no negative), or cfg driven to 1.0 this step -> single model, cond only
        if self.uncond_inner is None or (math.isclose(self.cfg, 1.0) and not model_options.get("disable_cfg1_optimization", False)):
            return cond

        uncond_model_options = model_options
        if "multigpu_clones" in model_options: # TODO: support multigpu instead of just running uncond on a single GPU
            uncond_model_options = {k: v for k, v in model_options.items() if k != "multigpu_clones"}
        uncond = comfy.samplers.calc_cond_batch(self.uncond_inner, [self._uncond_conds], x, timestep, uncond_model_options)[0]
        return comfy.samplers.cfg_function(self.inner_model, cond, uncond, self.cfg, x, timestep,
                                           model_options=model_options, cond=positive, uncond=self._uncond_conds)

class DualModelGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DualModelGuider",
            display_name="Dual Model CFG Guider",
            category="model/sampling/guiders",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="Model used for the positive (conditional) pass."),
                io.Model.Input("model_negative", optional=True, tooltip="Model used for the negative (unconditional) pass. Use the same model for ordinary CFG."),
                io.Conditioning.Input("positive"),
                io.Float.Input("cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Conditioning.Input("negative", optional=True, tooltip="Negative conditioning run on the negative model. Leave unconnected for a text-free (image-only) unconditional pass."),
            ],
            outputs=[io.Guider.Output()],
        )

    @classmethod
    def execute(cls, model, positive, cfg, model_negative=None, negative=None) -> io.NodeOutput:
        if negative is None:
            negative = [[None, {}]]  # null cond -> no cross_attn -> model runs image-only

        guider = Guider_DualModel(model, model_negative) if model_negative is not None else comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return io.NodeOutput(guider)

    get_guider = execute

class DisableNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DisableNoise",
            search_aliases=["zero noise"],
            category="model/sampling/noise",
            inputs=[],
            outputs=[io.Noise.Output()]
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(Noise_EmptyNoise())

    get_noise = execute


class RandomNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RandomNoise",
            category="model/sampling/noise",
            inputs=[io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True)],
            outputs=[io.Noise.Output()]
        )

    @classmethod
    def execute(cls, noise_seed) -> io.NodeOutput:
        return io.NodeOutput(Noise_RandomNoise(noise_seed))

    get_noise = execute


class SamplerCustomAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerCustomAdvanced",
            category="model/sampling/custom",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ]
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image) -> io.NodeOutput:
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        if "x0" in x0_output:
            x0_out = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            if samples.is_nested:
                latent_shapes = [x.shape for x in samples.unbind()]
                x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
            out_denoised = latent.copy()
            out_denoised["samples"] = x0_out
        else:
            out_denoised = out
        return io.NodeOutput(out, out_denoised)

    sample = execute

class AddNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AddNoise",
            category="model/sampling/noise",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Noise.Input("noise"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
            ],
            outputs=[
                io.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, model, noise, sigmas, latent_image) -> io.NodeOutput:
        if len(sigmas) == 0:
            return io.NodeOutput(latent_image)

        latent = latent_image
        latent_image = latent["samples"]

        noisy = noise.generate_noise(latent)

        model_sampling = model.get_model_object("model_sampling")
        process_latent_out = model.get_model_object("process_latent_out")
        process_latent_in = model.get_model_object("process_latent_in")

        if len(sigmas) > 1:
            scale = torch.abs(sigmas[0] - sigmas[-1])
        else:
            scale = sigmas[0]

        if torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = process_latent_in(latent_image)
        noisy = model_sampling.noise_scaling(scale, noisy, latent_image)
        noisy = process_latent_out(noisy)
        noisy = torch.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)

        out = latent.copy()
        out["samples"] = noisy
        return io.NodeOutput(out)

    add_noise = execute

class ManualSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ManualSigmas",
            search_aliases=["custom noise schedule", "define sigmas"],
            category="model/sampling/sigmas",
            is_experimental=True,
            inputs=[
                io.String.Input("sigmas", default="1, 0.5", multiline=False)
            ],
            outputs=[io.Sigmas.Output()]
        )

    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
        sigmas = re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)
        sigmas = [float(i) for i in sigmas]
        sigmas = torch.FloatTensor(sigmas)
        return io.NodeOutput(sigmas)

class CFGOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CFGOverride",
            display_name="CFG Override",
            description="Override cfg to a fixed value over a [start, end] percent (sigma) range. "
                        "With multiple overrides, the one nearest the sampler wins on overlap.",
            category="model/sampling/guiders",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, cfg, start_percent, end_percent) -> io.NodeOutput:
        ms = model.get_model_object("model_sampling")
        sigma_hi = ms.percent_to_sigma(start_percent)  # percent->sigma decreasing, so hi >= lo
        sigma_lo = ms.percent_to_sigma(end_percent)

        def predict_noise_wrapper(executor, *args, **kwargs):
            sigma = float(args[1].flatten()[0])        # args = (x, timestep, model_options, seed)
            if not (sigma_lo <= sigma <= sigma_hi):
                return executor(*args, **kwargs)
            guider = executor.class_obj                # guider.cfg feeds cond_scale
            saved = guider.cfg
            guider.cfg = cfg
            try:
                return executor(*args, **kwargs)
            finally:
                guider.cfg = saved                     # restore for other steps/overrides

        m = model.clone()
        m.add_wrapper(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, predict_noise_wrapper)
        return io.NodeOutput(m)


class CustomSamplersExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SamplerCustom,
            CFGOverride,
            BasicScheduler,
            KarrasScheduler,
            ExponentialScheduler,
            PolyexponentialScheduler,
            LaplaceScheduler,
            VPScheduler,
            BetaSamplingScheduler,
            SDTurboScheduler,
            KSamplerSelect,
            SamplerEulerAncestral,
            SamplerEulerAncestralCFGPP,
            SamplerLMS,
            SamplerDPMPP_3M_SDE,
            SamplerDPMPP_2M_SDE,
            SamplerDPMPP_SDE,
            SamplerDPMPP_2S_Ancestral,
            SamplerDPMAdaptative,
            SamplerER_SDE,
            SamplerSASolver,
            SamplerSEEDS2,
            SplitSigmas,
            SplitSigmasDenoise,
            FlipSigmas,
            SetFirstSigma,
            ExtendIntermediateSigmas,
            SamplingPercentToSigma,
            CFGGuider,
            DualCFGGuider,
            DualModelGuider,
            BasicGuider,
            RandomNoise,
            DisableNoise,
            AddNoise,
            SamplerCustomAdvanced,
            ManualSigmas,
        ]


async def comfy_entrypoint() -> CustomSamplersExtension:
    return CustomSamplersExtension()
