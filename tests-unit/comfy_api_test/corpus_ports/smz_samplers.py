"""smZNodes' "dpmpp_2m_alt" sampler (smZNodes.py), ported to the V2 API.

The original inserts its name into ``KSampler.SAMPLERS`` and sets ``k_diffusion_sampling.sample_dpmpp_2m_alt``
at import. The port declares it: ``ComfyExtension.get_samplers``. The sampling loop is the original's.
"""
import torch

from comfy_api.index import ComfyExtension


@torch.no_grad()
def sample_dpmpp_2m_alt(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()  # noqa: E731
    t_fn = lambda sigma: sigma.log().neg()  # noqa: E731
    old_denoised = None

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        sigma_progress = i / len(sigmas)
        adjustment_factor = 1 + (0.15 * (sigma_progress * sigma_progress))
        old_denoised = denoised * adjustment_factor
    return x


class SmzSamplers(ComfyExtension):
    async def get_node_list(self):
        return []

    async def get_samplers(self):
        return {"dpmpp_2m_alt": sample_dpmpp_2m_alt}
