import math

import torch

import comfy.model_sampling


def time_snr_shift(shift, value):
    if shift == 1.0:
        return value
    return shift * value / (1.0 + (shift - 1.0) * value)


def inverse_time_snr_shift(shift, value):
    if shift == 1.0:
        return value
    return value / (shift - (shift - 1.0) * value)


def upstream_timesteps(steps, shift, device=None):
    base = torch.linspace(0.0, 1.0, steps + 1, device=device)
    return 1.0 - time_snr_shift(shift, 1.0 - base)


def upstream_sigmas(steps, shift, device=None):
    return 1.0 - upstream_timesteps(steps, shift, device=device)


def resolution_noise_scale(
    height, width, base_seq_len=64, noise_scale=1.0, maximum=16.0
):
    token_height = math.ceil(height / 32)
    token_width = math.ceil(width / 32)
    scale = math.sqrt(token_height * token_width / base_seq_len) * noise_scale
    return min(scale, maximum)


class SenseNovaModelSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST
):
    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        base_timesteps = torch.linspace(multiplier, 0.0, timesteps + 1)
        self.register_buffer("sigmas", self.sigma(base_timesteps))

    def timestep(self, sigma):
        base_sigma = inverse_time_snr_shift(self.shift, sigma)
        return (1.0 - base_sigma) * self.multiplier

    def sigma(self, timestep):
        base_sigma = 1.0 - timestep / self.multiplier
        return time_snr_shift(self.shift, base_sigma)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(time_snr_shift(self.shift, 1.0 - percent))

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = comfy.model_sampling.reshape_sigma(sigma, noise.ndim)
        scale = resolution_noise_scale(
            latent_image.shape[-2],
            latent_image.shape[-1],
            noise_scale=self.noise_scale,
        )
        return sigma * (scale * noise) + (1.0 - sigma) * latent_image
