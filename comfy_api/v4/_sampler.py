from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypedDict

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io

from ._noise import NoiseGenerator


class SamplerArgs(TypedDict, total=False):
    """What a sampler function passes back whole: `model(x, sigma, **extra_args)`.

    ComfyUI owns this dict and may add keys; only the keys below are part of the API.
    """

    seed: int | None
    denoise_mask: torch.Tensor | None


class SamplerStep(TypedDict):
    """What a sampler function passes to `callback` once per step.

    `i` is the 0-based step index; progress shows `i + 1` of `len(sigmas) - 1`. `denoised` is the
    step's prediction, shown as the preview.
    """

    x: torch.Tensor
    i: int
    sigma: torch.Tensor
    sigma_hat: torch.Tensor
    denoised: torch.Tensor


class Denoiser(Protocol):
    """ComfyUI's model for one sampling run. Only calling it is part of the API."""

    def __call__(
        self, x: torch.Tensor, sigma: torch.Tensor, **extra_args: Any
    ) -> torch.Tensor:
        """Return the denoised prediction of `x` at noise level `sigma`, shaped like `x`.

        `sigma` holds one noise level per batch item, as `sigmas[i] * x.new_ones([x.shape[0]])`
        gives.
        """
        ...


class SamplerFunction(Protocol):
    """A sampling loop written as k-diffusion writes them; returns the sampled `x`.

    `model`, `x` and `sigmas` come by position, the rest by keyword, with each of
    the options given to `eng.sampler.custom` as one more keyword. `x` is the input
    latent in the model's range, already noised to `sigmas[0]` the way the model
    noises a latent, which differs between models: a flow model mixes noise and
    latent instead of adding them. `disable` is True when the loop should not draw
    its own console progress bar.
    """

    def __call__(
        self,
        model: Denoiser,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: SamplerArgs,
        callback: Callable[[SamplerStep], None] | None,
        disable: bool | None,
    ) -> torch.Tensor: ...


class Sampler(ProxiedSingleton):
    async def named(
        self,
        name: str,
        options: Mapping[str, bool | int | float | str] | None = None,
    ) -> io.Sampler.Type:
        """Return one of ComfyUI's samplers by name, such as "euler", with a copy of `options`.

        `options` are the sampler's keyword arguments, such as `s_noise`; one the sampler does not
        take raises `TypeError` when sampling, not here. Raises `AttributeError` for a name ComfyUI
        does not know.
        """
        from comfy.samplers import sampler_object

        sampler = sampler_object(name)
        sampler.extra_options = dict(options or {})
        return sampler

    async def custom(
        self, function: SamplerFunction, options: Mapping[str, object] | None = None
    ) -> io.Sampler.Type:
        """Return a sampler whose loop is `function`, called with a copy of `options` as keywords."""
        from comfy.samplers import KSAMPLER

        return KSAMPLER(function, extra_options=dict(options or {}))

    async def sigmas(
        self, model: io.Model.Type, scheduler: str, steps: int, denoise: float = 1.0
    ) -> torch.Tensor:
        """Return the `steps + 1` noise levels a named scheduler gives this model, on the CPU.

        `denoise` below 1 keeps the end of a longer schedule; 0 gives an empty tensor.
        Raises ValueError for a scheduler ComfyUI does not know.
        """
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        return BasicScheduler.execute(
            model=model, scheduler=scheduler, steps=steps, denoise=denoise
        ).result[0]

    async def sample(
        self,
        model: io.Model.Type,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: io.Conditioning.Type,
        negative: io.Conditioning.Type,
        latent: io.Latent.Type,
        denoise: float = 1.0,
        disable_noise: bool = False,
        start_step: int | None = None,
        last_step: int | None = None,
        force_full_denoise: bool = False,
    ) -> io.Latent.Type:
        """Sample as the KSampler nodes do; returns a shallow copy of `latent` with new `samples`.

        Other entries, such as `batch_index` and `noise_mask`, are the same objects as in `latent`;
        the copy has no `downscale_ratio_spacial` or `downscale_ratio_temporal`. Reports progress
        and previews as KSampler does. A sampler or scheduler name ComfyUI does not know is
        replaced by the first in its list, as in KSampler.
        """
        from nodes import common_ksampler

        return common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
        )[0]

    async def sample_custom(
        self,
        noise: NoiseGenerator,
        guider: io.Guider.Type,
        sampler: io.Sampler.Type,
        sigmas: torch.Tensor,
        latent: io.Latent.Type,
    ) -> io.Latent.Type:
        """Sample as SamplerCustomAdvanced does; returns a shallow copy of `latent` with new `samples`.

        Other entries, such as `batch_index` and `noise_mask`, are the same objects as in `latent`;
        the copy has no `downscale_ratio_spacial` or `downscale_ratio_temporal`. `noise.seed` also
        seeds the sampler's own randomness. Reports progress and previews as SamplerCustomAdvanced
        does.
        """
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

        return SamplerCustomAdvanced.execute(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        ).result[0]
