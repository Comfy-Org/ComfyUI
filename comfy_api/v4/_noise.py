from __future__ import annotations

from typing import Protocol

import torch

from comfy_api.internal.singleton import ProxiedSingleton
from comfy_api.latest import io


class NoiseGenerator(Protocol):
    """What ComfyUI uses from a NOISE value; a pack's own class may provide it.

    `seed` also seeds the sampler's own randomness, such as an ancestral sampler's.
    """

    seed: int

    def generate_noise(self, input_latent: io.Latent.Type) -> torch.Tensor:
        """Return noise shaped like `input_latent["samples"]`, at unit scale.

        Sampling scales it to the first noise level.
        """
        ...


class Noise(ProxiedSingleton):
    async def random(self, seed: int) -> NoiseGenerator:
        """Return seeded noise, as the RandomNoise node gives; the same as KSampler's for a seed."""
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        return Noise_RandomNoise(seed)

    async def empty(self) -> NoiseGenerator:
        """Return zero noise, as the DisableNoise node gives.

        Its seed is 0, so a sampler with its own randomness, such as an ancestral one, repeats the
        same randomness every run.
        """
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise

        return Noise_EmptyNoise()

    async def generate(
        self, noise: NoiseGenerator, latent: io.Latent.Type
    ) -> torch.Tensor:
        """Return the noise a NOISE value gives for `latent` as given.

        Sampling first fits an all-zero latent to the model's channels, so the noise it draws for
        such a latent can differ from this.
        """
        return noise.generate_noise(latent)
