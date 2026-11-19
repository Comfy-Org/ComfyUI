from ..diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ..diffusionmodules.openaimodel import Timestep
import torch

class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean.to(x.device)) * 1. / self.data_std.to(x.device)
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std.to(x.device)) + self.data_mean.to(x.device)
        return x

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level, seed=seed)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level
