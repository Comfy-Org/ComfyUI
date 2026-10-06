from typing import Optional
import torch
import numpy as np


class DiagonalGaussianDistribution:

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, rng: Optional[torch.Generator] = None):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

        r = torch.empty_like(self.mean).normal_(generator=rng)
        x = self.mean + self.std * r

        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:

                return 0.5 * torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
            else:
                return 0.5 * (torch.pow(self.mean - other.mean, 2) / other.var +
                              self.var / other.var - 1.0 - self.logvar + other.logvar)

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
                               dim=dims)

    def mode(self):
        return self.mean