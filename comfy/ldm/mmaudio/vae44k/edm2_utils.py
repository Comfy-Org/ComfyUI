import math

import torch


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


class MPConv1D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))

    def forward(self, x, gain=None):
        weight = self.weight if gain is None else self.weight * gain
        return torch.nn.functional.conv1d(x, weight, padding=(weight.shape[-1] // 2, ))
