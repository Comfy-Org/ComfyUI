# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

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
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

        self.weight_norm_removed = False

    def forward(self, x, gain=1):
        assert self.weight_norm_removed, 'call remove_weight_norm() before inference'
        weight = self.weight * gain
        return torch.nn.functional.conv1d(x, weight, padding=(weight.shape[-1] // 2, ))

    def remove_weight_norm(self):
        w = self.weight.to(torch.float32)
        w = normalize(w)  # traditional weight normalization
        w = w / math.sqrt(w[0].numel())
        w = w.to(self.weight.dtype)
        self.weight.data.copy_(w)

        self.weight_norm_removed = True
        return self
