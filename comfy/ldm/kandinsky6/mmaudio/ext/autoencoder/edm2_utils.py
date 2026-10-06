# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
# Improved diffusion model architecture proposed in the paper
# "Analyzing and Improving the Training Dynamics of Diffusion Models".

import numpy as np
import torch

#----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)


#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class Normalize(torch.nn.Module):

    def __init__(self, dim=None, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return normalize(x, dim=self.dim, eps=self.eps)


#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x,
                                          f.tile([c, 1, 1, 1]),
                                          groups=c,
                                          stride=2,
                                          padding=(pad, ))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]),
                                                groups=c,
                                                stride=2,
                                                padding=(pad, ))


#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


class MPSiLU(torch.nn.Module):

    def forward(self, x):
        return mp_silu(x)


#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t)**2 + t**2)


#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t)**2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv1D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

        self.weight_norm_removed = False

    def forward(self, x, gain=1):
        assert self.weight_norm_removed, 'call remove_weight_norm() before inference'

        w = self.weight * gain
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 3
        return torch.nn.functional.conv1d(x, w, padding=(w.shape[-1] // 2, ))

    def remove_weight_norm(self):
        w = self.weight.to(torch.float32)
        w = normalize(w)  # traditional weight normalization
        w = w / np.sqrt(w[0].numel())
        w = w.to(self.weight.dtype)
        self.weight.data.copy_(w)

        self.weight_norm_removed = True
        return self
