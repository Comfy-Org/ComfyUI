import torch
import torch.nn as nn
import torch.nn.functional as F

from .edm2_utils import (MPConv1D, mp_silu, mp_sum, normalize)


def nonlinearity(x):
    # swish
    return mp_silu(x)


class ResnetBlock1D(nn.Module):

    def __init__(self, *, in_dim, out_dim=None, conv_shortcut=False, kernel_size=3, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim
        self.use_conv_shortcut = conv_shortcut
        self.use_norm = use_norm

        self.conv1 = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
        self.conv2 = MPConv1D(out_dim, out_dim, kernel_size=kernel_size)
        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                self.conv_shortcut = MPConv1D(in_dim, out_dim, kernel_size=kernel_size)
            else:
                self.nin_shortcut = MPConv1D(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # pixel norm
        if self.use_norm:
            x = normalize(x, dim=1)

        h = x
        h = nonlinearity(h)
        h = self.conv1(h)

        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return mp_sum(x, h, t=0.3)


class AttnBlock1D(nn.Module):

    def __init__(self, in_channels, num_heads=1):
        super().__init__()
        self.in_channels = in_channels

        self.num_heads = num_heads
        self.qkv = MPConv1D(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = MPConv1D(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = x
        y = self.qkv(h)
        y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[-1])
        q, k, v = normalize(y, dim=2).unbind(3)

        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(h.shape[0], -1, h.shape[2])

        h = self.proj_out(h)

        return mp_sum(x, h, t=0.3)


class Upsample1D(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = MPConv1D(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest-exact')  # support 3D tensor(B,C,T)
        if self.with_conv:
            x = self.conv(x)
        return x
