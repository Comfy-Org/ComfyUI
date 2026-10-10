import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.modules.diffusionmodules.model import vae_attention
import math
import comfy.ops
ops = comfy.ops.disable_weight_init

def nonlinearity(x):
    # swish
    return torch.nn.functional.silu(x) / 0.596

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / math.sqrt((1 - t)**2 + t**2)

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(norm, eps) / math.sqrt(norm.numel() / x.numel())
    return x / norm.to(x.dtype)

class ResnetBlock1D(nn.Module):

    def __init__(self, *, in_dim, out_dim=None, conv_shortcut=False, kernel_size=3, use_norm=True):
        super().__init__()
        self.in_dim = in_dim
        out_dim = in_dim if out_dim is None else out_dim
        self.out_dim = out_dim
        self.use_conv_shortcut = conv_shortcut
        self.use_norm = use_norm

        self.conv1 = ops.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = ops.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        if self.in_dim != self.out_dim:
            if self.use_conv_shortcut:
                self.conv_shortcut = ops.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            else:
                self.nin_shortcut = ops.Conv1d(in_dim, out_dim, kernel_size=1, padding=0, bias=False)

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
        self.qkv = ops.Conv1d(in_channels, in_channels * 3, kernel_size=1, padding=0, bias=False)
        self.proj_out = ops.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.optimized_attention = vae_attention()

    def forward(self, x):
        h = x
        y = self.qkv(h)
        y = y.reshape(y.shape[0], -1, 3, y.shape[-1])
        q, k, v = normalize(y, dim=1).unbind(2)

        h = self.optimized_attention(q, k, v)
        h = self.proj_out(h)

        return mp_sum(x, h, t=0.3)


class Upsample1D(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest-exact')  # support 3D tensor(B,C,T)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv1 = ops.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
            self.conv2 = ops.Conv1d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        if self.with_conv:
            x = self.conv1(x)

        x = F.avg_pool1d(x, kernel_size=2, stride=2)

        if self.with_conv:
            x = self.conv2(x)

        return x