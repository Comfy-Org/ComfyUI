import comfy_kitchen
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD


SLAB = 16  # latent frames per slab; the GroupNorm statistics stay clip-wide, so slabbing is exact


def clip_stats(x, norm):
    n, c, t, h, w = x.shape
    return comfy_kitchen.group_norm_stats3d(x.reshape(n, c, 1, t * h, w), norm.num_groups, norm.eps)


def slabs(t, halo):
    for t0 in range(0, t, SLAB):
        t1 = min(t0 + SLAB, t)
        yield t0, t1, max(t0 - halo, 0), min(t1 + halo, t)


def norm_silu(x, norm, stats, scale=None, shift=None):
    """SiLU(GroupNorm(x) * (1 + scale) + shift) of a slab, normalized with the clip's stats; channels-last out."""
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        w, b = weight, bias
        if scale is not None:  # the modulation folds into the norm affine
            w, b = weight * (1 + scale), torch.addcmul(shift, bias, 1 + scale)
        n, c, t, h, wd = x.shape
        # kitchen normalizes per frame; one frame spanning the clip gives nn.GroupNorm's statistics
        out = comfy_kitchen.group_norm_silu_pad3d(x.reshape(n, c, 1, t * h, wd), w, b, norm.num_groups, norm.eps, (0, 0, 0, 0, 0), silu=True, stats=stats)
        return out.view(n, c, t, h, wd)
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)


def conv3d_cl(conv, x):
    # cast weights arrive channels-first; matching the activations keeps cuDNN and the residual adds on fast paths
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(conv, x, offloadable=True)
    try:
        return F.conv3d(x, weight.contiguous(memory_format=torch.channels_last_3d), bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    finally:
        comfy.ops.uncast_bias_weight(conv, weight, bias, offload_stream)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dtype=None, device=None, operations=None):
        super().__init__()
        # Identity slots keep the checkpoint's module indices (the activations run in norm_silu)
        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device), nn.Identity(),
            operations.Conv3d(channels, channels, 3, padding=1, dtype=dtype, device=device))
        self.emb_layers = nn.Sequential(nn.SiLU(), operations.Linear(emb_channels, 2 * channels, dtype=dtype, device=device))
        self.out_norm = operations.GroupNorm(32, channels, dtype=dtype, device=device)
        self.out_layers = nn.Sequential(
            nn.Identity(), nn.Identity(),
            operations.Conv3d(channels, channels, 3, padding=1, dtype=dtype, device=device))

    def forward(self, x, emb):
        scale, shift = self.emb_layers(emb)[0].chunk(2)
        h = torch.empty_like(x)
        stats = clip_stats(x, self.in_layers[0])
        for t0, t1, a, b in slabs(x.shape[2], 1):
            h[:, :, t0:t1] = conv3d_cl(self.in_layers[2], norm_silu(x[:, :, a:b], self.in_layers[0], stats))[:, :, t0 - a:t1 - a]
        stats = clip_stats(h, self.out_norm)
        for t0, t1, a, b in slabs(x.shape[2], 1):
            x[:, :, t0:t1] += conv3d_cl(self.out_layers[2], norm_silu(h[:, :, a:b], self.out_norm, stats, scale, shift))[:, :, t0 - a:t1 - a]
        return x


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.GroupNorm(32, channels, dtype=dtype, device=device)
        self.dwconv = operations.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0), groups=channels, dtype=dtype, device=device)
        self.pwconv = operations.Conv3d(channels, channels, 1, dtype=dtype, device=device)

    def forward(self, x):
        out = torch.empty_like(x)
        stats = clip_stats(x, self.norm)
        # the depthwise temporal conv as shifted multiply-adds: torch's depthwise conv3d has no channels-last kernel
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self.dwconv, x, offloadable=True)
        try:
            k = weight.shape[2]
            for t0, t1, a, b in slabs(x.shape[2], k // 2):
                h = norm_silu(x[:, :, a:b], self.norm, stats)
                acc = torch.empty_like(h).copy_(bias.view(1, -1, 1, 1, 1))
                for i in range(k):
                    s = i - k // 2
                    acc[:, :, max(0, -s):b - a - max(0, s)].addcmul_(h[:, :, max(0, s):b - a + min(0, s)], weight[:, 0, i].view(1, -1, 1, 1, 1))
                out[:, :, t0:t1] = self.pwconv(acc[:, :, t0 - a:t1 - a])
        finally:
            comfy.ops.uncast_bias_weight(self.dwconv, weight, bias, offload_stream)
        return x.add_(out)


class MiniMaxH3LatentUpscaler(nn.Module):
    """3D-conv blocks at the input size, a trilinear resize, then blocks at the output size; every res block is
    conditioned on the scale factor and every other one is followed by a temporal conv."""

    def __init__(self, in_channels=24, channels=512, in_blocks=12, out_blocks=12, temporal_kernel=5, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        ops = {"dtype": dtype, "device": device, "operations": operations}
        self.conv_in = operations.Conv3d(in_channels, channels, 3, padding=1, dtype=dtype, device=device)
        self.embed = nn.Sequential(operations.Linear(1, 64, dtype=dtype, device=device), nn.SiLU(), operations.Linear(64, 64, dtype=dtype, device=device))
        self.in_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList()
        for blocks, count in ((self.in_blocks, in_blocks), (self.out_blocks, out_blocks)):
            for i in range(count):
                blocks.append(ResBlockEmb3D(channels, 64, **ops))
                if i % 2 == 0:
                    blocks.append(TemporalConv(channels, temporal_kernel, **ops))
        self.norm_out = operations.GroupNorm(32, channels, dtype=dtype, device=device)
        self.conv_out = operations.Conv3d(channels, in_channels, 3, padding=1, dtype=dtype, device=device)

    def get_dtype(self):
        return self.dtype

    def memory_required(self, shape, size):
        # two full-size activations plus slab temporaries worth ~75 frames (measured), at the larger of the two stages
        n, _, t, h, w = shape
        return n * self.conv_in.out_channels * (2 * t + 96) * max(h * w, size[1] * size[2]) * self.dtype.itemsize

    def run_blocks(self, blocks, x, emb):
        for block in blocks:
            x = block(x, emb) if isinstance(block, ResBlockEmb3D) else block(x)
        return x

    def forward(self, x, size):
        """Normalized H3 video latent [B, 24, T, H, W] -> the same at size (T, H', W')."""
        scale = (size[-1] / x.shape[-1] + size[-2] / x.shape[-2]) / 2
        mean = torch.tensor(LATENTS_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        # trained on VAE latents normalized a second time with the same stats
        x = (x - mean) / std
        emb = self.embed(torch.full((1, 1), scale - 1, device=x.device, dtype=x.dtype))
        # activations stay channels-last, the layout norm_silu emits and the convs consume natively
        x = self.run_blocks(self.in_blocks, self.conv_in(x).contiguous(memory_format=torch.channels_last_3d), emb)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False).contiguous(memory_format=torch.channels_last_3d)
        x = self.run_blocks(self.out_blocks, x, emb)
        out = torch.empty((x.shape[0], self.conv_out.out_channels) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)
        stats = clip_stats(x, self.norm_out)
        for t0, t1, a, b in slabs(x.shape[2], 1):
            out[:, :, t0:t1] = conv3d_cl(self.conv_out, norm_silu(x[:, :, a:b], self.norm_out, stats))[:, :, t0 - a:t1 - a]
        return torch.addcmul(mean, out, std)
