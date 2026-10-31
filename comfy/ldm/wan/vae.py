# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from comfy.ldm.modules.diffusionmodules.model import vae_attention, torch_cat_if_needed

import comfy.ops
ops = comfy.ops.disable_weight_init

CACHE_T = 2


class CausalConv3d(ops.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None):
        if cache_x is None and x.shape[2] == 1:
            #Fast path - the op will pad for use by truncating the weight
            #and save math on a pile of zeros.
            return super().forward(x, autopad="causal_zero")

        if self._padding > 0:
            padding_needed = self._padding
            if cache_x is not None:
                cache_x = cache_x.to(x.device)
                padding_needed = max(0, padding_needed - cache_x.shape[2])
            padding_shape = list(x.shape)
            padding_shape[2] = padding_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            x = torch_cat_if_needed([padding, cache_x, x], dim=2)
            del cache_x

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


STRIP_ELEMS = 2 ** 24


def strip_apply(fn, x, scale=1, halo=1, out=None, accumulate=False):
    # fn(strip, first row) on row strips with `halo` context rows; out may be x itself, so each strip's last rows wait for the next to read them
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x, 0)
    size = x.shape[-2]
    step = -(-size // n)
    put = torch.Tensor.add_ if accumulate else torch.Tensor.copy_
    held = None
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo), lo).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        if held is not None:
            put(out.narrow(-2, a * scale - held.shape[-2], held.shape[-2]), held)
        keep = min(halo, b - a)
        put(out.narrow(-2, a * scale, (b - a - keep) * scale), y.narrow(-2, 0, (b - a - keep) * scale))
        held = y.narrow(-2, (b - a - keep) * scale, keep * scale).clone() if keep else None
    if held is not None:
        put(out.narrow(-2, size * scale - held.shape[-2], held.shape[-2]), held)
    return out


def cached_chain(layers, x, feat_cache=None, feat_idx=[0]):
    # norm, act and conv layers as one strip function; each conv sees its cached frames' matching rows, and the
    # strips rebuild the caches from the chunk's last frames
    convs = [layer for layer in layers if isinstance(layer, CausalConv3d)]
    old = new = [None] * len(convs)
    if feat_cache is not None:
        i = feat_idx[0]
        feat_idx[0] += len(convs)
        old = feat_cache[i:i + len(convs)]
        new = [x.new_empty(x.shape[0], conv.in_channels, min(CACHE_T, x.shape[2]), *x.shape[-2:]) for conv in convs]
        feat_cache[i:i + len(convs)] = new

    def fn(s, lo):
        k = 0
        for layer in layers:
            if isinstance(layer, CausalConv3d):
                if new[k] is not None:
                    # the k convs before this one spoiled k rows at the strip's top edge; the bottom ones get rewritten by the next strip
                    off = min(lo, k)
                    new[k].narrow(-2, lo + off, s.shape[-2] - off).copy_(s[:, :, -CACHE_T:, off:])
                s = layer(s, None if old[k] is None else old[k].narrow(-2, lo, s.shape[-2]))
                k += 1
            else:
                s = layer(s)
        return s
    return fn


def strip_residual(block, x, feat_cache=None, feat_idx=[0]):
    # strips accumulate into x, or land in its leading channels when the block narrows
    fn = cached_chain(block.residual, x, feat_cache, feat_idx)
    if block.in_dim == block.out_dim:
        return strip_apply(fn, x, halo=2, out=x, accumulate=True)
    out = x.narrow(1, 0, block.out_dim) if block.in_dim > block.out_dim else None
    return strip_apply(lambda s, lo: fn(s, lo).add_(block.shortcut(s)), x, halo=2, out=out)


def strip_down(resample, x):
    # the stride-2 conv with its (0, 1, 0, 1) pad in strips: output rows [a, b) read input rows [2a, 2b + 1)
    rows = x.shape[-2] // 2
    n = -(-x.numel() // STRIP_ELEMS)
    if n <= 1:
        return resample(x)
    step = -(-rows // n)
    out = None
    for a in range(0, rows, step):
        b = min(rows, a + step)
        hi = min(x.shape[-2], 2 * b + 1)
        y = resample[1](F.pad(x.narrow(-2, 2 * a, hi - 2 * a), (0, 1, 0, int(hi == x.shape[-2]))))
        if out is None:
            out = y.new_empty(*y.shape[:-2], rows, y.shape[-1])
        out.narrow(-2, a, b - a).copy_(y)
    return out


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                ops.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        if self.mode != 'none':
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            if self.mode in ('upsample2d', 'upsample3d'):
                x = strip_apply(lambda s, lo: self.resample(s), x, scale=2)
            else:
                x = strip_down(self.resample, x)
            x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x[:, :, -1:, :, :].clone()  # the next block accumulates into x
                else:

                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x

                    deferred_x = feat_cache[idx + 1]
                    if deferred_x is not None:
                        x = torch.cat([deferred_x, x], 2)
                        feat_cache[idx + 1] = None

                    if x.shape[2] == 1 and not final:
                        feat_cache[idx + 1] = x
                        x = None

                feat_idx[0] += 2
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        return strip_residual(self, x, feat_cache, feat_idx)


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = ops.Conv2d(dim, dim * 3, 1)
        self.proj = ops.Conv2d(dim, dim, 1)
        self.optimized_attention = vae_attention()

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = self.optimized_attention(q, k, v)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 input_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
                if x is None:
                    return None
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, final=final)
            else:
                x = layer(x)

        ## head
        return strip_apply(cached_chain(self.head, x, feat_cache, feat_idx), x)


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 output_channels=3,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1))

    def run_up(self, layer_idx, x_ref, feat_cache, feat_idx, out_chunks):
        x = x_ref[0]
        x_ref[0] = None
        if layer_idx >= len(self.upsamples):
            out_chunks.append(strip_apply(cached_chain(self.head, x, feat_cache, feat_idx), x))
            return

        layer = self.upsamples[layer_idx]
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

        if isinstance(layer, Resample) and layer.mode == 'upsample3d' and x.shape[2] > 2:
            for frame_idx in range(0, x.shape[2], 2):
                self.run_up(
                    layer_idx + 1,
                    [x[:, :, frame_idx:frame_idx + 2, :, :]],
                    feat_cache,
                    feat_idx.copy(),
                    out_chunks,
                )
            del x
            return

        next_x_ref = [x]
        del x
        self.run_up(layer_idx + 1, next_x_ref, feat_cache, feat_idx, out_chunks)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        out_chunks = []

        self.run_up(0, [x], feat_cache, feat_idx, out_chunks)
        return out_chunks


def count_cache_layers(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d) or (isinstance(m, Resample) and m.mode == 'downsample3d'):
            count += 1
    return count


class WanVAE(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 image_channels=3,
                 conv_out_channels=3,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, image_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, conv_out_channels, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def encode(self, x):
        conv_idx = [0]
        ## cache
        t = x.shape[2]
        t = 1 + ((t - 1) // 4) * 4
        iter_ = 1 + (t - 1) // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.encoder)
        ## 对encode输入的x，按时间拆分为1、2、2、2....(总帧数先按4N+1向下取整)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                    final=(i == (iter_ - 1)))
                if out_ is None:
                    continue
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z):
        # z: [b,c,t,h,w]
        iter_ = 1 + z.shape[2] // 2
        feat_map = None
        if iter_ > 1:
            feat_map = [None] * count_cache_layers(self.decoder)
        x = self.conv2(z)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, 1 + 2 * (i - 1):1 + 2 * i, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx)
                out += out_
        return torch.cat(out, 2)
