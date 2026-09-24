# original version: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .vae import AttentionBlock, CausalConv3d, RMS_norm

import comfy.ops
ops = comfy.ops.disable_weight_init

CACHE_T = 2


class Resample(nn.Module):

    def __init__(self, dim, mode, temporal_kernel=3):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ops.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ops.Conv2d(dim, dim, 3, padding=1),
                # ops.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (temporal_kernel, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] != "Rep"):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                    cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] == "Rep"):
                        cache_x = torch.cat(
                            [
                                torch.zeros_like(cache_x).to(cache_x.device),
                                cache_x
                            ],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
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
        x = rearrange(x, "b c t h w -> (b t) c h w")
        if feat_cache is None and self.mode in ("upsample2d", "upsample3d"):
            x = strip_apply(self.resample, x, scale=2)
        else:
            x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


STRIP_ELEMS = 2 ** 24


def strip_apply(fn, x, scale=1, halo=1, out=None):
    # strips of rows bound cudnn's conv workspace, a halo row per 3x3 conv keeps them exact
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, a * scale, (b - a) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


def conv3x3(in_dim, out_dim, temporal_kernel=3):
    return CausalConv3d(in_dim, out_dim, (temporal_kernel, 3, 3), padding=(temporal_kernel // 2, 1, 1))


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0, temporal_kernel=3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            conv3x3(in_dim, out_dim, temporal_kernel),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv3x3(out_dim, out_dim, temporal_kernel),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity())

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is None:
            # single image: the whole block runs in strips so its intermediates never exist at full size
            return strip_apply(lambda s: self.residual(s).add_(self.shortcut(s)), x, halo=2)
        old_x = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, cache_list=feat_cache, cache_idx=idx)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + self.shortcut(old_x)


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_downsample=False,
                 down_flag=False,
                 temporal_kernel=3):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_upsample=False,
                 up_flag=False,
                 temporal_kernel=3):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            if feat_cache is None:
                return strip_apply(lambda s: self.avg_shortcut(s, first_chunk), x, scale=self.avg_shortcut.factor_s, halo=0, out=x_main)
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        in_channels=12,
        temporal_kernel=3,
    ):
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
        self.conv1 = conv3x3(in_channels, dims[0], temporal_kernel)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i]
                if i < len(temperal_downsample) else False)
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                    temporal_kernel=temporal_kernel,
                ))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            conv3x3(out_dim, z_dim, temporal_kernel),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        out_channels=12,
        temporal_kernel=3,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        # init block
        self.conv1 = conv3x3(z_dim, dims[0], temporal_kernel)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(
                temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                    temporal_kernel=temporal_kernel,
                ))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            conv3x3(out_dim, out_channels, temporal_kernel),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x, first_chunk=first_chunk)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE(nn.Module):

    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        image_channels=3,
        patch_size=2,
        temporal_kernel=3,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.patch_size = patch_size

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
            in_channels=image_channels * patch_size * patch_size,
            temporal_kernel=temporal_kernel,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            out_channels=image_channels * patch_size * patch_size,
            temporal_kernel=temporal_kernel,
        )

    def encode(self, x):
        if x.ndim == 4:
            # single image: no temporal cache, which would keep every conv input alive for the whole pass
            x = patchify(x.unsqueeze(2), patch_size=self.patch_size)
            return self.conv1(self.encoder(x)).chunk(2, dim=1)[0].squeeze(2)
        conv_idx = [0]
        feat_map = [None] * count_conv3d(self.encoder)
        x = patchify(x, patch_size=self.patch_size)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z):
        if z.ndim == 4:
            out = self.decoder(self.conv2(z.unsqueeze(2)), first_chunk=True)
            return unpatchify(out, patch_size=self.patch_size).squeeze(2)
        conv_idx = [0]
        feat_map = [None] * count_conv3d(self.decoder)
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=feat_map,
                    feat_idx=conv_idx,
                )
                out = torch.cat([out, out_], 2)
        out = unpatchify(out, patch_size=self.patch_size)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)
