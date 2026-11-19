# CogVideoX VAE - ported to ComfyUI native ops
# Architecture reference: diffusers AutoencoderKLCogVideoX
# Style reference: comfy/ldm/wan/vae.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init


class CausalConv3d(nn.Module):
    """Causal 3D convolution with temporal padding.

    Uses comfy.ops.Conv3d with autopad='causal_zero' fast path: when input has
    a single temporal frame and no cache, the 3D conv weight is sliced to act
    as a 2D conv, avoiding computation on zero-padded temporal dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode="constant"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel, height_kernel, width_kernel = kernel_size
        self.time_kernel_size = time_kernel
        self.pad_mode = pad_mode

        height_pad = (height_kernel - 1) // 2
        width_pad = (width_kernel - 1) // 2
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_kernel - 1, 0)

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = ops.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=(0, height_pad, width_pad),
        )

    def forward(self, x, conv_cache=None):
        if self.pad_mode == "replicate":
            x = F.pad(x, self.time_causal_padding, mode="replicate")
            conv_cache = None
        else:
            kernel_t = self.time_kernel_size
            if kernel_t > 1:
                if conv_cache is None and x.shape[2] == 1:
                    # Fast path: single frame, no cache. All temporal padding
                    # frames are copies of the input (replicate-style), so the
                    # 3D conv reduces to a 2D conv with summed temporal kernel.
                    w = comfy.ops.cast_to_input(self.conv.weight, x)
                    b = comfy.ops.cast_to_input(self.conv.bias, x) if self.conv.bias is not None else None
                    w2d = w.sum(dim=2, keepdim=True)
                    out = F.conv3d(x, w2d, b,
                                   self.conv.stride, self.conv.padding,
                                   self.conv.dilation, self.conv.groups)
                    return out, None
                cached = [conv_cache] if conv_cache is not None else [x[:, :, :1]] * (kernel_t - 1)
                x = torch.cat(cached + [x], dim=2)
            conv_cache = x[:, :, -self.time_kernel_size + 1:].clone() if self.time_kernel_size > 1 else None

        out = self.conv(x)
        return out, conv_cache


def _interpolate_zq(zq, target_size):
    """Interpolate latent z to target (T, H, W), matching CogVideoX's first-frame-special handling."""
    t = target_size[0]
    if t > 1 and t % 2 == 1:
        z_first = F.interpolate(zq[:, :, :1], size=(1, target_size[1], target_size[2]))
        z_rest = F.interpolate(zq[:, :, 1:], size=(t - 1, target_size[1], target_size[2]))
        return torch.cat([z_first, z_rest], dim=2)
    return F.interpolate(zq, size=target_size)


class SpatialNorm3D(nn.Module):
    """Spatially conditioned normalization."""
    def __init__(self, f_channels, zq_channels, groups=32):
        super().__init__()
        self.norm_layer = ops.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def forward(self, f, zq, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        if zq.shape[-3:] != f.shape[-3:]:
            zq = _interpolate_zq(zq, f.shape[-3:])

        conv_y, new_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        return self.norm_layer(f) * conv_y + conv_b, new_cache


class ResnetBlock3D(nn.Module):
    """3D ResNet block with optional spatial norm."""
    def __init__(self, in_channels, out_channels=None, temb_channels=512, groups=32,
                 eps=1e-6, act_fn="silu", spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_norm_dim = spatial_norm_dim

        if act_fn == "silu":
            self.nonlinearity = nn.SiLU()
        elif act_fn == "swish":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = nn.SiLU()

        if spatial_norm_dim is None:
            self.norm1 = ops.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = ops.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = SpatialNorm3D(in_channels, spatial_norm_dim, groups=groups)
            self.norm2 = SpatialNorm3D(out_channels, spatial_norm_dim, groups=groups)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels, out_channels)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

        if in_channels != out_channels:
            self.conv_shortcut = ops.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        residual = x

        if zq is not None:
            x, new_cache["norm1"] = self.norm1(x, zq, conv_cache=conv_cache.get("norm1"))
        else:
            x = self.norm1(x)

        x = self.nonlinearity(x)
        x, new_cache["conv1"] = self.conv1(x, conv_cache=conv_cache.get("conv1"))

        if temb is not None and hasattr(self, "temb_proj"):
            x = x + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            x, new_cache["norm2"] = self.norm2(x, zq, conv_cache=conv_cache.get("norm2"))
        else:
            x = self.norm2(x)

        x = self.nonlinearity(x)
        x, new_cache["conv2"] = self.conv2(x, conv_cache=conv_cache.get("conv2"))

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual, new_cache


class Downsample3D(nn.Module):
    """3D downsampling with optional temporal compression."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, compress_time=False):
        super().__init__()
        self.conv = ops.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            b, c, t, h, w = x.shape
            x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
            if t % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)

        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


class Upsample3D(nn.Module):
    """3D upsampling with optional temporal decompression."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, compress_time=False):
        super().__init__()
        self.conv = ops.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            if x.shape[2] > 1 and x.shape[2] % 2 == 1:
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]
                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            elif x.shape[2] > 1:
                x = F.interpolate(x, scale_factor=2.0)
            else:
                x = x.squeeze(2)
                x = F.interpolate(x, scale_factor=2.0)
                x = x[:, :, None, :, :]
        else:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, scale_factor=2.0)
            x = x.reshape(b, t, c, *x.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        return x


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, add_downsample=True,
                 compress_time=False, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=groups, eps=eps, act_fn=act_fn, pad_mode=pad_mode,
            )
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample3D(out_channels, out_channels, compress_time=compress_time)]) if add_downsample else None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x, new_cache


class MidBlock3D(nn.Module):
    def __init__(self, in_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels, groups=groups, eps=eps,
                act_fn=act_fn, spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        return x, new_cache


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, spatial_norm_dim=16,
                 add_upsample=True, compress_time=False, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels, groups=groups, eps=eps,
                act_fn=act_fn, spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode,
            )
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsample3D(out_channels, out_channels, compress_time=compress_time)]) if add_upsample else None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        if self.upsamplers is not None:
            for us in self.upsamplers:
                x = us(x)
        return x, new_cache


class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=16,
                 block_out_channels=(128, 256, 256, 512),
                 layers_per_block=3, act_fn="silu",
                 eps=1e-6, groups=32, pad_mode="first",
                 temporal_compression_ratio=4):
        super().__init__()
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            self.down_blocks.append(DownBlock3D(
                in_channels=input_channel, out_channels=output_channel,
                temb_channels=0, num_layers=layers_per_block,
                eps=eps, act_fn=act_fn, groups=groups,
                add_downsample=not is_final, compress_time=compress_time,
            ))

        self.mid_block = MidBlock3D(
            in_channels=block_out_channels[-1], temb_channels=0,
            num_layers=2, eps=eps, act_fn=act_fn, groups=groups, pad_mode=pad_mode,
        )

        self.norm_out = ops.GroupNorm(groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, x, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        x, new_cache["conv_in"] = self.conv_in(x, conv_cache=conv_cache.get("conv_in"))

        for i, block in enumerate(self.down_blocks):
            key = f"down_block_{i}"
            x, new_cache[key] = block(x, None, None, conv_cache.get(key))

        x, new_cache["mid_block"] = self.mid_block(x, None, None, conv_cache=conv_cache.get("mid_block"))

        x = self.norm_out(x)
        x = self.conv_act(x)
        x, new_cache["conv_out"] = self.conv_out(x, conv_cache=conv_cache.get("conv_out"))

        return x, new_cache


class Decoder3D(nn.Module):
    def __init__(self, in_channels=16, out_channels=3,
                 block_out_channels=(128, 256, 256, 512),
                 layers_per_block=3, act_fn="silu",
                 eps=1e-6, groups=32, pad_mode="first",
                 temporal_compression_ratio=4):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CausalConv3d(in_channels, reversed_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.mid_block = MidBlock3D(
            in_channels=reversed_channels[0], temb_channels=0,
            num_layers=2, eps=eps, act_fn=act_fn, groups=groups,
            spatial_norm_dim=in_channels, pad_mode=pad_mode,
        )

        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for i in range(len(block_out_channels)):
            prev_channel = output_channel
            output_channel = reversed_channels[i]
            is_final = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            self.up_blocks.append(UpBlock3D(
                in_channels=prev_channel, out_channels=output_channel,
                temb_channels=0, num_layers=layers_per_block + 1,
                eps=eps, act_fn=act_fn, groups=groups,
                spatial_norm_dim=in_channels,
                add_upsample=not is_final, compress_time=compress_time,
            ))

        self.norm_out = SpatialNorm3D(reversed_channels[-1], in_channels, groups=groups)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(reversed_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, sample, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        x, new_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        x, new_cache["mid_block"] = self.mid_block(x, None, sample, conv_cache=conv_cache.get("mid_block"))

        for i, block in enumerate(self.up_blocks):
            key = f"up_block_{i}"
            x, new_cache[key] = block(x, None, sample, conv_cache=conv_cache.get(key))

        x, new_cache["norm_out"] = self.norm_out(x, sample, conv_cache=conv_cache.get("norm_out"))
        x = self.conv_act(x)
        x, new_cache["conv_out"] = self.conv_out(x, conv_cache=conv_cache.get("conv_out"))

        return x, new_cache



class AutoencoderKLCogVideoX(nn.Module):
    """CogVideoX VAE. Spatial tiling/slicing handled by ComfyUI's VAE wrapper.

    Uses rolling temporal decode: conv_in + mid_block + temporal up_blocks run
    on the full (low-res) tensor, then the expensive spatial-only up_blocks +
    norm_out + conv_out are processed in small temporal chunks with conv_cache
    carrying causal state between chunks. This keeps peak VRAM proportional to
    chunk_size rather than total frame count.
    """

    def __init__(self,
                 in_channels=3, out_channels=3,
                 block_out_channels=(128, 256, 256, 512),
                 latent_channels=16, layers_per_block=3,
                 act_fn="silu", eps=1e-6, groups=32,
                 temporal_compression_ratio=4,
                 ):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_compression_ratio = temporal_compression_ratio

        self.encoder = Encoder3D(
            in_channels=in_channels, out_channels=latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            act_fn=act_fn, eps=eps, groups=groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = Decoder3D(
            in_channels=latent_channels, out_channels=out_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            act_fn=act_fn, eps=eps, groups=groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

    def encode(self, x):
        t = x.shape[2]
        frame_batch = self.num_sample_frames_batch_size
        remainder = t % frame_batch
        conv_cache = None
        enc = []

        # Process remainder frames first so only the first chunk can have an
        # odd temporal dimension — where Downsample3D's first-frame-special
        # handling in temporal compression is actually correct.
        if remainder > 0:
            chunk, conv_cache = self.encoder(x[:, :, :remainder], conv_cache=conv_cache)
            enc.append(chunk.to(x.device))

        for start in range(remainder, t, frame_batch):
            chunk, conv_cache = self.encoder(x[:, :, start:start + frame_batch], conv_cache=conv_cache)
            enc.append(chunk.to(x.device))

        enc = torch.cat(enc, dim=2)
        mean, _ = enc.chunk(2, dim=1)
        return mean

    def decode(self, z):
        return self._decode_rolling(z)

    def _decode_batched(self, z):
        """Original batched decode - processes 2 latent frames through full decoder."""
        t = z.shape[2]
        frame_batch = self.num_latent_frames_batch_size
        num_batches = max(t // frame_batch, 1)
        conv_cache = None
        dec = []
        for i in range(num_batches):
            remaining = t % frame_batch
            start = frame_batch * i + (0 if i == 0 else remaining)
            end = frame_batch * (i + 1) + remaining
            chunk, conv_cache = self.decoder(z[:, :, start:end], conv_cache=conv_cache)
            dec.append(chunk.cpu())
        return torch.cat(dec, dim=2).to(z.device)

    def _decode_rolling(self, z):
        """Rolling decode - processes low-res layers on full tensor, then rolls
        through expensive high-res layers in temporal chunks."""
        decoder = self.decoder
        device = z.device

        # Determine which up_blocks have temporal upsample vs spatial-only.
        # Temporal up_blocks are cheap (low res), spatial-only are expensive.
        temporal_compress_level = int(np.log2(self.temporal_compression_ratio))
        split_at = temporal_compress_level  # first N up_blocks do temporal upsample

        # Phase 1: conv_in + mid_block + temporal up_blocks on full tensor (low/medium res)
        x, _ = decoder.conv_in(z)
        x, _ = decoder.mid_block(x, None, z)

        for i in range(split_at):
            x, _ = decoder.up_blocks[i](x, None, z)

        # Phase 2: remaining spatial-only up_blocks + norm_out + conv_out in temporal chunks
        remaining_blocks = list(range(split_at, len(decoder.up_blocks)))
        chunk_size = 4  # pixel frames per chunk through high-res layers
        t_expanded = x.shape[2]

        if t_expanded <= chunk_size or len(remaining_blocks) == 0:
            # Small enough to process in one go
            for i in remaining_blocks:
                x, _ = decoder.up_blocks[i](x, None, z)
            x, _ = decoder.norm_out(x, z)
            x = decoder.conv_act(x)
            x, _ = decoder.conv_out(x)
            return x

        # Expand z temporally once to match Phase 2's time dimension.
        # z stays at latent spatial resolution so this is small (~16 MB vs ~1.3 GB
        # for the old approach of pre-interpolating to every pixel resolution).
        z_time_expanded = _interpolate_zq(z, (t_expanded, z.shape[3], z.shape[4]))

        # Process in temporal chunks, interpolating spatially per-chunk to avoid
        # allocating full [B, C, t_expanded, H, W] tensors at each resolution.
        dec_out = []
        conv_caches = {}

        for chunk_start in range(0, t_expanded, chunk_size):
            chunk_end = min(chunk_start + chunk_size, t_expanded)
            x_chunk = x[:, :, chunk_start:chunk_end]
            z_t_chunk = z_time_expanded[:, :, chunk_start:chunk_end]
            z_spatial_cache = {}

            for i in remaining_blocks:
                block = decoder.up_blocks[i]
                cache_key = f"up_block_{i}"
                hw_key = (x_chunk.shape[3], x_chunk.shape[4])
                if hw_key not in z_spatial_cache:
                    if z_t_chunk.shape[3] == hw_key[0] and z_t_chunk.shape[4] == hw_key[1]:
                        z_spatial_cache[hw_key] = z_t_chunk
                    else:
                        z_spatial_cache[hw_key] = F.interpolate(z_t_chunk, size=(z_t_chunk.shape[2], hw_key[0], hw_key[1]))
                x_chunk, new_cache = block(x_chunk, None, z_spatial_cache[hw_key], conv_cache=conv_caches.get(cache_key))
                conv_caches[cache_key] = new_cache

            hw_key = (x_chunk.shape[3], x_chunk.shape[4])
            if hw_key not in z_spatial_cache:
                z_spatial_cache[hw_key] = F.interpolate(z_t_chunk, size=(z_t_chunk.shape[2], hw_key[0], hw_key[1]))
            x_chunk, new_cache = decoder.norm_out(x_chunk, z_spatial_cache[hw_key], conv_cache=conv_caches.get("norm_out"))
            conv_caches["norm_out"] = new_cache
            x_chunk = decoder.conv_act(x_chunk)
            x_chunk, new_cache = decoder.conv_out(x_chunk, conv_cache=conv_caches.get("conv_out"))
            conv_caches["conv_out"] = new_cache

            dec_out.append(x_chunk.cpu())
            del z_spatial_cache

        del x, z_time_expanded
        return torch.cat(dec_out, dim=2).to(device)
