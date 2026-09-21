from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from contextlib import contextmanager
from comfy.utils import ProgressBar

from comfy.ldm.seedvr.constants import (
    BYTEDANCE_BLOCK_OUT_CHANNELS,
    BYTEDANCE_GN_CHUNKS_FP16,
    BYTEDANCE_GN_CHUNKS_FP32,
    BYTEDANCE_LOGVAR_CLAMP_MAX,
    BYTEDANCE_LOGVAR_CLAMP_MIN,
    BYTEDANCE_SLICING_SAMPLE_MIN,
    BYTEDANCE_VAE_CONV_MEM_GIB,
    BYTEDANCE_VAE_NORM_MEM_GIB,
    BYTEDANCE_VAE_SCALING_FACTOR,
    BYTEDANCE_VAE_SHIFTING_FACTOR,
    BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE,
    BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE,
    SEEDVR2_LATENT_CHANNELS,
)
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.model import vae_attention

import math
from enum import Enum

import logging
import comfy.model_management
import comfy.ops
ops = comfy.ops.manual_cast


def _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap, temporal_scale=1):
    if temporal_size is None:
        return None

    temporal_size = int(temporal_size)
    if temporal_size <= 0:
        return None

    temporal_overlap = max(0, int(temporal_overlap or 0))
    temporal_overlap = min(temporal_overlap, temporal_size - 1)
    temporal_step = temporal_size - temporal_overlap
    temporal_scale = max(1, int(temporal_scale))
    return max(1, math.ceil(temporal_step / temporal_scale))


def _seedvr2_clamped_spatial_overlap(overlap, tile_size):
    overlap = max(0, int(overlap))
    tile_size = max(1, int(tile_size))
    return min(overlap, tile_size - 1)


def tiled_vae(
    x,
    vae_model,
    tile_size=(512, 512),
    tile_overlap=(64, 64),
    temporal_size=16,
    temporal_overlap=0,
    encode=True,
):
    if x.ndim != 5:
        x = x.unsqueeze(2)

    _, _, d, h, w = x.shape

    sf_s = getattr(vae_model, "spatial_downsample_factor", BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE)
    sf_t = getattr(vae_model, "temporal_downsample_factor", BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE)
    if encode:
        slicing_attr = "slicing_sample_min_size"
        slicing_min_size = _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap)
    else:
        slicing_attr = "slicing_latent_min_size"
        slicing_min_size = _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap, sf_t)
    if encode:
        ti_h, ti_w = tile_size
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0], ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1], ti_w)
        blend_ov_h = max(0, ov_h // sf_s)
        blend_ov_w = max(0, ov_w // sf_s)
        target_d = (d + sf_t - 1) // sf_t
        target_h = (h + sf_s - 1) // sf_s
        target_w = (w + sf_s - 1) // sf_s
    else:
        ti_h = max(1, tile_size[0] // sf_s)
        ti_w = max(1, tile_size[1] // sf_s)
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0] // sf_s, ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1] // sf_s, ti_w)
        blend_ov_h = ov_h * sf_s
        blend_ov_w = ov_w * sf_s

        target_d = max(1, d * sf_t - (sf_t - 1))
        target_h = h * sf_s
        target_w = w * sf_s

    stride_h = max(1, ti_h - ov_h)
    stride_w = max(1, ti_w - ov_w)

    storage_device = vae_model.device
    result = None
    count = None
    def run_temporal_chunks(spatial_tile, model=vae_model):
        t_chunk = spatial_tile.contiguous()
        old_device = getattr(model, "device", None)
        model.device = t_chunk.device
        old_slicing_min_size = getattr(model, slicing_attr, None)
        if old_slicing_min_size is not None and slicing_min_size is not None:
            if slicing_min_size <= 0:
                setattr(model, slicing_attr, t_chunk.shape[2])
            else:
                setattr(model, slicing_attr, slicing_min_size)
        try:
            if encode:
                out = model.encode(t_chunk)
            else:
                out = model.decode_(t_chunk)
        finally:
            if old_slicing_min_size is not None and slicing_min_size is not None:
                setattr(model, slicing_attr, old_slicing_min_size)
            if old_device is not None:
                model.device = old_device
        if out.ndim == 4:
            out = out.unsqueeze(2)
        return out.to(storage_device)

    ramp_cache = {}
    def get_ramp(steps):
        if steps not in ramp_cache:
            t = torch.linspace(0, 1, steps=steps, device=storage_device, dtype=torch.float32)
            ramp_cache[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramp_cache[steps]

    tile_ranges = []
    for y_idx in range(0, h, stride_h):
        y_end = min(y_idx + ti_h, h)
        if y_idx > 0 and (y_end - y_idx) <= ov_h:
            continue
        for x_idx in range(0, w, stride_w):
            x_end = min(x_idx + ti_w, w)
            if x_idx > 0 and (x_end - x_idx) <= ov_w:
                continue
            tile_ranges.append((y_idx, y_end, x_idx, x_end))

    total_tiles = len(tile_ranges)
    bar = ProgressBar(total_tiles)
    single_spatial_tile = h <= ti_h and w <= ti_w

    def run_tile(tile_index, tile_range):
        y_idx, y_end, x_idx, x_end = tile_range
        tile_x = x[:, :, :, y_idx:y_end, x_idx:x_end]
        tile_out = run_temporal_chunks(tile_x)
        return tile_index, y_idx, y_end, x_idx, x_end, tile_out

    ordered_tile_outputs = (
        run_tile(tile_index, tile_range)
        for tile_index, tile_range in enumerate(tile_ranges)
    )

    for _, y_idx, y_end, x_idx, x_end, tile_out in ordered_tile_outputs:

        if single_spatial_tile:
            result = tile_out[:, :, :target_d, :target_h, :target_w]
            if result.device != x.device or result.dtype != x.dtype:
                result = result.to(device=x.device, dtype=x.dtype)
            if x.shape[2] == 1 and sf_t == 1:
                result = result.squeeze(2)
            bar.update(1)
            return result

        if result is None:
            b_out, c_out = tile_out.shape[0], tile_out.shape[1]
            result = torch.zeros((b_out, c_out, target_d, target_h, target_w), device=storage_device, dtype=torch.float32)
            count = torch.zeros((1, 1, 1, target_h, target_w), device=storage_device, dtype=torch.float32)

        if encode:
            ys, ye = y_idx // sf_s, (y_idx // sf_s) + tile_out.shape[3]
            xs, xe = x_idx // sf_s, (x_idx // sf_s) + tile_out.shape[4]
            cur_ov_h = max(0, min(blend_ov_h, tile_out.shape[3] // 2))
            cur_ov_w = max(0, min(blend_ov_w, tile_out.shape[4] // 2))
        else:
            ys, ye = y_idx * sf_s, (y_idx * sf_s) + tile_out.shape[3]
            xs, xe = x_idx * sf_s, (x_idx * sf_s) + tile_out.shape[4]
            cur_ov_h = max(0, min(blend_ov_h, tile_out.shape[3] // 2))
            cur_ov_w = max(0, min(blend_ov_w, tile_out.shape[4] // 2))

        w_h = torch.ones((tile_out.shape[3],), device=storage_device)
        w_w = torch.ones((tile_out.shape[4],), device=storage_device)

        if cur_ov_h > 0:
            r = get_ramp(cur_ov_h)
            if y_idx > 0:
                w_h[:cur_ov_h] = r
            if y_end < h:
                w_h[-cur_ov_h:] = 1.0 - r

        if cur_ov_w > 0:
            r = get_ramp(cur_ov_w)
            if x_idx > 0:
                w_w[:cur_ov_w] = r
            if x_end < w:
                w_w[-cur_ov_w:] = 1.0 - r

        final_weight = w_h.view(1,1,1,-1,1) * w_w.view(1,1,1,1,-1)

        valid_d = min(tile_out.shape[2], result.shape[2])
        tile_out = tile_out[:, :, :valid_d, :, :]

        tile_out.mul_(final_weight)

        result[:, :, :valid_d, ys:ye, xs:xe] += tile_out
        count[:, :, :, ys:ye, xs:xe] += final_weight

        del tile_out, final_weight, w_h, w_w
        bar.update(1)

    result.div_(count.clamp(min=1e-6))

    if result.device != x.device or result.dtype != x.dtype:
        result = result.to(device=x.device, dtype=x.dtype)

    if x.shape[2] == 1 and sf_t == 1:
        result = result.squeeze(2)

    return result

_NORM_LIMIT = float("inf")
def get_norm_limit():
    return _NORM_LIMIT


def set_norm_limit(value: Optional[float] = None):
    global _NORM_LIMIT
    if value is None:
        value = float("inf")
    _NORM_LIMIT = value

@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding

class MemoryState(Enum):
    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3

def get_cache_size(conv_module, input_len, pad_len, dim=0):
    dilated_kernel_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernel_size) // conv_module.stride[dim] + 1
    remain_len = (
        input_len + pad_len - ((output_len - 1) * conv_module.stride[dim] + dilated_kernel_size)
    )
    overlap_len = dilated_kernel_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len

    if output_len <= 0:
        raise ValueError(
            f"SeedVR2 VAE cache input is too short for convolution: input_len={input_len}, pad_len={pad_len}."
        )
    return cache_len

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, BYTEDANCE_LOGVAR_CLAMP_MIN, BYTEDANCE_LOGVAR_CLAMP_MAX)

    def mode(self):
        return self.mean

class SpatialNorm(nn.Module):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = ops.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = ops.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = ops.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.out_dim = query_dim
        self.heads = heads

        if norm_num_groups is not None:
            self.group_norm = ops.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        self.to_q = ops.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = ops.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = ops.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([])
        self.to_out.append(ops.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Identity())

        self.optimized_vae_attention = vae_attention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if input_ndim == 4 and self.heads == 1:
            query = query.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            key = key.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            value = value.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            hidden_states = self.optimized_vae_attention(query, key, value).reshape(batch_size, self.heads, head_dim, height * width).transpose(2, 3)
        else:
            hidden_states = optimized_attention(query, key, value, heads = self.heads, skip_reshape=True, skip_output_reshape=True)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    if isinstance(norm_layer, (nn.LayerNorm, nn.RMSNorm)):
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)
            x = norm_layer(x)
            x = x.permute(0, 3, 1, 2)
            return x.to(input_dtype)
        if x.ndim == 5:
            x = x.permute(0, 2, 3, 4, 1)
            x = norm_layer(x)
            x = x.permute(0, 4, 1, 2, 3)
            return x.to(input_dtype)
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x).to(input_dtype)
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = x.transpose(1, 2).reshape(b * t, c, h, w)
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy > get_norm_limit():
                num_chunks = min(BYTEDANCE_GN_CHUNKS_FP16 if x.element_size() == 2 else BYTEDANCE_GN_CHUNKS_FP32, norm_layer.num_groups)
                if norm_layer.num_groups % num_chunks != 0:
                    raise ValueError(
                        f"SeedVR2 VAE GroupNorm groups must divide chunks: groups={norm_layer.num_groups}, chunks={num_chunks}."
                    )
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                weights = comfy.ops.cast_to_input(norm_layer.weight, x).chunk(num_chunks, dim=0)
                biases = comfy.ops.cast_to_input(norm_layer.bias, x).chunk(num_chunks, dim=0)
                x = list(x.chunk(num_chunks, dim=1))
                for i, (w, bias) in enumerate(zip(weights, biases)):
                    x[i] = F.group_norm(x[i], num_groups_per_chunk, w, bias, norm_layer.eps)
                    x[i] = x[i].to(input_dtype)
                x = torch.cat(x, dim=1)
            else:
                x = norm_layer(x)
            x = x.reshape((b, t, x.size(1), x.size(2), x.size(3))).transpose(1, 2)
            return x.to(input_dtype)
    raise TypeError(f"SeedVR2 VAE unsupported norm layer type: {type(norm_layer).__name__}")

_receptive_field_t = Literal["half", "full"]

def extend_head(tensor, times: int = 2, memory = None):
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    if times < 0:
        raise ValueError(f"SeedVR2 VAE extend_head expected times >= 0, got {times}.")
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)

def cache_send_recv(tensor, cache_size, times, memory=None):
    recv_buffer = None

    if memory is not None:
        recv_buffer = memory.to(tensor[0])
    elif times > 0:
        tile_repeat = [1] * tensor[0].ndim
        tile_repeat[2] = times
        recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)

    return recv_buffer

class InflatedCausalConv3d(ops.Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode,
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.padding = (0, *self.padding[1:])
        self.memory_limit = float("inf")
        self.logged_once = False

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def _conv_forward(self, input, weight, bias, *args, **kwargs):
        try:
            return super()._conv_forward(input, weight, bias, *args, **kwargs)
        except NotImplementedError:
            # for: Could not run 'aten::cudnn_convolution' with arguments from the 'CPU' backend
            if not self.logged_once:
                logging.warning("VAE is on CPU for decoding. This is most likely due to not enough memory")
                self.logged_once = True
            return F.conv3d(input, weight, bias, *args, **kwargs)

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return super().forward(x)

        shape = list(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        for i, pad_sum in enumerate((padding[4] + padding[5], padding[2] + padding[3], padding[0] + padding[1])):
            shape[-3 + i] += pad_sum
        memory_occupy = math.prod(shape) * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            x_concat = x
            if prev_cache is not None:
                x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)

            def pad_and_forward():
                padded = F.pad(x_concat, padding, mode='constant', value=0.0)
                if not padded.is_contiguous():
                    padded = padded.contiguous()
                with ignore_padding(self):
                    return torch.nn.Conv3d.forward(self, padded)

            return pad_and_forward()

        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        cache = None
        for idx in range(len(x)):
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_cache_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_cache_size != 0:
                if next_cache_size > x[idx].size(split_dim):
                    raise ValueError(
                        f"SeedVR2 VAE cache size {next_cache_size} exceeds split size {x[idx].size(split_dim)}."
                    )
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_cache_size:].transpose(0, split_dim)
                )

            x[idx] = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache
            )

            cache = next_cache

        output = torch.cat(x, dim=split_dim)
        return output

    def forward(
        self,
        input,
        memory_state: MemoryState = MemoryState.UNSET,
        memory_cache = None,
    ) -> Tensor:
        if memory_state == MemoryState.UNSET:
            raise ValueError("SeedVR2 VAE convolution requires an explicit MemoryState.")
        if memory_cache is None:
            memory_cache = {}
        if memory_state != MemoryState.ACTIVE:
            memory_cache.pop(self, None)
        if (
            math.isinf(self.memory_limit)
            and torch.is_tensor(input)
        ):
            return self.basic_forward(input, memory_state, memory_cache)
        return self.slicing_forward(input, memory_state, memory_cache)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET, memory_cache = None):
        mem_size = self.stride[0] - self.kernel_size[0]
        memory = memory_cache.get(self) if memory_cache is not None else None
        if (memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        next_memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if memory_cache is not None and memory_state != MemoryState.DISABLED:
            if next_memory is None:
                memory_cache.pop(self, None)
            else:
                memory_cache[self] = next_memory
        return super().forward(input)

    def slicing_forward(
        self,
        input,
        memory_state: MemoryState = MemoryState.UNSET,
        memory_cache = None,
    ) -> Tensor:
        if memory_cache is None:
            memory_cache = {}
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        memory = memory_cache.get(self) if memory_cache is not None else None
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=memory, times=self.temporal_padding * 2
        )

        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            if cache_size <= input[-1].size(2):
                memory_cache[self] = input[-1][:, :, -cache_size:].detach().contiguous()

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                if cache_size > input[i].size(2):
                    raise ValueError(f"SeedVR2 VAE cache size {cache_size} exceeds input length {input[i].size(2)}.")
                next_cache = input[i][:, :, -cache_size:]

            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache
            )

            cache = next_cache

        return input[0] if squeeze_out else input

def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)

class Upsample3D(nn.Module):

    def __init__(
        self,
        channels,
        out_channels = None,
        inflation_mode = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        conv = InflatedCausalConv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1

        upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
        self.upscale_conv = ops.Conv3d(
            self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
        )

        self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state=None,
        memory_cache=None,
    ) -> torch.FloatTensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"SeedVR2 upsample expected {self.channels} channels, got {hidden_states.shape[1]}.")

        hidden_states = self.upscale_conv(hidden_states)
        b, channels, f, h, w = hidden_states.shape
        c = channels // (self.spatial_ratio * self.spatial_ratio * self.temporal_ratio)
        hidden_states = hidden_states.view(b, self.spatial_ratio, self.spatial_ratio, self.temporal_ratio, c, f, h, w)
        hidden_states = hidden_states.permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(
            b,
            c,
            f * self.temporal_ratio,
            h * self.spatial_ratio,
            w * self.spatial_ratio,
        )

        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states = remove_head(hidden_states)

        hidden_states = self.conv(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(
        self,
        channels,
        out_channels = None,
        inflation_mode = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        self.conv = InflatedCausalConv3d(
            self.channels,
            self.out_channels,
            kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
            stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            padding=(1 if self.temporal_down else 0, 0, 0),
            inflation_mode=inflation_mode,
        )


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state = None,
        memory_cache = None,
    ) -> torch.FloatTensor:

        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"SeedVR2 downsample expected {self.channels} channels, got {hidden_states.shape[1]}.")

        if self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"SeedVR2 downsample expected {self.channels} channels after padding, got {hidden_states.shape[1]}.")

        hidden_states = self.conv(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        skip_time_act: bool = False,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.output_scale_factor = output_scale_factor
        self.skip_time_act = skip_time_act
        self.nonlinearity = nn.SiLU()
        if temb_channels is not None:
            self.time_emb_proj = ops.Linear(temb_channels, self.out_channels)
        else:
            self.time_emb_proj = None
        self.norm1 = ops.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        if groups_out is None:
            groups_out = groups
        self.norm2 = ops.GroupNorm(num_groups=groups_out, num_channels=self.out_channels, eps=eps, affine=True)
        self.use_in_shortcut = self.in_channels != self.out_channels
        self.conv1 = InflatedCausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = InflatedCausalConv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedCausalConv3d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                inflation_mode=inflation_mode,
            )

    def forward(self, input_tensor, temb, memory_state = None, memory_cache = None):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None:
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state, memory_cache=memory_cache)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    output_scale_factor=output_scale_factor,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        out_channels=out_channels,
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state = None,
        memory_cache = None,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state, memory_cache=memory_cache)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    output_scale_factor=output_scale_factor,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        memory_state=None,
        memory_cache=None,
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state, memory_cache=memory_cache)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                output_scale_factor=output_scale_factor,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups if resnet_time_scale_shift == "default" else None
                        ),
                        spatial_norm_dim=(
                            temb_channels if resnet_time_scale_shift == "spatial" else None
                        ),
                        residual_connection=True,
                        bias=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    output_scale_factor=output_scale_factor,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, memory_state=None, memory_cache=None):
        video_length = hidden_states.size(2)
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state, memory_cache=memory_cache)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                b, c, f, h, w = hidden_states.shape
                hidden_states = hidden_states.transpose(1, 2).reshape(b * f, c, h, w)
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = hidden_states.reshape(b, video_length, c, h, w).transpose(1, 2)
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state, memory_cache=memory_cache)

        return hidden_states


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention=True,
        temporal_down_num: int = 2,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1

            if down_block_type != "DownEncoderBlock3D":
                raise ValueError(f"SeedVR2 encoder only supports DownEncoderBlock3D, got {down_block_type}.")

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.conv_norm_out = ops.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )


    def forward(
        self,
        sample: torch.FloatTensor,
        memory_state = None,
        memory_cache = None,
    ) -> torch.FloatTensor:
        sample = sample.to(next(self.parameters()).device)
        sample = self.conv_in(sample, memory_state=memory_state, memory_cache=memory_cache)
        for down_block in self.down_blocks:
            sample = down_block(sample, memory_state=memory_state, memory_cache=memory_cache)

        sample = self.mid_block(sample, memory_state=memory_state, memory_cache=memory_cache)

        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state, memory_cache=memory_cache)

        return sample


class Decoder3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention=True,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = None

        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            if up_block_type != "UpDecoderBlock3D":
                raise ValueError(f"SeedVR2 decoder only supports UpDecoderBlock3D, got {up_block_type}.")
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = ops.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )


    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
        memory_state = None,
        memory_cache = None,
    ) -> torch.FloatTensor:

        sample = sample.to(next(self.parameters()).device)
        sample = self.conv_in(sample, memory_state=memory_state, memory_cache=memory_cache)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        sample = self.mid_block(sample, latent_embeds, memory_state=memory_state, memory_cache=memory_cache)
        sample = sample.to(upscale_dtype)

        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds, memory_state=memory_state, memory_cache=memory_cache)

        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state, memory_cache=memory_cache)

        return sample

class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        layers_per_block: int = 2,
        latent_channels: int = SEEDVR2_LATENT_CHANNELS,
        norm_num_groups: int = 32,
        temporal_scale_num: int = 2,
        inflation_mode = "pad",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size = BYTEDANCE_SLICING_SAMPLE_MIN,
    ):
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (2**temporal_scale_num)
        block_out_channels = BYTEDANCE_BLOCK_OUT_CHANNELS
        down_block_types = ("DownEncoderBlock3D",) * 4
        up_block_types = ("UpDecoderBlock3D",) * 4
        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            temporal_down_num=temporal_scale_num,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            temporal_up_num=temporal_scale_num,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.use_slicing = True

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        h = self.slicing_encode(x)
        posterior = DiagonalGaussianDistribution(h).mode()

        if not return_dict:
            return (posterior,)

        return posterior

    def decode_(
        self, z: torch.Tensor, return_dict: bool = True
    ):
        decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return decoded

    def _encode(
        self, x, memory_state = MemoryState.DISABLED, memory_cache = None
    ) -> torch.Tensor:
        _x = x.to(self.device)
        h = self.encoder(_x, memory_state=memory_state, memory_cache=memory_cache)
        return h.to(x.device)

    def _decode(
        self, z, memory_state = MemoryState.DISABLED, memory_cache = None
    ) -> torch.Tensor:
        _z = z.to(self.device)
        output = self.decoder(_z, memory_state=memory_state, memory_cache=memory_cache)
        return output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size:
            memory_cache = {}
            split_size = max(
                self.slicing_sample_min_size,
                getattr(self, "temporal_downsample_factor", 1),
            )
            x_slices = list(x[:, :, 1:].split(split_size=split_size, dim=2))
            min_active_len = getattr(self, "temporal_downsample_factor", 1)
            if len(x_slices) > 1 and x_slices[-1].shape[2] < min_active_len:
                x_slices[-2] = torch.cat((x_slices[-2], x_slices[-1]), dim=2)
                x_slices.pop()
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                    memory_cache=memory_cache,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE, memory_cache=memory_cache)
                )
            out = torch.cat(encoded_slices, dim=2)
            return out
        else:
            return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size:
            memory_cache = {}
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                    memory_cache=memory_cache,
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE, memory_cache=memory_cache)
                )
            out = torch.cat(decoded_slices, dim=2)
            return out
        else:
            return self._decode(z)

    def forward(self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all"):
        def _unwrap(value):
            return value[0] if isinstance(value, tuple) else value

        if mode == "encode":
            return _unwrap(self.encode(x))
        if mode == "decode":
            return _unwrap(self.decode_(x))
        if mode == "all":
            latent = _unwrap(self.encode(x))
            return _unwrap(self.decode_(latent))
        raise ValueError(f"Unknown SeedVR2 VAE forward mode: {mode}")

class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self,
        spatial_downsample_factor = 8,
        temporal_downsample_factor = 4,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        super().__init__()
        self.set_memory_limit(BYTEDANCE_VAE_CONV_MEM_GIB, BYTEDANCE_VAE_NORM_MEM_GIB)

    def forward(self, x: torch.FloatTensor):
        z, p = self._encode_with_raw_latent(x)
        x = self.decode(z)
        return x, z, p

    def _encode_with_raw_latent(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        self.device = x.device
        p = super().encode(x)
        z = p.squeeze(2)
        return z, p

    def encode(self, x):
        z, _ = self._encode_with_raw_latent(x)
        return z

    def decode(self, z, seedvr2_tiling=None):
        seedvr2_tiling = {} if seedvr2_tiling is None else seedvr2_tiling
        if not isinstance(seedvr2_tiling, dict):
            raise RuntimeError(
                "SeedVR2 VideoAutoencoderKLWrapper.decode: `seedvr2_tiling` must be a dict; "
                f"got {type(seedvr2_tiling).__name__} with value {seedvr2_tiling!r}."
            )

        if z.ndim == 5:
            _, c, _, _, _ = z.shape
            if c != SEEDVR2_LATENT_CHANNELS:
                raise RuntimeError(
                    "SeedVR2 VideoAutoencoderKLWrapper.decode: 5-D latent input must "
                    f"have {SEEDVR2_LATENT_CHANNELS} channels; got shape {tuple(z.shape)}."
                )
            latent = z
        elif z.ndim == 4:
            b, tc, h, w = z.shape
            if tc % SEEDVR2_LATENT_CHANNELS != 0:
                raise RuntimeError(
                    "SeedVR2 VideoAutoencoderKLWrapper.decode: 4-D latent input must "
                    f"use collapsed channel layout (B, {SEEDVR2_LATENT_CHANNELS}*T, H, W); "
                    f"got shape {tuple(z.shape)}."
                )
            latent = z.reshape(b, SEEDVR2_LATENT_CHANNELS, -1, h, w)
        else:
            raise RuntimeError(
                "SeedVR2 VideoAutoencoderKLWrapper.decode: latent input must be "
                f"4-D collapsed (B, {SEEDVR2_LATENT_CHANNELS}*T, H, W) or "
                f"5-D (B, {SEEDVR2_LATENT_CHANNELS}, T, H, W); "
                f"got shape {tuple(z.shape)}."
            )
        scale = BYTEDANCE_VAE_SCALING_FACTOR
        shift = BYTEDANCE_VAE_SHIFTING_FACTOR
        latent = latent / scale + shift

        self.device = latent.device
        enable_tiling = seedvr2_tiling.get("enable_tiling", False)

        if enable_tiling:
            decode_seedvr2_args = dict(seedvr2_tiling)
            decode_seedvr2_args.pop("enable_tiling", None)
            tile_h, tile_w = decode_seedvr2_args.get("tile_size", (512, 512))
            ov_h, ov_w = decode_seedvr2_args.get("tile_overlap", (64, 64))
            decode_seedvr2_args["tile_overlap"] = (
                min(ov_h, max(0, tile_h - 8)),
                min(ov_w, max(0, tile_w - 8)),
            )
            x = tiled_vae(latent, self, **decode_seedvr2_args, encode=False)
            if x.ndim == 4:
                # tiled_vae squeezes the temporal axis when
                # temporal_downsample_factor == 1 AND latent T == 1
                # (see tiled_vae line 179-180); re-add it so the post-decode
                # pipeline can keep batch and time distinct on the tiled path.
                x = x.unsqueeze(2)
        else:
            x = super().decode_(latent)

        h, w = x.shape[-2:]
        w2 = w - (w % 2)
        h2 = h - (h % 2)
        x = x[..., :h2, :w2]

        return x

    def decode_tiled(self, z, tile_x=32, tile_y=32, overlap=8, tile_t=None, overlap_t=None):
        # SeedVR2's causal VAE owns temporal via the MemoryState cache; external
        # temporal tiling breaks that continuity, so only spatial tiling is applied.
        sf = self.spatial_downsample_factor
        seedvr2_tiling = {
            "enable_tiling": True,
            "tile_size": (tile_y * sf, tile_x * sf),
            "tile_overlap": (overlap * sf, overlap * sf),
            "temporal_size": None,
            "temporal_overlap": None,
        }
        return self.decode(z, seedvr2_tiling=seedvr2_tiling)

    def encode_tiled(self, x, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        # External temporal tiling knobs are discarded; the causal VAE keeps its
        # own internal MemoryState slicing.
        if tile_y is None:
            tile_y = 512
        if tile_x is None:
            tile_x = 512
        if overlap is None:
            overlap_y = 64
            overlap_x = 64
        else:
            overlap_y = overlap
            overlap_x = overlap
        overlap_y = min(overlap_y, max(0, tile_y - 8))
        overlap_x = min(overlap_x, max(0, tile_x - 8))
        self.device = x.device
        return tiled_vae(
            x,
            self,
            tile_size=(tile_y, tile_x),
            tile_overlap=(overlap_y, overlap_x),
            temporal_size=None,
            temporal_overlap=None,
            encode=True,
        )

    def comfy_format_encoded(self, samples):
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        samples = samples.contiguous()
        samples = samples * BYTEDANCE_VAE_SCALING_FACTOR
        return samples

    def comfy_memory_used_decode(self, shape):
        bytes_per_output_pixel = 160

        def output_pixels(latent_t, latent_h, latent_w):
            output_t = max(1, (latent_t - 1) * 4 + 1)
            return output_t * latent_h * 8 * latent_w * 8

        # SeedVR2 decode performs full-frame LAB histogram matching: fp32 channels
        # plus int64 sort indices dominate peak memory, not the VAE weight dtype.
        if len(shape) == 5:
            candidates = []
            if shape[1] == SEEDVR2_LATENT_CHANNELS:
                candidates.append((shape[2], shape[3], shape[4]))
            if shape[-1] == SEEDVR2_LATENT_CHANNELS:
                candidates.append((shape[1], shape[2], shape[3]))
            if len(candidates) == 0:
                candidates.append((shape[2], shape[3], shape[4]))
            pixels = max(output_pixels(*candidate) for candidate in candidates)
        elif len(shape) == 4:
            latent_t = max(1, (shape[1] + SEEDVR2_LATENT_CHANNELS - 1) // SEEDVR2_LATENT_CHANNELS)
            pixels = output_pixels(latent_t, shape[2], shape[3])
        else:
            pixels = output_pixels(1, shape[-2], shape[-1])
        return pixels * bytes_per_output_pixel

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
