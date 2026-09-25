from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from comfy.utils import ProgressBar

from comfy.ldm.seedvr.constants import (
    BYTEDANCE_BLOCK_OUT_CHANNELS,
    BYTEDANCE_GN_CHUNKS_FP16,
    BYTEDANCE_GN_CHUNKS_FP32,
    BYTEDANCE_SLICING_SAMPLE_MIN,
    BYTEDANCE_VAE_CONV_MEM_GIB,
    BYTEDANCE_VAE_NORM_MEM_GIB,
    BYTEDANCE_VAE_SCALING_FACTOR,
    BYTEDANCE_VAE_SHIFTING_FACTOR,
    BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE,
    BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE,
    SEEDVR2_LATENT_CHANNELS,
    SEEDVR2_VAE_CACHE_QUANT_BYTES,
    SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL,
    SEEDVR2_DECODE_BYTES_PER_OUTPUT_PIXEL,
    SEEDVR2_DECODE_FIXED_BYTES,
    SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL,
    SEEDVR2_DECODE_LAB_BYTES_PER_OUTPUT_PIXEL,
    SEEDVR2_ENCODE_BYTES_PER_PIXEL,
    SEEDVR2_ENCODE_FIXED_BYTES,
    SEEDVR2_EAGER_DECODE_FACTOR,
    SEEDVR2_EAGER_ENCODE_FACTOR,
    SEEDVR2_MAX_TILE_LATENT,
    SEEDVR2_MIN_TILE_LATENT,
    SEEDVR2_TILE_MEM_HEADROOM,
)
from comfy.ldm.modules.diffusionmodules.model import vae_attention

import collections
import dataclasses
import math
from enum import Enum

import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy_kitchen

_CL = torch.channels_last_3d


def _ck_eligible(x):
    """fp16 on CUDA: the channels_last_3d path; the kitchen conv itself needs the fp16-accumulate opt-in."""
    return x.is_cuda and x.dtype == torch.float16 and x.dim() == 5


def _fp16_accumulate_wanted(x):
    """The kitchen conv accumulates in fp16 (1.4-1.9x over cuDNN on this decoder, 73.9 -> 58 dB
    against fp32); it is taken only when the user opted into fp16 accumulation, as the MiniMax H3
    VAE and the fp16 GEMMs do."""
    return comfy.ops._fp16_linear_wanted(x)


def _ndhwc_like(t):
    """channels_last_3d, or a frame window of such a tensor (batch of one)."""
    return t.is_contiguous(memory_format=_CL) or (t.dim() == 5 and t.stride(1) == 1 and t.stride(4) == t.size(1))


def _match_layout(t, ref):
    """Return ``t`` in ``ref``'s memory format, so a cat that lists it first keeps the layout."""
    if ref.is_contiguous(memory_format=_CL) and not t.is_contiguous(memory_format=_CL):
        return t.contiguous(memory_format=_CL)
    return t


def _ndhwc_filter(conv):
    """Store a resident filter NDHWC once, so cast_bias_weight hands it back without a per-call copy."""
    if conv.weight.is_cuda and not conv.weight.is_contiguous(memory_format=_CL):
        conv.weight.data = conv.weight.data.contiguous(memory_format=_CL)
ops = comfy.ops.manual_cast


def _seedvr2_clamped_spatial_overlap(overlap, tile_size):
    overlap = max(0, int(overlap))
    tile_size = max(1, int(tile_size))
    return min(overlap, tile_size - 1)


def tiled_vae(x, vae_model, tile_size=(512, 512), tile_overlap=(64, 64), encode=True):
    """Spatial tiles, each through the model's own temporally sliced encode or decode, blended with
    cosine ramps over the overlaps."""
    _, _, d, h, w = x.shape
    sf_s, sf_t = vae_model.spatial_downsample_factor, vae_model.temporal_downsample_factor
    if encode:
        ti_h, ti_w = tile_size
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0], ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1], ti_w)
        blend_ov_h, blend_ov_w = ov_h // sf_s, ov_w // sf_s
        target_d = (d + sf_t - 1) // sf_t
        target_h, target_w = (h + sf_s - 1) // sf_s, (w + sf_s - 1) // sf_s
    else:
        ti_h, ti_w = max(1, tile_size[0] // sf_s), max(1, tile_size[1] // sf_s)
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0] // sf_s, ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1] // sf_s, ti_w)
        blend_ov_h, blend_ov_w = ov_h * sf_s, ov_w * sf_s
        target_d = max(1, d * sf_t - (sf_t - 1))
        target_h, target_w = h * sf_s, w * sf_s
    stride_h, stride_w = max(1, ti_h - ov_h), max(1, ti_w - ov_w)
    storage_device = vae_model.device

    def run(tile):
        tile = tile.contiguous()
        out = vae_model.encode(tile) if encode else vae_model.slicing_decode(tile)
        if out.ndim == 4:   # encode drops the time axis of a single-frame latent
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

    bar = ProgressBar(len(tile_ranges))
    if h <= ti_h and w <= ti_w:
        result = run(x)[:, :, :target_d, :target_h, :target_w]
        bar.update(1)
        return result.to(device=x.device, dtype=x.dtype)

    result = count = None
    for y_idx, y_end, x_idx, x_end in tile_ranges:
        tile_out = run(x[:, :, :, y_idx:y_end, x_idx:x_end])
        if result is None:
            result = torch.zeros((tile_out.shape[0], tile_out.shape[1], target_d, target_h, target_w), device=storage_device, dtype=torch.float32)
            count = torch.zeros((1, 1, 1, target_h, target_w), device=storage_device, dtype=torch.float32)

        ys = y_idx // sf_s if encode else y_idx * sf_s
        xs = x_idx // sf_s if encode else x_idx * sf_s
        ye, xe = ys + tile_out.shape[3], xs + tile_out.shape[4]
        cur_ov_h = min(blend_ov_h, tile_out.shape[3] // 2)
        cur_ov_w = min(blend_ov_w, tile_out.shape[4] // 2)

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
        final_weight = w_h.view(1, 1, 1, -1, 1) * w_w.view(1, 1, 1, 1, -1)

        valid_d = min(tile_out.shape[2], result.shape[2])
        tile_out = tile_out[:, :, :valid_d]
        tile_out.mul_(final_weight)
        result[:, :, :valid_d, ys:ye, xs:xe] += tile_out
        count[:, :, :, ys:ye, xs:xe] += final_weight
        del tile_out, final_weight, w_h, w_w
        bar.update(1)

    result.div_(count.clamp(min=1e-6))
    return result.to(device=x.device, dtype=x.dtype)

_NORM_LIMIT = float("inf")
def get_norm_limit():
    return _NORM_LIMIT


def set_norm_limit(value: Optional[float] = None):
    global _NORM_LIMIT
    if value is None:
        value = float("inf")
    _NORM_LIMIT = value


class CausalMemoryCache:
    """Per-convolution temporal tails, kept between the slices of one encode or decode.

    Large channels_last CUDA tails are int8-packed (ConvRot, per-token scales); with ``offload`` the
    packed tails live in host memory and the next reader's tail is prefetched while this one runs.
    A tail can be a ring of per-frame entries, so extending it packs one frame. free() ends the pass.
    """

    CONVROT_GROUPSIZE = 256

    def __init__(self, offload=False):
        self.offload = offload
        self.plain = {}
        self.packed = {}       # key -> (qdata, params, shape), on the host when offloaded
        self._order = []       # keys in packing order, which is the order the next slice reads them
        self._staged = {}      # key -> (qdata, params, event) on its way back to the GPU
        self._spare = {}       # (shape, dtype) -> host buffers of dropped entries
        self._pinned = []
        self._landing = None   # (event, sources) of the copy to the host still in flight
        self._stream = None
        self._device = None
        self._rings = {}       # key -> deque of per-frame subkeys, oldest first
        self._ring_serial = 0

    def _transfer_stream(self, device):
        """The one stream this cache copies on, so copies reusing a host buffer stay ordered;
        it starts after the compute queued so far. None copies on the compute stream."""
        if self._stream is None:
            self._stream = comfy.model_management.get_offload_stream(device)
        if self._stream is not None:
            self._stream.wait_stream(comfy.model_management.current_stream(device))
        return self._stream

    def _host_like(self, t):
        spare = self._spare.get((t.shape, t.dtype))
        if spare:
            return spare.pop()
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu")
        if comfy.model_management.pin_memory(host, evict_active=False):
            self._pinned.append(host)
        return host

    def _to_host(self, *tensors):
        """Start copying ``tensors`` to host buffers; they stay alive until the copy lands.
        One copy in flight at a time: sources held for copies the CPU ran ahead of are what grows
        the peak (2x at 720p on either allocator); record_stream instead grows cudaMallocAsync's pool."""
        if self._landing is not None:
            self._landing[0].synchronize()
        self._device = tensors[0].device
        stream = self._transfer_stream(self._device)
        host = [comfy.model_management.cast_to(t, None, torch.device("cpu"), non_blocking=True, stream=stream, r=self._host_like(t))
                for t in tensors]
        event = torch.cuda.Event()
        event.record(stream)
        self._landing = (event, tensors)
        return host

    def park(self, owner, frames):
        """Hold channels_last ``frames`` unpacked in host memory until popped, the first already on
        its way back; returns their keys."""
        keys = [(owner, "parked", i) for i in range(len(frames))]
        host = self._to_host(*(f.permute(0, 2, 3, 4, 1) for f in frames))
        for key, h, f in zip(keys, host, frames):
            self.packed[key] = (h, None, f.shape)
            if key not in self._order:
                self._order.append(key)
        if keys:
            self._staged[keys[0]] = self._bring_back(keys[0], self._device)
        return keys

    def _bring_back(self, key, device):
        """Start the copy of an offloaded entry back to the GPU; returns (data, params, event)."""
        hq, params, _ = self.packed[key]
        # allocated on the compute stream, which the copy then waits on: queued work may still be releasing the block
        qdata = torch.empty_like(hq, device=device)
        stream = self._transfer_stream(device)
        comfy.model_management.cast_to(hq, None, device, non_blocking=True, stream=stream, r=qdata)
        if params is not None:
            scale = torch.empty_like(params.scale, device=device)
            comfy.model_management.cast_to(params.scale, None, device, non_blocking=True, stream=stream, r=scale)
            params = dataclasses.replace(params, scale=scale)
        event = torch.cuda.Event()
        event.record(stream)
        return qdata, params, event

    def _top(self, key):
        return key[0] if isinstance(key, tuple) and len(key) == 2 and key[0] in self._rings else key

    def _prefetch_after(self, top, device):
        """The reader after this one is next in line; have its tail on the way while this one runs."""
        try:
            nxt = self._order[self._order.index(top) + 1]
        except (ValueError, IndexError):
            return
        for sub in self._rings.get(nxt, (nxt,)):
            if sub in self.packed and sub not in self._staged:
                self._staged[sub] = self._bring_back(sub, device)

    def push_frames(self, key, frames, keep):
        """Extend ``key``'s tail by ``frames``, keeping the last ``keep``; each frame is its own entry,
        stored as given (the buffered convs pass frames of their padded buffer, border included)."""
        ring = self._rings.setdefault(key, collections.deque())
        for j in range(frames.size(2)):
            self._ring_serial += 1
            sub = (key, self._ring_serial)
            frame = frames[:, :, j:j + 1]
            if not self._packable(frame):
                frame = frame.clone(memory_format=_CL)   # held or sent to the host: not as a view of the caller's buffer
            self[sub] = frame
            ring.append(sub)
        self.keep_last(key, keep)
        if key not in self._order:
            self._order.append(key)

    def keep_last(self, key, keep):
        """Drop all but the newest ``keep`` frames of ``key``'s ring."""
        ring = self._rings.get(key, ())
        while len(ring) > keep:
            self.discard(ring.popleft())

    def _packable(self, value):
        return (
            torch.is_tensor(value)
            and value.dim() == 5
            and value.is_cuda
            and value.is_contiguous(memory_format=_CL)
            and value.shape[1] % 4 == 0   # ConvRot rotates power-of-4 channel groups
            and value.numel() * value.element_size() >= SEEDVR2_VAE_CACHE_QUANT_BYTES
        )

    def __setitem__(self, key, value):
        self.discard(key)   # a whole tail replaces a ring too
        if self._packable(value):
            c = value.shape[1]
            g = self.CONVROT_GROUPSIZE
            while g > 4 and c % g:
                g //= 4
            # rows of the NDHWC view are tokens: the rotation spreads each token's outliers over its channels
            qdata, params = comfy.quant_ops.TensorWiseINT8Layout.quantize(
                value.permute(0, 2, 3, 4, 1).reshape(-1, c), is_weight=True, per_channel=True, convrot=True, convrot_groupsize=g)
        elif self.offload and value.is_cuda and value.dim() == 5 and value.is_contiguous(memory_format=_CL):
            # too small to be worth packing, but off the GPU all the same: long-lived small blocks
            # fragment the big activations' segments (720p needed 2 GiB more free VRAM)
            qdata, params = value.permute(0, 2, 3, 4, 1), None
        else:
            self.plain[key] = value
            return
        if self.offload:
            if params is None:
                qdata, = self._to_host(qdata)
            else:
                qdata, scale = self._to_host(qdata, params.scale)
                params = dataclasses.replace(params, scale=scale)
            top = self._top(key)
            if top not in self._order:
                self._order.append(top)
        self.packed[key] = (qdata, params, value.shape)

    def get(self, key, default=None):
        """A whole entry; a ring's frames are read with for_each_frame."""
        if key not in self.plain and key not in self.packed:
            return default
        out = self._get_entry(key)
        if self.offload and self._device is not None:
            self._prefetch_after(key, self._device)
        return out

    def for_each_frame(self, key, fn, count=None):
        """``fn(i, frame)`` over the first ``count`` frames of ``key``'s ring one at a time, never holding
        the whole unpacked tail."""
        for i, sub in enumerate(list(self._rings[key])[:count]):
            fn(i, self._get_entry(sub))
        if self.offload and self._device is not None:
            self._prefetch_after(key, self._device)

    def _get_entry(self, key):
        if key in self.plain:
            return self.plain[key]
        qdata, params, (b, c, t, h, w) = self.packed[key]
        if self.offload:
            qdata, params, event = self._staged.pop(key, None) or self._bring_back(key, self._device)
            comfy.model_management.current_stream(self._device).wait_event(event)
        if params is None:   # parked unpacked, NDHWC
            return qdata.permute(0, 4, 1, 2, 3)
        flat = comfy.quant_ops.TensorWiseINT8Layout.dequantize(qdata, params)
        return flat.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)

    def pop(self, key, default=None):
        value = self.get(key, default)
        self.discard(key)
        return value

    def discard(self, key):
        """Drop an entry, or a ring's frames, without bringing it back; its host buffers are reused."""
        for sub in self._rings.pop(key, ()):
            self.discard(sub)
        self.plain.pop(key, None)
        entry = self.packed.pop(key, None)
        staged = self._staged.pop(key, None)
        if staged is not None:   # its GPU copy is freed on the compute stream
            comfy.model_management.current_stream(self._device).wait_event(staged[2])
        if entry is not None and self.offload:
            for t in (entry[0],) if entry[1] is None else (entry[0], entry[1].scale):
                self._spare.setdefault((t.shape, t.dtype), []).append(t)

    def free(self):
        """Unpin the host buffers at the end of the pass; the cache is empty afterwards."""
        if self._pinned:
            torch.cuda.synchronize(self._device)   # no copy may still be using a buffer when it is unpinned
        for t in self._pinned:
            comfy.model_management.unpin_memory(t)
        self._pinned, self._spare, self._staged, self._landing = [], {}, {}, None
        self.plain, self.packed, self._rings, self._order = {}, {}, {}, []

    def __contains__(self, key):
        return key in self.plain or key in self.packed or key in self._rings


def _eager_factor(dtype, factor):
    """The memory figures are fitted on the fp16 channels-last path; other dtypes take the eager path."""
    return 1 if dtype == torch.float16 else factor * dtype.itemsize / 2


def _offload_caches_for(frame_pixels):
    """Whether a pass over this frame area keeps its packed caches in host memory: when there is
    room for them, reclaiming RAM from inactive models if needed."""
    return comfy.model_management.ensure_pin_budget(frame_pixels * SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL, evict_active=False)


class MemoryState(Enum):
    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


def _frame_chunked(owner, run, source, chunk, memory_state, memory_cache, frame_pixels):
    """``run(source, memory_state, memory_cache)`` over a causal, spatial-only stage ``chunk`` frames at
    a time, the cache carrying state between chunks: exact, and only a chunk's activations are live.
    ``source`` is a 1-list the stage pops, so no caller keeps its input alive (the input of a module
    call is held until that module returns)."""
    if not chunk or source[0].size(2) <= chunk:
        return run(source, memory_state, memory_cache)
    local_cache = memory_state == MemoryState.DISABLED or memory_cache is None
    if local_cache:
        # under the same host-memory policy as a sliced pass
        memory_cache = CausalMemoryCache(offload=_offload_caches_for(frame_pixels))
        memory_state = MemoryState.INITIALIZING
    outs = []
    try:
        pieces = list(source.pop().split(chunk, dim=2))
        if memory_cache.offload and pieces[0].is_contiguous(memory_format=_CL):
            # the rest waits in host memory, fetched a chunk ahead: resident, the whole input sat
            # under every chunk's peak (2.5 GiB for the decoder's tail at 1080p)
            pieces[0] = pieces[0].clone(memory_format=_CL)
            pieces[1:] = memory_cache.park(owner, pieces[1:])

        def take(i):
            piece, pieces[i] = pieces[i], None
            return [piece if torch.is_tensor(piece) else memory_cache.pop(piece)]
        for i in range(len(pieces)):
            outs.append(run(take(i), memory_state if i == 0 else MemoryState.ACTIVE, memory_cache))
    finally:
        if local_cache:
            memory_cache.free()
    return torch.cat(outs, dim=2)

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

class Attention(nn.Module):
    """The mid-block's single-head self-attention over each (c, h, w) frame, residual included."""
    def __init__(self, query_dim, norm_num_groups, eps):
        super().__init__()
        self.group_norm = ops.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        self.to_q = ops.Linear(query_dim, query_dim)
        self.to_k = ops.Linear(query_dim, query_dim)
        self.to_v = ops.Linear(query_dim, query_dim)
        self.to_out = nn.ModuleList([ops.Linear(query_dim, query_dim)])
        self.optimized_vae_attention = vae_attention()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b, c, h, w = hidden_states.shape
        x = self.group_norm(hidden_states.view(b, c, h * w)).transpose(1, 2)
        q, k, v = (proj(x).transpose(1, 2).reshape(b, c, h, w) for proj in (self.to_q, self.to_k, self.to_v))
        x = self.optimized_vae_attention(q, k, v).reshape(b, c, h * w).transpose(1, 2)
        return self.to_out[0](x).transpose(1, 2).reshape(b, c, h, w) + hidden_states


def causal_norm_wrapper(norm_layer: nn.GroupNorm, x: torch.Tensor, silu: bool = False) -> torch.Tensor:
    """Per-frame GroupNorm of a (b, c, t, h, w) tensor, with the SiLU folded in where the kernel does both."""
    if _ck_eligible(x) and norm_layer.affine:
        # accepts either layout and emits channels_last_3d, which is where the convolution wants its input
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm_layer, x, offloadable=True)
        try:
            return comfy_kitchen.group_norm_silu_pad3d(
                x, weight, bias, norm_layer.num_groups, norm_layer.eps, (0, 0, 0, 0, 0), silu,
            )
        finally:
            comfy.ops.uncast_bias_weight(norm_layer, weight, bias, offload_stream)
    b, c, t, h, w = x.shape
    x = x.transpose(1, 2).reshape(b * t, c, h, w)
    if x.numel() * x.element_size() / 1024**3 > get_norm_limit():
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
        x = torch.cat(x, dim=1)
    else:
        x = norm_layer(x)
    x = x.reshape((b, t, x.size(1), x.size(2), x.size(3))).transpose(1, 2)
    if silu:
        x = F.silu(x)
    return x


class InflatedCausalConv3d(ops.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.padding = (0, *self.padding[1:])
        self.memory_limit = float("inf")   # GiB of padded input above which slicing_forward splits

    def _conv_padded(self, x, padding):
        """The plain conv with ``padding`` in place of the module's own, through comfy's _conv_forward."""
        orig_padding, self.padding = self.padding, padding
        try:
            return torch.nn.Conv3d.forward(self, x)
        finally:
            self.padding = orig_padding

    def _conv_without_padding_copy(self, x, prev_cache, padding):
        """Convolve with native symmetric padding and slice off the surplus; None with a halo or stride."""
        if prev_cache is not None or any(st != 1 for st in self.stride):
            return None

        pads = ((padding[4], padding[5]), (padding[2], padding[3]), (padding[0], padding[1]))
        native = tuple(max(lo, hi) for lo, hi in pads)
        out = self._conv_padded(x, native)

        for i, (dim, (lo, hi)) in enumerate(zip((2, 3, 4), pads)):
            width = self.dilation[i] * (self.kernel_size[i] - 1)
            length = x.size(dim) + lo + hi - width
            start = native[i] - lo
            if start or out.size(dim) != length:
                out = out.narrow(dim, start, length)
        return out

    @staticmethod
    def _padded_input(x, prev_cache, padding, concat_dim, channels_last=None):
        """The conv's padded input (halo included) in one buffer, written once; ``copy_`` also casts the halo."""
        pads = ((padding[4], padding[5]), (padding[2], padding[3]), (padding[0], padding[1]))
        cache_len = 0 if prev_cache is None else prev_cache.size(concat_dim)
        if cache_len == 0 and not any(lo or hi for lo, hi in pads):
            return x.contiguous(memory_format=_CL) if channels_last else x

        shape = list(x.shape)
        for dim, (lo, hi) in zip((2, 3, 4), pads):
            shape[dim] += lo + hi
        shape[concat_dim] += cache_len

        if channels_last is None:
            channels_last = x.is_contiguous(memory_format=_CL)
        buf = torch.empty(shape, dtype=x.dtype, device=x.device,
                          memory_format=_CL if channels_last else torch.contiguous_format)
        for dim, (lo, hi) in zip((2, 3, 4), pads):
            if lo:
                buf.narrow(dim, 0, lo).zero_()
            if hi:
                buf.narrow(dim, shape[dim] - hi, hi).zero_()

        core = buf
        for dim, (lo, hi) in zip((2, 3, 4), pads):
            core = core.narrow(dim, lo, shape[dim] - lo - hi)
        if cache_len:
            core.narrow(concat_dim, 0, cache_len).copy_(prev_cache)
            core.narrow(concat_dim, cache_len, x.size(concat_dim)).copy_(x)
        else:
            core.copy_(x)
        return buf

    def _ck_applies(self, x):
        """Whether this conv runs on the channels_last_3d path; any input layout, the pad copy lands it there."""
        return (_ck_eligible(x) and self.padding_mode == "zeros" and self.groups == 1
                and all(d == 1 for d in self.dilation))

    def _ck_conv(self, padded):
        """Run an already-padded channels_last_3d input through the conv, kitchen or cuDNN."""
        _ndhwc_filter(self)
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, padded, offloadable=True)
        try:
            weight_cl = weight.contiguous(memory_format=_CL)
            if _fp16_accumulate_wanted(padded):
                return comfy_kitchen.fp16_conv3d(padded, weight_cl, bias, stride=tuple(self.stride))
            # cuDNN keeps channels_last_3d in and out, so the layout still saves its transposes
            return F.conv3d(padded, weight_cl, bias, stride=tuple(self.stride))
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)

    def _symmetric_padding(self):
        return tuple(p for p in reversed(self.padding) for _ in range(2))

    # The ck kernels index a buffer with 32-bit byte offsets; a view handed to one call stays
    # under this many elements (2 GiB of fp16), which is one call at 1080p and two at 1440p.
    _CK_MAX_VIEW_ELEMENTS = 2 ** 30

    # _ck_taps runs each halo tap in this many bands of output rows: on cuDNN, which cannot accumulate
    # into the conv's output, a tap's own output is then that fraction of a frame; the kitchen conv
    # accumulates in place, and the bands keep its views under the 32-bit offsets
    _TAP_BANDS = 4

    def _ck_buffered(self, like, x_shape, memory_state, memory_cache, write_body, residual=None, pad=None):
        """Run this conv from one padded buffer: halo in front, ``write_body`` writes the padded frames
        behind it, and the conv reads and writes frame-window views. ``residual`` goes into the kitchen
        conv's epilogue (added explicitly on the cuDNN path). ``pad`` is the buffer's spatial border
        (top, bottom, left, right), the conv's own padding by default. Replicates forward()'s cache handling.
        """
        b, c, t, h, w = x_shape
        _, ph, pw = self.padding
        pad = pad or (ph, ph, pw, pw)
        kt = self.kernel_size[0]
        cache_size = kt - self.stride[0]
        if memory_cache is None:
            memory_cache = CausalMemoryCache()   # DISABLED: nothing is kept anyway
        elif memory_state != MemoryState.ACTIVE:
            memory_cache.pop(self, None)
        has_memory = self in memory_cache
        halo = cache_size if has_memory else self.temporal_padding * 2
        if t == 1 and halo == 2 and kt == 3 and tuple(self.stride) == (1, 1, 1) and pad == (ph, ph, pw, pw):
            return self._ck_taps(like, x_shape, memory_state, memory_cache, write_body, residual, has_memory)
        hp, wp = h + pad[0] + pad[1], w + pad[2] + pad[3]

        buf = torch.empty((b, c, halo + t, hp, wp), dtype=like.dtype, device=like.device, memory_format=_CL)
        write_body(buf[:, :, halo:])
        del write_body   # and with it the source it captured: nothing but the buffer feeds the conv
        if halo:
            front = buf[:, :, :halo]
            if has_memory:
                # after the body, whose source is freed by then; the frames come with their zero border
                memory_cache.for_each_frame(self, lambda i, frame: front[:, :, i:i + 1].copy_(frame))
            else:
                # the causal head: the first frame repeated, already padded
                front.copy_(buf[:, :, halo:halo + 1].expand(-1, -1, halo, -1, -1))
        if cache_size and memory_state in (MemoryState.INITIALIZING, MemoryState.ACTIVE):
            if has_memory:
                # the halo frames are already the cache's newest entries: push only the new ones
                memory_cache.push_frames(self, buf[:, :, halo:][:, :, -min(t, cache_size):], cache_size)
            else:
                memory_cache.push_frames(self, buf[:, :, -cache_size:], cache_size)

        _ndhwc_filter(self)
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, buf, offloadable=True)
        try:
            weight_cl = weight.contiguous(memory_format=_CL)
            stride = sd, sh, sw = tuple(self.stride)
            out_hw = ((hp - self.kernel_size[1]) // sh + 1) * ((wp - self.kernel_size[2]) // sw + 1)
            fused_residual = None
            # the kitchen's views must stay under its 32-bit offsets; when even one output frame's window
            # does not, cuDNN takes the whole buffer (the kitchen would decline and copy a second output)
            if (not _fp16_accumulate_wanted(buf) or kt * c * hp * wp > self._CK_MAX_VIEW_ELEMENTS
                    or weight.size(0) * out_hw > self._CK_MAX_VIEW_ELEMENTS):
                out = F.conv3d(buf, weight_cl, bias, stride=stride)   # cuDNN, NDHWC in and out
            else:
                zt = (buf.size(2) - kt) // sd + 1
                out = torch.empty((b, weight.size(0), zt, (hp - self.kernel_size[1]) // sh + 1, (wp - self.kernel_size[2]) // sw + 1),
                                  dtype=buf.dtype, device=buf.device, memory_format=_CL)
                chunk = min(zt, (self._CK_MAX_VIEW_ELEMENTS // (c * hp * wp) - kt) // sd + 1,
                            self._CK_MAX_VIEW_ELEMENTS // (weight.size(0) * out_hw))
                if residual is not None and residual.shape == out.shape and residual.dtype == out.dtype and _ndhwc_like(residual):
                    fused_residual = residual   # otherwise added after: the epilogue wants the output's shape and layout
                for z0 in range(0, zt, chunk):
                    z1 = min(zt, z0 + chunk)
                    comfy_kitchen.fp16_conv3d(buf[:, :, z0 * sd:(z1 - 1) * sd + kt], weight_cl, bias,
                                              residual=None if fused_residual is None else fused_residual[:, :, z0:z1],
                                              stride=stride, out=out[:, :, z0:z1])
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
        if residual is not None and fused_residual is None:
            out += residual
        return out

    def _ck_taps(self, like, x_shape, memory_state, memory_cache, write_body, residual, has_memory):
        """A one-frame chunk against its two-frame halo as three single-frame convs summed into the
        output: the 3-frame buffer was the tail's peak (1.5 GiB lower floor at 1080p). The halo taps
        run straight from the unpacked frames, their padding in place of a buffer; with fp16
        accumulation the kitchen conv adds each tap, and the residual, into the output in its
        epilogue (1.7x over cuDNN plus an add). Without a cache the halo is the frame itself, so the
        taps sum first."""
        b, c, _, h, w = x_shape
        _, ph, pw = self.padding
        buf = torch.empty((b, c, 1, h + 2 * ph, w + 2 * pw), dtype=like.dtype, device=like.device, memory_format=_CL)
        write_body(buf)
        del write_body
        _ndhwc_filter(self)
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, buf, offloadable=True)
        try:
            tap = weight[:, :, 2:] if has_memory else weight.sum(dim=2, keepdim=True)
            tap = tap.contiguous(memory_format=_CL)
            fp16_acc = _fp16_accumulate_wanted(buf)
            if fp16_acc:
                fused = (residual is not None and residual.shape == (b, weight.size(0), 1, h, w)
                         and residual.dtype == buf.dtype and _ndhwc_like(residual))
                out = comfy_kitchen.fp16_conv3d(buf, tap, bias, residual=residual if fused else None)
                if fused:
                    residual = None
            else:
                out = F.conv3d(buf, tap, bias)
            if memory_state in (MemoryState.INITIALIZING, MemoryState.ACTIVE):
                # pushed now so the buffer can go; three kept, as the oldest is still this call's first tap
                memory_cache.push_frames(self, buf if has_memory else buf.expand(-1, -1, 2, -1, -1), 3)
            del buf
            if has_memory:
                rows = -(-out.size(3) // self._TAP_BANDS)

                def halo_tap(i, frame):
                    tap_i = weight[:, :, i:i + 1].contiguous(memory_format=_CL)
                    for r0 in range(0, out.size(3), rows):
                        r1 = min(out.size(3), r0 + rows)
                        band, x = out[:, :, :, r0:r1], frame[:, :, :, r0:r1 + 2 * ph]
                        if fp16_acc:
                            comfy_kitchen.fp16_conv3d(x, tap_i, residual=band, out=band)
                        else:
                            band.add_(F.conv3d(x, tap_i))
                memory_cache.for_each_frame(self, halo_tap, count=2)
                memory_cache.keep_last(self, 2)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
        if residual is not None:
            out += residual
        return out

    def _ck_buffered_applies(self, x, memory_state):
        """Whether _ck_buffered can take this call: the frame-offset write needs a batch of one."""
        return self._ck_applies(x) and memory_state != MemoryState.UNSET and x.size(0) == 1

    def fused_norm_conv(self, source, norm, memory_state, memory_cache, residual=None):
        """GroupNorm + SiLU written straight into the conv's buffer; ``source`` (a 1-list) is emptied
        once read so the activation is freed early. None when the fused path does not apply."""
        x = source[0]
        if not (self._ck_buffered_applies(x, memory_state) and isinstance(norm, nn.GroupNorm)
                and norm.affine):
            return None
        _, ph, pw = self.padding
        like, shape = x.new_empty(0), x.shape
        del x
        gw, gb, offload_stream = comfy.ops.cast_bias_weight(norm, like, offloadable=True)

        def write_body(dst):
            comfy_kitchen.group_norm_silu_pad3d(
                source.pop(), gw, gb, norm.num_groups, norm.eps, (pw, pw, ph, ph, 0), True,
                zero_pad=True, out=dst,
            )
        try:
            return self._ck_buffered(like, shape, memory_state, memory_cache, write_body, residual=residual)
        finally:
            comfy.ops.uncast_bias_weight(norm, gw, gb, offload_stream)

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        shape = list(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        for i, pad_sum in enumerate((padding[4] + padding[5], padding[2] + padding[3], padding[0] + padding[1])):
            shape[-3 + i] += pad_sum
        memory_occupy = math.prod(shape) * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            if self._ck_applies(x):
                return self._ck_conv(self._padded_input(x, prev_cache, padding, split_dim - 1, channels_last=True))
            out = self._conv_without_padding_copy(x, prev_cache, padding)
            if out is not None:
                return out
            padded = self._padded_input(x, prev_cache, padding, split_dim - 1)
            return self._conv_padded(padded, (0, 0, 0))

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
        # a 1x1x1 conv has no halo to save by splitting: split and concatenated, its output is held twice
        if (
            (math.isinf(self.memory_limit) or tuple(self.kernel_size) == (1, 1, 1))
            and torch.is_tensor(input)
        ):
            return self.basic_forward(input, memory_state, memory_cache)
        return self.slicing_forward(input, memory_state, memory_cache)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET, memory_cache = None):
        mem_size = self.stride[0] - self.kernel_size[0]
        memory = memory_cache.get(self) if memory_cache is not None else None
        if (memory is not None) and (memory_state == MemoryState.ACTIVE):
            head = memory.to(input)
        elif self.temporal_padding:
            head = torch.tile(input[:, :, :1], [1, 1, self.temporal_padding * 2, 1, 1])
        else:
            head = None
        if self._ck_applies(input):
            # The halo and the zero border share one buffer: the kernel needs a padded copy
            # anyway, so the cat with the head is folded into it. On the largest
            # activations that is one full pass and one activation's worth of peak.
            padded = self._padded_input(input, head, self._symmetric_padding(), 2, channels_last=True)
            if mem_size != 0 and memory_state != MemoryState.DISABLED and memory_cache is not None:
                # From the halo'd buffer, head included: a one-frame slice
                # still leaves a two-frame cache. The spatial border is cropped back off.
                _, ph, pw = self.padding
                tail = padded[:, :, mem_size:, ph:padded.size(3) - ph or None, pw:padded.size(4) - pw or None]
                memory_cache[self] = tail.detach().clone(memory_format=_CL)
            elif memory_cache is not None and memory_state != MemoryState.DISABLED:
                memory_cache.pop(self, None)
            return self._ck_conv(padded)
        input = input if head is None else torch.cat((_match_layout(head, input), input), dim=2)
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

    def slicing_forward(self, input, memory_state, memory_cache):
        """forward() for a finite memory_limit: the halo is handed to memory_limit_conv, which splits
        the conv along H, then W, while its padded input exceeds the limit."""
        cache_size = self.kernel_size[0] - self.stride[0]
        memory = memory_cache.get(self)
        if memory is not None:
            cache = _match_layout(memory.to(input), input)
        elif self.temporal_padding:
            cache = _match_layout(torch.tile(input[:, :, :1], [1, 1, self.temporal_padding * 2, 1, 1]), input)
        else:
            cache = None

        if memory_state in (MemoryState.INITIALIZING, MemoryState.ACTIVE) and cache_size != 0:
            if cache_size > input.size(2) and cache is not None:
                input = torch.cat([cache, input], dim=2)
                cache = None
            if cache_size <= input.size(2):
                # clone, not contiguous(): for a batch of one a frame slice of an NDHWC tensor
                # already satisfies the channels_last_3d check, and contiguous() would return the
                # view, pinning the whole activation for the next slice instead of two frames.
                tail = input[:, :, -cache_size:].detach()
                memory_cache[self] = tail.clone(
                    memory_format=_CL if input.is_contiguous(memory_format=_CL) else torch.contiguous_format)

        return self.memory_limit_conv(input, padding=self._symmetric_padding(), prev_cache=cache)

class Upsample3D(nn.Module):
    """2x spatial (and 2x temporal) upsampling: a 1x1 expansion, its pixel shuffle, a causal 3x3x3 conv."""

    def __init__(self, channels, temporal_up: bool = False):
        super().__init__()
        self.temporal_up = temporal_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2
        self.upscale_conv = ops.Conv3d(channels, channels * self.spatial_ratio ** 2 * self.temporal_ratio, kernel_size=1, padding=0)
        self.conv = InflatedCausalConv3d(channels, channels, 3, padding=1)

    def _expand(self, x, weight, bias, i, j, t):
        """One shuffle phase of the 1x1 expansion; its output channels are (sr, sr, tr, c). A helper
        so the weight slice's copy is freed before the phase is written out."""
        c = x.size(1)
        k = ((i * self.spatial_ratio + j) * self.temporal_ratio + t) * c
        wk = weight[k:k + c].contiguous(memory_format=_CL)
        bk = None if bias is None else bias[k:k + c]
        if _fp16_accumulate_wanted(x):
            return comfy_kitchen.fp16_conv3d(x, wk, bk)
        return F.conv3d(x, wk, bk)

    def _expand_into(self, dst, x, phases, drop_head):
        """Write the 1x1 expansion of ``x`` for temporal ``phases`` into the conv's padded buffer ``dst``,
        one shuffle phase at a time at its strided place: a phase's output is a quarter (an eighth) of
        the expanded tensor. With ``drop_head`` the first input frame keeps only its first phase, as
        the video's first frame is not doubled."""
        sr, tr = self.spatial_ratio, self.temporal_ratio
        _, ph, pw = self.conv.padding
        h_out, w_out = x.size(3) * sr, x.size(4) * sr
        for dim, lo, hi in ((3, ph, ph + h_out), (4, pw, pw + w_out)):
            if lo:
                dst.narrow(dim, 0, lo).zero_()
                dst.narrow(dim, hi, dst.size(dim) - hi).zero_()
        core = dst[:, :, :, ph:ph + h_out, pw:pw + w_out]
        _ndhwc_filter(self.upscale_conv)
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self.upscale_conv, x, offloadable=True)
        try:
            for i in range(sr):
                for j in range(sr):
                    for t in phases:
                        y = self._expand(x, weight, bias, i, j, t)
                        frames = core[:, :, :, i::sr, j::sr]
                        if len(phases) == 1:
                            frames.copy_(y)
                        elif not drop_head:
                            frames[:, :, t::tr].copy_(y)
                        elif t == 0:
                            frames[:, :, 0].copy_(y[:, :, 0])
                            frames[:, :, 1::tr].copy_(y[:, :, 1:])
                        else:
                            frames[:, :, t + 1::tr].copy_(y[:, :, 1:])
                        del y
        finally:
            comfy.ops.uncast_bias_weight(self.upscale_conv, weight, bias, offload_stream)

    def frames_apply(self, x, memory_state):
        return x.size(2) == 1 and x.is_contiguous(memory_format=_CL) and self.conv._ck_buffered_applies(x, memory_state)

    def frames(self, source, memory_state, memory_cache):
        """This upsampler over a one-frame ``source`` (a 1-list), an output frame at a time: each
        temporal phase's expansion goes straight into the conv's buffer and the conv runs on that frame
        alone. Each output is yielded boxed, so nothing here holds it while the caller uses it."""
        x = source.pop()
        b, c, _, h, w = x.shape
        sr = self.spatial_ratio
        phases = [0] if self.temporal_up and memory_state != MemoryState.ACTIVE else range(self.temporal_ratio)
        for n, t in enumerate(phases):
            state = memory_state if n == 0 else MemoryState.ACTIVE
            box = [self.conv._ck_buffered(x.new_empty(0), (b, c, 1, h * sr, w * sr), state, memory_cache,
                                          lambda dst, t=t: self._expand_into(dst, x, [t], False))]
            yield box
            del box

    def run(self, source, memory_state, memory_cache):
        """The upsampler on a 1-list ``source`` it empties: a module call would hold its input until it returns."""
        sr, tr = self.spatial_ratio, self.temporal_ratio
        drop_head = self.temporal_up and memory_state != MemoryState.ACTIVE
        x = source[0]
        if x.is_contiguous(memory_format=_CL) and self.conv._ck_buffered_applies(x, memory_state):
            b, c, f, h, w = x.shape
            like = x.new_empty(0)                 # dtype/device only, shares nothing
            del x
            return self.conv._ck_buffered(like, (b, c, f * tr - int(drop_head), h * sr, w * sr), memory_state, memory_cache,
                                          lambda dst: self._expand_into(dst, source.pop(), range(tr), drop_head))
        del x

        hidden_states = self.upscale_conv(source.pop())
        b, channels, f, h, w = hidden_states.shape
        c = channels // (sr * sr * tr)
        hidden_states = hidden_states.view(b, sr, sr, tr, c, f, h, w).permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(
            b, c, f * tr, h * sr, w * sr)
        if drop_head:
            hidden_states = torch.cat((hidden_states[:, :, :1], hidden_states[:, :, 2:]), dim=2)
        return self.conv(hidden_states, memory_state=memory_state, memory_cache=memory_cache)


class Downsample3D(nn.Module):
    """2x spatial (and 2x temporal) downsampling: a strided conv over a right/bottom zero border."""

    def __init__(self, channels, temporal_down: bool = False):
        super().__init__()
        self.conv = InflatedCausalConv3d(
            channels,
            channels,
            kernel_size=(3 if temporal_down else 1, 3, 3),
            stride=(2 if temporal_down else 1, 2, 2),
            padding=(1 if temporal_down else 0, 0, 0),
        )

    def run(self, source, memory_state, memory_cache):
        """The downsampler on a 1-list ``source`` it empties: a module call would hold its input until it returns."""
        x = source[0]
        if x.is_contiguous(memory_format=_CL) and self.conv._ck_buffered_applies(x, memory_state):
            # straight into the conv's buffer, its right and bottom zero border in place of the pad copy
            b, c, t, h, w = x.shape
            like = x.new_empty(0)
            del x

            def write_body(dst):
                dst[:, :, :, h:].zero_()
                dst[:, :, :, :h, w:].zero_()
                dst[:, :, :, :h, :w].copy_(source.pop())
            return self.conv._ck_buffered(like, (b, c, t, h, w), memory_state, memory_cache, write_body, pad=(0, 1, 0, 1))
        del x

        hidden_states = F.pad(source.pop(), (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states, memory_state=memory_state, memory_cache=memory_cache)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = ops.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm2 = ops.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv1 = InflatedCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = InflatedCausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = InflatedCausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input_tensor, memory_state, memory_cache):
        hidden_states = self.conv1.fused_norm_conv([input_tensor], self.norm1, memory_state, memory_cache)
        if hidden_states is None:
            hidden_states = causal_norm_wrapper(self.norm1, input_tensor, silu=True)
            hidden_states = self.conv1(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        # An identity skip rides in conv2's epilogue, sparing a pass over the output. A 1x1 shortcut
        # runs after conv2 instead: computed first, its output would sit under conv2's buffer and
        # output (the input is held either way).
        residual = input_tensor if self.conv_shortcut is None else None
        source = [hidden_states]
        del hidden_states
        hidden_states = self.conv2.fused_norm_conv(source, self.norm2, memory_state, memory_cache, residual=residual)
        if hidden_states is None:
            hidden_states = causal_norm_wrapper(self.norm2, source.pop(), silu=True)
            hidden_states = self.conv2(hidden_states, memory_state=memory_state, memory_cache=memory_cache)
        elif residual is not None:
            return hidden_states
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state, memory_cache=memory_cache)
        return hidden_states.add_(input_tensor)


def _run_block(resnets, samplers, source, memory_state, memory_cache):
    """A down/up block's layers one by one on a 1-list ``source`` it empties, each sampler handed its
    input boxed: a module call holds its input until it returns, so nothing may hold a layer's input
    past the layer that consumes it."""
    sample = source.pop()
    for resnet in resnets:
        sample = resnet(sample, memory_state, memory_cache)
    for sampler in samplers or ():
        box = [sample]
        del sample
        sample = sampler.run(box, memory_state, memory_cache)
    return sample


class DownEncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, resnet_groups: int,
                 add_downsample: bool, temporal_down: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels, groups=resnet_groups)
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample3D(out_channels, temporal_down=temporal_down)]) if add_downsample else None

    def run(self, source, memory_state, memory_cache):
        return _run_block(self.resnets, self.downsamplers, source, memory_state, memory_cache)


class UpDecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, resnet_groups: int,
                 add_upsample: bool, temporal_up: bool):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels, groups=resnet_groups)
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsample3D(out_channels, temporal_up=temporal_up)]) if add_upsample else None

    def run(self, source, memory_state, memory_cache):
        return _run_block(self.resnets, self.upsamplers, source, memory_state, memory_cache)


class UNetMidBlock3D(nn.Module):
    """A resnet, single-head self-attention within each frame, a resnet."""

    def __init__(self, channels: int, resnet_groups: int):
        super().__init__()
        self.attentions = nn.ModuleList([Attention(channels, norm_num_groups=resnet_groups, eps=1e-6)])
        self.resnets = nn.ModuleList([ResnetBlock3D(channels, channels, groups=resnet_groups) for _ in range(2)])

    def forward(self, hidden_states, memory_state, memory_cache):
        hidden_states = self.resnets[0](hidden_states, memory_state, memory_cache)
        b, c, f, h, w = hidden_states.shape
        hidden_states = self.attentions[0](hidden_states.transpose(1, 2).reshape(b * f, c, h, w))
        hidden_states = hidden_states.reshape(b, f, c, h, w).transpose(1, 2)
        return self.resnets[1](hidden_states, memory_state, memory_cache)


class Encoder3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_out_channels, layers_per_block: int,
                 norm_num_groups: int, temporal_down_num: int):
        super().__init__()
        self.head_blocks = max(0, len(block_out_channels) - temporal_down_num - 1)   # the spatial-only down blocks
        self.conv_in = InflatedCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            self.down_blocks.append(DownEncoderBlock3D(
                input_channel,
                output_channel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                add_downsample=i < len(block_out_channels) - 1,
                temporal_down=i >= len(block_out_channels) - temporal_down_num - 1,
            ))
        self.mid_block = UNetMidBlock3D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.conv_norm_out = ops.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_out = InflatedCausalConv3d(block_out_channels[-1], 2 * out_channels, 3, padding=1)

    # Frames per pass through the full-resolution, spatial-only head; None runs a slice's frames at once.
    head_frames = 1

    def _head(self, source, memory_state, memory_cache):
        box = [self.conv_in(source.pop(), memory_state=memory_state, memory_cache=memory_cache)]
        for down_block in self.down_blocks[:self.head_blocks]:
            box = [down_block.run(box, memory_state, memory_cache)]
        return box.pop()

    def forward(self, sample, memory_state, memory_cache):
        frame_pixels = sample.size(-2) * sample.size(-1)
        source = [sample]
        del sample
        box = [_frame_chunked(self, self._head, source, self.head_frames, memory_state, memory_cache, frame_pixels)]
        for down_block in self.down_blocks[self.head_blocks:]:
            box = [down_block.run(box, memory_state, memory_cache)]
        sample = self.mid_block(box.pop(), memory_state, memory_cache)
        sample = causal_norm_wrapper(self.conv_norm_out, sample, silu=True)
        return self.conv_out(sample, memory_state=memory_state, memory_cache=memory_cache)


class Decoder3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_out_channels, layers_per_block: int,
                 norm_num_groups: int, temporal_up_num: int):
        super().__init__()
        self.temporal_up_num = temporal_up_num
        self.conv_in = InflatedCausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock3D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            self.up_blocks.append(UpDecoderBlock3D(
                prev_output_channel,
                output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                add_upsample=i < len(block_out_channels) - 1,
                temporal_up=i < temporal_up_num,
            ))
        self.conv_norm_out = ops.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_out = InflatedCausalConv3d(block_out_channels[0], out_channels, 3, padding=1)

    # Frames per pass through the spatial-only tail; None runs a slice's frames at once.
    tail_frames = 1

    def _tail(self, source, memory_state, memory_cache):
        for up_block in self.up_blocks[self.temporal_up_num:]:
            source = [up_block.run(source, memory_state, memory_cache)]
        out = self.conv_out.fused_norm_conv(source, self.conv_norm_out, memory_state, memory_cache)
        if out is None:
            sample = causal_norm_wrapper(self.conv_norm_out, source.pop(), silu=True)
            out = self.conv_out(sample, memory_state=memory_state, memory_cache=memory_cache)
        return out

    def forward(self, sample, memory_state, memory_cache):
        sample = self.conv_in(sample, memory_state=memory_state, memory_cache=memory_cache)
        sample = self.mid_block(sample, memory_state, memory_cache)

        *head, last = self.up_blocks[:self.temporal_up_num]
        box = [sample]
        del sample
        for up_block in head:
            box = [up_block.run(box, memory_state, memory_cache)]
        sample = box.pop()
        for resnet in last.resnets:
            sample = resnet(sample, memory_state, memory_cache)

        # From the last temporal upsampler on, everything is causal and per frame: it runs an input
        # frame at a time, each of its output frames going on through the spatial-only tail as it is made.
        ups = sum(1 for blk in self.up_blocks[self.temporal_up_num - 1:] if blk.upsamplers is not None)
        frame_pixels = sample.size(-2) * sample.size(-1) * 4 ** ups
        up = last.upsamplers[0]

        def upsample_and_tail(source, state, cache):
            if up.frames_apply(source[0], state):
                outs = [self._tail(box, state if n == 0 else MemoryState.ACTIVE, cache)
                        for n, box in enumerate(up.frames(source, state, cache))]
                return torch.cat(outs, dim=2)
            box = [up.run(source, state, cache)]
            return _frame_chunked(self, self._tail, box, self.tail_frames, state, cache, frame_pixels)
        source = [sample]
        del sample
        return _frame_chunked(last, upsample_and_tail, source, self.tail_frames and max(1, self.tail_frames // up.temporal_ratio),
                              memory_state, memory_cache, frame_pixels)


class VideoAutoencoderKLWrapper(nn.Module):
    """The SeedVR2 causal video VAE, run a temporal slice at a time with the causal caches carried
    across slices."""
    spatial_downsample_factor = BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE
    temporal_downsample_factor = BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE
    slicing_sample_min_size = BYTEDANCE_SLICING_SAMPLE_MIN
    slicing_latent_min_size = BYTEDANCE_SLICING_SAMPLE_MIN // BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D(3, SEEDVR2_LATENT_CHANNELS, BYTEDANCE_BLOCK_OUT_CHANNELS, layers_per_block=2,
                                 norm_num_groups=32, temporal_down_num=2)
        self.decoder = Decoder3D(SEEDVR2_LATENT_CHANNELS, 3, BYTEDANCE_BLOCK_OUT_CHANNELS, layers_per_block=2,
                                 norm_num_groups=32, temporal_up_num=2)
        set_norm_limit(BYTEDANCE_VAE_NORM_MEM_GIB)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.memory_limit = BYTEDANCE_VAE_CONV_MEM_GIB

    def _encode(self, x, memory_state=MemoryState.DISABLED, memory_cache=None):
        h = self.encoder(x.to(self.device), memory_state=memory_state, memory_cache=memory_cache)
        return h.to(x.device)

    def _decode(self, z, memory_state=MemoryState.DISABLED, memory_cache=None):
        output = self.decoder(z.to(self.device), memory_state=memory_state, memory_cache=memory_cache)
        return output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) > 1:   # a video at a time: the buffered path and the memory estimate are per video
            return torch.cat([self.slicing_encode(x[i:i + 1]) for i in range(x.size(0))])
        if (x.shape[2] - 1) <= self.slicing_sample_min_size:
            return self._encode(x.to(self.device))
        memory_cache = CausalMemoryCache(offload=_offload_caches_for(x.shape[-2] * x.shape[-1]))
        split_size = max(self.slicing_sample_min_size, self.temporal_downsample_factor)
        x_slices = list(x[:, :, 1:].split(split_size=split_size, dim=2))   # _encode moves each to the device
        if len(x_slices) > 1 and x_slices[-1].shape[2] < self.temporal_downsample_factor:
            x_slices[-2] = torch.cat((x_slices[-2], x_slices[-1]), dim=2)
            x_slices.pop()
        try:
            encoded_slices = [self._encode(torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                                           memory_state=MemoryState.INITIALIZING, memory_cache=memory_cache)]
            for x_slice in x_slices[1:]:
                encoded_slices.append(self._encode(x_slice, memory_state=MemoryState.ACTIVE, memory_cache=memory_cache))
        finally:
            memory_cache.free()
        return torch.cat(encoded_slices, dim=2)

    def slicing_decode(self, z: torch.Tensor, output_buffer=None) -> torch.Tensor:
        """Decode slice by slice; with ``output_buffer`` each slice is written into it (no cat, no accumulation)."""
        if z.size(0) > 1:   # a video at a time: the buffered path and the memory estimate are per video
            outs = [self.slicing_decode(z[i:i + 1], None if output_buffer is None else output_buffer[i:i + 1])
                    for i in range(z.size(0))]
            return output_buffer if output_buffer is not None else torch.cat(outs)
        if (z.shape[2] - 1) <= self.slicing_latent_min_size:
            decoded = self._decode(z)
            return decoded if output_buffer is None else output_buffer.copy_(decoded)
        memory_cache = CausalMemoryCache(offload=_offload_caches_for(z.shape[-2] * z.shape[-1] * self.spatial_downsample_factor ** 2))
        z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size, dim=2)
        decoded_slices = []
        write_pos = 0

        def emit(decoded):
            nonlocal write_pos
            if output_buffer is None:
                decoded_slices.append(decoded)
                return
            output_buffer[:, :, write_pos:write_pos + decoded.size(2)].copy_(decoded)
            write_pos += decoded.size(2)

        try:
            emit(self._decode(torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                              memory_state=MemoryState.INITIALIZING, memory_cache=memory_cache))
            for z_slice in z_slices[1:]:
                emit(self._decode(z_slice, memory_state=MemoryState.ACTIVE, memory_cache=memory_cache))
        finally:
            memory_cache.free()
        return output_buffer if output_buffer is not None else torch.cat(decoded_slices, dim=2)

    def encode(self, x, device=None):
        """The latent mean of (b, c, t, h, w) pixels, or of an image batch; a single latent frame loses
        its time axis. sd.py leaves the pixels wherever they were when it uses the chunked-io protocol:
        the slicing moves each slice to ``device`` as it encodes, so the whole clip never sits on the GPU."""
        if x.ndim == 4:
            x = x.unsqueeze(2)
        self.device = x.device if device is None else device
        return self.slicing_encode(x).chunk(2, dim=1)[0].squeeze(2)

    # sd.py preallocates the output on its device/dtype and hands slices of it to decode(); the
    # decoded frames then never accumulate on the GPU (see slicing_decode).
    comfy_has_chunked_io = True

    def decode_output_shape(self, input_shape):
        b, c, t, h, w = self._latent_dims(input_shape)
        frames = max(1, (t - 1) * self.temporal_downsample_factor + 1)
        return (b, 3, frames, h * self.spatial_downsample_factor, w * self.spatial_downsample_factor)

    @staticmethod
    def _latent_dims(shape):
        """(b, c, t, h, w) of a 5-D latent or the collapsed 4-D (B, 16*T, H, W) form."""
        if len(shape) == 5:
            return tuple(shape)
        b, tc, h, w = shape
        return b, SEEDVR2_LATENT_CHANNELS, tc // SEEDVR2_LATENT_CHANNELS, h, w

    def _unscaled(self, z):
        """A scaled latent, in either form, as the decoder's (b, c, t, h, w) input."""
        return z.reshape(self._latent_dims(z.shape)) / BYTEDANCE_VAE_SCALING_FACTOR + BYTEDANCE_VAE_SHIFTING_FACTOR

    def decode(self, z, output_buffer=None):
        latent = self._unscaled(z)
        self.device = latent.device
        return self.slicing_decode(latent, output_buffer=output_buffer)

    # comfy_memory_used_decode is fitted against measured peaks, so a decode it says will not fit
    # can be tiled without trying first. See preferred_decode_tile for the sizing that follows.
    comfy_decode_estimate_is_reliable = True

    def preferred_decode_tile(self, free_memory, dtype):
        """Largest square tile, in latent units, whose decode fits ``free_memory``, from the same per-pixel
        figures as the estimate: the temporal caches count too when there is no room to pin them."""
        factor = _eager_factor(dtype, SEEDVR2_EAGER_DECODE_FACTOR)
        budget = free_memory * SEEDVR2_TILE_MEM_HEADROOM - SEEDVR2_DECODE_FIXED_BYTES * factor
        if budget <= 0:
            return SEEDVR2_MIN_TILE_LATENT
        area = budget / (SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL * factor)
        if not _offload_caches_for(area):
            area = budget / ((SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL + SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL) * factor)
        side = int(math.sqrt(max(area, 1.0))) // self.spatial_downsample_factor
        side = (side // 8) * 8  # keep tiles a whole number of latent blocks
        return max(SEEDVR2_MIN_TILE_LATENT, min(SEEDVR2_MAX_TILE_LATENT, side))

    def decode_tiled(self, z, tile_x=32, tile_y=32, overlap=8, tile_t=None, overlap_t=None):
        # SeedVR2's causal VAE owns temporal via the MemoryState cache; external
        # temporal tiling breaks that continuity, so only spatial tiling is applied.
        latent = self._unscaled(z)
        self.device = latent.device
        sf = self.spatial_downsample_factor
        tile_h, tile_w, ov = tile_y * sf, tile_x * sf, overlap * sf
        return tiled_vae(latent, self, tile_size=(tile_h, tile_w),
                         tile_overlap=(min(ov, max(0, tile_h - 8)), min(ov, max(0, tile_w - 8))), encode=False)

    def encode_tiled(self, x, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        # External temporal tiling knobs are discarded; the causal VAE keeps its
        # own internal MemoryState slicing.
        tile_y = 512 if tile_y is None else tile_y
        tile_x = 512 if tile_x is None else tile_x
        overlap = 64 if overlap is None else overlap
        self.device = x.device
        return tiled_vae(x, self, tile_size=(tile_y, tile_x),
                         tile_overlap=(min(overlap, max(0, tile_y - 8)), min(overlap, max(0, tile_x - 8))), encode=True)

    def comfy_format_encoded(self, samples):
        if samples.ndim == 4:
            samples = samples.unsqueeze(2)
        samples = samples.contiguous()
        samples = samples * BYTEDANCE_VAE_SCALING_FACTOR
        return samples

    def comfy_memory_used_encode(self, shape, dtype):
        """Peak device memory of an encode of (b, c, t, h, w) pixels, in bytes: one frame's area sets it
        (the videos of a batch are encoded one at a time)."""
        peak = shape[-2] * shape[-1] * SEEDVR2_ENCODE_BYTES_PER_PIXEL + SEEDVR2_ENCODE_FIXED_BYTES
        return int(peak * _eager_factor(dtype, SEEDVR2_EAGER_ENCODE_FACTOR))

    def comfy_memory_used_decode(self, shape, dtype):
        """Peak device memory of a decode, in bytes: one frame's area sets it, the clip barely moves it
        (the videos of a batch are decoded one at a time)."""
        _, _, latent_t, latent_h, latent_w = self._latent_dims(shape)
        area = latent_h * self.spatial_downsample_factor * latent_w * self.spatial_downsample_factor
        frames = max(1, (latent_t - 1) * self.temporal_downsample_factor + 1)
        decode_peak = (
            area * SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL
            + frames * area * SEEDVR2_DECODE_BYTES_PER_OUTPUT_PIXEL
            + SEEDVR2_DECODE_FIXED_BYTES
        )
        if not _offload_caches_for(area):
            decode_peak += area * SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL   # the temporal caches stay resident
        # The node's colour correction runs after the decode, one frame (LAB) or a free-memory
        # sized chunk (wavelet, adain) at a time with its own OOM back-off, so it never coincides
        # with the decode's working set and only a frame of it is worth reserving.
        colour_transfer = area * SEEDVR2_DECODE_LAB_BYTES_PER_OUTPUT_PIXEL
        return int(max(decode_peak * _eager_factor(dtype, SEEDVR2_EAGER_DECODE_FACTOR), colour_transfer))
