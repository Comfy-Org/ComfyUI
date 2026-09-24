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
    SEEDVR2_VAE_CACHE_QUANT_BYTES,
    SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL,
    SEEDVR2_DECODE_BYTES_PER_OUTPUT_PIXEL,
    SEEDVR2_DECODE_FIXED_BYTES,
    SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL,
    SEEDVR2_DECODE_LAB_BYTES_PER_OUTPUT_PIXEL,
    SEEDVR2_ENCODE_BYTES_PER_PIXEL,
    SEEDVR2_ENCODE_FIXED_BYTES,
    SEEDVR2_MAX_TILE_LATENT,
    SEEDVR2_MIN_TILE_LATENT,
    SEEDVR2_TILE_MEM_HEADROOM,
)
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.model import vae_attention

import collections
import dataclasses
import math
from enum import Enum

import logging
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


_CACHE_MISS = object()


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

    def _entries_of(self, top):
        ring = self._rings.get(top)
        return list(ring) if ring is not None else [top]

    def _prefetch_after(self, top, device):
        """The reader after this one is next in line; have its tail on the way while this one runs."""
        try:
            nxt = self._order[self._order.index(top) + 1]
        except (ValueError, IndexError):
            return
        for sub in self._entries_of(nxt):
            if sub in self.packed and sub not in self._staged:
                self._staged[sub] = self._bring_back(sub, device)

    def push_frames(self, key, frames, keep):
        """Extend ``key``'s tail by ``frames``, keeping the last ``keep``; each frame is its own entry."""
        if key in self.plain or key in self.packed:
            whole = self.pop(key)
            ring = self._rings[key] = collections.deque()
            for j in range(whole.size(2)):
                self._ring_serial += 1
                sub = (key, self._ring_serial)
                self[sub] = whole[:, :, j:j + 1].clone(memory_format=_CL)
                ring.append(sub)
        ring = self._rings.setdefault(key, collections.deque())
        for j in range(frames.size(2)):
            self._ring_serial += 1
            sub = (key, self._ring_serial)
            self[sub] = frames[:, :, j:j + 1].clone(memory_format=_CL)
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
        if key in self._rings:   # a whole tail replaces the ring
            for sub in self._rings.pop(key):
                self.discard(sub)
        self.discard(key)
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
        ring = self._rings.get(key)
        if ring is not None:
            frames = [self._get_entry(sub) for sub in ring]
            out = torch.cat(frames, dim=2) if len(frames) > 1 else frames[0]
        elif key in self.plain or key in self.packed:
            out = self._get_entry(key)
        else:
            return default
        if self.offload and self._device is not None:
            self._prefetch_after(key, self._device)
        return out

    def for_each_frame(self, key, fn, count=None):
        """``fn(i, frame)`` over get(key)'s first ``count`` frames one at a time, never holding the
        whole unpacked tail."""
        ring = self._rings.get(key)
        if ring is None:
            whole = self._get_entry(key)
            for i in range(whole.size(2) if count is None else count):
                fn(i, whole[:, :, i:i + 1])
            del whole
        else:
            for i, sub in enumerate(list(ring)[:count]):
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
        if key not in self:
            return default
        value = self.get(key)
        for sub in self._rings.pop(key, None) or (key,):
            self.discard(sub)
        return value

    def discard(self, key):
        """Drop an entry without bringing it back; its host buffers are reused."""
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

    def __getitem__(self, key):
        value = self.get(key, _CACHE_MISS)
        if value is _CACHE_MISS:
            raise KeyError(key)
        return value


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
    """``run(x, memory_state, memory_cache)`` over a causal, spatial-only stage ``chunk`` frames at a
    time, the cache carrying state between chunks: exact, and only a chunk's activations are live.
    ``source`` (a 1-list) is emptied, so nothing here keeps the whole input alive."""
    sample = source.pop()
    if not chunk or sample.size(2) <= chunk:
        return run(sample, memory_state, memory_cache)
    local_cache = memory_state == MemoryState.DISABLED or memory_cache is None
    if local_cache:
        # under the same host-memory policy as a sliced pass
        memory_cache = CausalMemoryCache(offload=_offload_caches_for(frame_pixels))
        memory_state = MemoryState.INITIALIZING
    outs = []
    try:
        pieces = list(sample.split(chunk, dim=2))
        del sample
        if memory_cache.offload and pieces[0].is_contiguous(memory_format=_CL):
            # the rest waits in host memory, fetched a chunk ahead: resident, the whole input sat
            # under every chunk's peak (2.5 GiB for the decoder's tail at 1080p)
            pieces[0] = pieces[0].clone(memory_format=_CL)
            pieces[1:] = memory_cache.park(owner, pieces[1:])

        def take(i):   # nothing here keeps the chunk alive once the stage moves past it
            piece, pieces[i] = pieces[i], None
            return piece if torch.is_tensor(piece) else memory_cache.pop(piece)
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


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor, silu: bool = False) -> torch.Tensor:
    """Per-frame normalization, with the SiLU folded in where the kernel does both."""
    input_dtype = x.dtype
    if isinstance(norm_layer, (nn.LayerNorm, nn.RMSNorm)):
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1)
            x = norm_layer(x)
            x = x.permute(0, 3, 1, 2)
            if silu:
                x = F.silu(x)
            return x.to(input_dtype)
        if x.ndim == 5:
            x = x.permute(0, 2, 3, 4, 1)
            x = norm_layer(x)
            x = x.permute(0, 4, 1, 2, 3)
            if silu:
                x = F.silu(x)
            return x.to(input_dtype)
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            out = norm_layer(x)
            if silu:
                out = F.silu(out)
            return out.to(input_dtype)
        if x.ndim == 5:
            if _ck_eligible(x) and isinstance(norm_layer, nn.GroupNorm) and norm_layer.affine:
                # Per-frame statistics, like the reshape below; accepts either layout and emits
                # channels_last_3d, which is where the convolution wants its input.
                weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm_layer, x, offloadable=True)
                try:
                    return comfy_kitchen.group_norm_silu_pad3d(
                        x, weight, bias, norm_layer.num_groups, norm_layer.eps, (0, 0, 0, 0, 0), silu,
                    ).to(input_dtype)
                finally:
                    comfy.ops.uncast_bias_weight(norm_layer, weight, bias, offload_stream)
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
            if silu:
                x = F.silu(x)
            return x.to(input_dtype)
    raise TypeError(f"SeedVR2 VAE unsupported norm layer type: {type(norm_layer).__name__}")

_receptive_field_t = Literal["half", "full"]

def extend_head(tensor, times: int = 2, memory = None):
    if memory is not None:
        return torch.cat((_match_layout(memory.to(tensor), tensor), tensor), dim=2)
    if times < 0:
        raise ValueError(f"SeedVR2 VAE extend_head expected times >= 0, got {times}.")
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        head = _match_layout(torch.tile(tensor[:, :, :1], tile_repeat), tensor)
        return torch.cat(tensors=(head, tensor), dim=2)

def cache_send_recv(tensor, cache_size, times, memory=None):
    recv_buffer = None

    if memory is not None:
        recv_buffer = _match_layout(memory.to(tensor[0]), tensor[0])
    elif times > 0:
        tile_repeat = [1] * tensor[0].ndim
        tile_repeat[2] = times
        recv_buffer = _match_layout(torch.tile(tensor[0][:, :, :1], tile_repeat), tensor[0])

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

    def _conv_without_padding_copy(self, x, prev_cache, padding):
        """Convolve with native symmetric padding and slice off the surplus; None with a halo or stride."""
        if prev_cache is not None or any(st != 1 for st in self.stride):
            return None

        pads = ((padding[4], padding[5]), (padding[2], padding[3]), (padding[0], padding[1]))
        native = tuple(max(lo, hi) for lo, hi in pads)

        orig_padding = self.padding
        self.padding = native
        try:
            out = torch.nn.Conv3d.forward(self, x)
        finally:
            self.padding = orig_padding

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

    def _ck_buffered(self, like, x_shape, memory_state, memory_cache, write_body, residual=None):
        """Run this conv from one padded buffer: halo in front, ``write_body`` writes the padded frames
        behind it, and the conv reads and writes frame-window views. ``residual`` goes into the kitchen
        conv's epilogue (added explicitly on the cuDNN path). Replicates forward()'s cache handling.
        """
        b, c, t, h, w = x_shape
        _, ph, pw = self.padding
        kt = self.kernel_size[0]
        cache_size = kt - self.stride[0]
        if memory_cache is None:
            memory_cache = CausalMemoryCache()   # DISABLED: nothing is kept anyway
        elif memory_state != MemoryState.ACTIVE:
            memory_cache.pop(self, None)
        has_memory = self in memory_cache
        halo = cache_size if has_memory else self.temporal_padding * 2
        if t == 1 and halo == 2 and kt == 3:
            return self._ck_taps(like, x_shape, memory_state, memory_cache, write_body, residual, has_memory)
        hp, wp = h + 2 * ph, w + 2 * pw

        buf = torch.empty((b, c, halo + t, hp, wp), dtype=like.dtype, device=like.device, memory_format=_CL)
        write_body(buf[:, :, halo:])
        del write_body   # and with it the source it captured: nothing but the buffer feeds the conv
        if halo:
            front = buf[:, :, :halo]
            if has_memory:
                # after the body, whose source is freed by then: the unpacked halo is two full frames
                core = front[:, :, :, ph:ph + h, pw:pw + w]
                memory_cache.for_each_frame(self, lambda i, frame: core[:, :, i:i + 1].copy_(frame))
                for dim, lo, hi in ((3, ph, ph + h), (4, pw, pw + w)):
                    if lo:
                        front.narrow(dim, 0, lo).zero_()
                        front.narrow(dim, hi, lo).zero_()   # symmetric border
            else:
                # extend_head's tiled first frame, already padded
                front.copy_(buf[:, :, halo:halo + 1].expand(-1, -1, halo, -1, -1))
        if cache_size and memory_state in (MemoryState.INITIALIZING, MemoryState.ACTIVE):
            core = buf[:, :, :, ph:ph + h, pw:pw + w]
            if has_memory:
                # the halo frames are already the cache's newest entries: push only the new ones
                # (push_frames splits a tail that was stored whole before extending it)
                memory_cache.push_frames(self, core[:, :, halo:][:, :, -min(t, cache_size):], cache_size)
            else:
                memory_cache.push_frames(self, core[:, :, -cache_size:], cache_size)

        _ndhwc_filter(self)
        weight, bias, offload_stream = comfy.ops.cast_bias_weight(self, buf, offloadable=True)
        try:
            weight_cl = weight.contiguous(memory_format=_CL)
            stride = tuple(self.stride)
            fused_residual = None
            if not _fp16_accumulate_wanted(buf):
                out = F.conv3d(buf, weight_cl, bias, stride=stride)   # cuDNN, NDHWC in and out
            else:
                zt = buf.size(2) - kt + 1
                out = torch.empty((b, weight.size(0), zt, hp - self.kernel_size[1] + 1, wp - self.kernel_size[2] + 1),
                                  dtype=buf.dtype, device=buf.device, memory_format=_CL)
                # each call's input and output views stay under the kernel's 32-bit offsets
                per_frame = max(c * hp * wp, weight.size(0) * out.size(3) * out.size(4))
                chunk = max(1, min(zt, self._CK_MAX_VIEW_ELEMENTS // per_frame - (kt - 1)))
                if residual is not None and residual.shape == out.shape and residual.dtype == out.dtype and _ndhwc_like(residual):
                    fused_residual = residual   # otherwise added after: the epilogue wants the output's shape and layout
                for z0 in range(0, zt, chunk):
                    z1 = min(zt, z0 + chunk)
                    comfy_kitchen.fp16_conv3d(buf[:, :, z0:z1 + kt - 1], weight_cl, bias,
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
        run on cuDNN straight from the unpacked frames, its padding in place of a buffer. Without a
        cache the halo is the frame itself, so the taps sum first."""
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
            if _fp16_accumulate_wanted(buf):
                out = comfy_kitchen.fp16_conv3d(buf, tap, bias)
            else:
                out = F.conv3d(buf, tap, bias)
            if memory_state in (MemoryState.INITIALIZING, MemoryState.ACTIVE):
                # pushed now so the buffer can go; three kept, as the oldest is still this call's first tap
                core = buf[:, :, :, ph:ph + h, pw:pw + w]
                memory_cache.push_frames(self, core if has_memory else core.expand(-1, -1, 2, -1, -1), 3)
            del buf
            if has_memory:
                def halo_tap(i, frame):
                    out.add_(F.conv3d(frame, weight[:, :, i:i + 1].contiguous(memory_format=_CL), padding=(0, ph, pw)))
                memory_cache.for_each_frame(self, halo_tap, count=2)
                memory_cache.keep_last(self, 2)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload_stream)
        if residual is not None:
            out += residual
        return out

    def _ck_buffered_applies(self, x, memory_state):
        """Whether _ck_buffered can take this call: the frame-offset write needs a batch of one,
        and stride is not folded into the buffer sizes."""
        return (self._ck_applies(x) and memory_state != MemoryState.UNSET and x.size(0) == 1
                and tuple(self.stride) == (1, 1, 1))

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
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            if self._ck_applies(x):
                return self._ck_conv(self._padded_input(x, None, self._symmetric_padding(), split_dim - 1, channels_last=True))
            return super().forward(x)

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
            with ignore_padding(self):
                return torch.nn.Conv3d.forward(self, padded)

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
            head = memory.to(input)
        elif self.temporal_padding:
            head = torch.tile(input[:, :, :1], [1, 1, self.temporal_padding * 2, 1, 1])
        else:
            head = None
        if self._ck_applies(input):
            # The halo and the zero border share one buffer: the kernel needs a padded copy
            # anyway, so the cat that extend_head would do is folded into it. On the largest
            # activations that is one full pass and one activation's worth of peak.
            padded = self._padded_input(input, head, self._symmetric_padding(), 2, channels_last=True)
            if mem_size != 0 and memory_state != MemoryState.DISABLED and memory_cache is not None:
                # From the halo'd buffer, as extend_head's result would be: a one-frame slice
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
                # clone, not contiguous(): for a batch of one a frame slice of an NDHWC tensor
                # already satisfies the channels_last_3d check, and contiguous() would return the
                # view, pinning the whole activation for the next slice instead of two frames.
                tail = input[-1][:, :, -cache_size:].detach()
                memory_cache[self] = tail.clone(
                    memory_format=_CL if input[-1].is_contiguous(memory_format=_CL) else torch.contiguous_format)

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

        if (_ck_eligible(hidden_states) and _fp16_accumulate_wanted(hidden_states)
                and hidden_states.is_contiguous(memory_format=_CL)):
            _ndhwc_filter(self.upscale_conv)
            weight, bias, offload_stream = comfy.ops.cast_bias_weight(self.upscale_conv, hidden_states, offloadable=True)
            try:
                hidden_states = comfy_kitchen.fp16_conv3d(hidden_states, weight.contiguous(memory_format=_CL), bias)
            finally:
                comfy.ops.uncast_bias_weight(self.upscale_conv, weight, bias, offload_stream)
        else:
            hidden_states = self.upscale_conv(hidden_states)
        b, channels, f, h, w = hidden_states.shape
        c = channels // (self.spatial_ratio * self.spatial_ratio * self.temporal_ratio)
        # channels_last_3d is only defined for 5-D tensors, so decide before the view.
        keep_layout = hidden_states.is_contiguous(memory_format=_CL)
        buffered = keep_layout and self.conv._ck_buffered_applies(hidden_states, memory_state)
        hidden_states = hidden_states.view(b, self.spatial_ratio, self.spatial_ratio, self.temporal_ratio, c, f, h, w)
        f_out, h_out, w_out = f * self.temporal_ratio, h * self.spatial_ratio, w * self.spatial_ratio
        if buffered:
            # One strided copy from the (b, sr, sr, tr, c, f, h, w) view into the conv's padded buffer;
            # the source is boxed so the copy is the last thing to hold it.
            sr, tr = self.spatial_ratio, self.temporal_ratio
            drop_head = self.temporal_up and memory_state != MemoryState.ACTIVE
            _, ph, pw = self.conv.padding
            like = hidden_states.new_empty(0)                 # dtype/device only, shares nothing
            source = [hidden_states]
            del hidden_states

            def as_split(core, frames):
                """(b, c, F, H, W) view -> (b, sr, sr, tr, c, f, h, w), the source's order."""
                return core.view(b, c, frames // tr, tr, h, sr, w, sr).permute(0, 5, 7, 3, 1, 2, 4, 6)

            def write_body(dst):
                for dim, lo, hi in ((3, ph, ph + h_out), (4, pw, pw + w_out)):
                    if lo:
                        dst.narrow(dim, 0, lo).zero_()
                        dst.narrow(dim, hi, dst.size(dim) - hi).zero_()
                core = dst[:, :, :, ph:ph + h_out, pw:pw + w_out]           # (b, c, F', H, W)
                src = source.pop()
                if not drop_head:
                    as_split(core, f_out).copy_(src)
                    return
                # frame 0 <- (f=0, t=0); frames 1.. <- f >= 1 (temporal_up means tr == 2)
                core[:, :, 0].view(b, c, h, sr, w, sr).permute(0, 3, 5, 1, 2, 4).copy_(
                    src[:, :, :, 0, :, 0])
                as_split(core[:, :, 1:], f_out - tr).copy_(src[:, :, :, :, :, 1:])
            return self.conv._ck_buffered(like, (b, c, f_out - int(drop_head), h_out, w_out),
                                          memory_state, memory_cache, write_body)
        if keep_layout:
            # Same element mapping as below, written so the copy lands in NDHWC and the
            # following convolution keeps the layout.
            hidden_states = hidden_states.permute(0, 5, 3, 6, 1, 7, 2, 4).reshape(
                b, f * self.temporal_ratio, h * self.spatial_ratio, w * self.spatial_ratio, c,
            ).permute(0, 4, 1, 2, 3)
        else:
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
        hidden_states = self.conv1.fused_norm_conv([input_tensor], self.norm1, memory_state, memory_cache)
        if hidden_states is None:
            hidden_states = causal_norm_wrapper(self.norm1, input_tensor, silu=True)
            hidden_states = self.conv1(hidden_states, memory_state=memory_state, memory_cache=memory_cache)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None:
            hidden_states = hidden_states + temb

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state, memory_cache=memory_cache)

        # The skip connection rides in the conv's epilogue when the scale is one (it always is in
        # the decoder), which spares a pass over the output per block.
        residual = input_tensor if self.output_scale_factor == 1.0 else None
        source = [hidden_states]
        del hidden_states
        hidden_states = self.conv2.fused_norm_conv(source, self.norm2, memory_state, memory_cache, residual=residual)
        if hidden_states is None:
            hidden_states = causal_norm_wrapper(self.norm2, source.pop(), silu=True)
            hidden_states = self.conv2(hidden_states, memory_state=memory_state, memory_cache=memory_cache)
            residual = None
        if residual is None:
            return (input_tensor + hidden_states) / self.output_scale_factor
        return hidden_states


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
        self.head_blocks = max(0, len(block_out_channels) - temporal_down_num - 1)   # the spatial-only down blocks

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


    # Frames per pass through the full-resolution, spatial-only head; None runs a slice's frames at once.
    head_frames = 1

    def _head(self, sample, memory_state, memory_cache):
        sample = self.conv_in(sample, memory_state=memory_state, memory_cache=memory_cache)
        for down_block in self.down_blocks[:self.head_blocks]:
            sample = down_block(sample, memory_state=memory_state, memory_cache=memory_cache)
        return sample

    def forward(
        self,
        sample: torch.FloatTensor,
        memory_state = None,
        memory_cache = None,
    ) -> torch.FloatTensor:
        sample = sample.to(next(self.parameters()).device)
        frame_pixels = sample.size(-2) * sample.size(-1)
        source = [sample]
        del sample
        sample = _frame_chunked(self, self._head, source, self.head_frames, memory_state, memory_cache, frame_pixels)
        for down_block in self.down_blocks[self.head_blocks:]:
            sample = down_block(sample, memory_state=memory_state, memory_cache=memory_cache)

        sample = self.mid_block(sample, memory_state=memory_state, memory_cache=memory_cache)

        sample = causal_norm_wrapper(self.conv_norm_out, sample, silu=True)
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

    # Frames per pass through the spatial-only tail; None runs a slice's frames at once.
    tail_frames = 1

    def _tail(self, sample, latent_embeds, memory_state, memory_cache):
        for up_block in self.up_blocks[self.temporal_up_num:]:
            sample = up_block(sample, latent_embeds, memory_state=memory_state, memory_cache=memory_cache)
        sample = causal_norm_wrapper(self.conv_norm_out, sample, silu=True)
        return self.conv_out(sample, memory_state=memory_state, memory_cache=memory_cache)

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

        *head, last = self.up_blocks[:self.temporal_up_num]
        for up_block in head:
            sample = up_block(sample, latent_embeds, memory_state=memory_state, memory_cache=memory_cache)
        for resnet in last.resnets:
            sample = resnet(sample, temb=None, memory_state=memory_state, memory_cache=memory_cache)

        # From the last temporal upsampler on, everything is causal and per frame: it runs an input
        # frame at a time (two output frames), and the spatial-only tail a frame at a time within that.
        ups = sum(1 for blk in self.up_blocks[self.temporal_up_num - 1:] if blk.upsamplers is not None)
        frame_pixels = sample.size(-2) * sample.size(-1) * 4 ** ups
        tr = last.upsamplers[0].temporal_ratio

        def upsample_and_tail(x, state, cache):
            for upsampler in last.upsamplers:
                x = upsampler(x, memory_state=state, memory_cache=cache)
            source = [x]
            del x
            return _frame_chunked(self, lambda y, s, c: self._tail(y, latent_embeds, s, c),
                                  source, self.tail_frames, state, cache, frame_pixels)
        source = [sample]
        del sample
        return _frame_chunked(last, upsample_and_tail, source, self.tail_frames and max(1, self.tail_frames // tr),
                              memory_state, memory_cache, frame_pixels)

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
        self, z: torch.Tensor, return_dict: bool = True, output_buffer=None
    ):
        decoded = self.slicing_decode(z, output_buffer=output_buffer)

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
            memory_cache = CausalMemoryCache(offload=_offload_caches_for(x.shape[-2] * x.shape[-1]))
            split_size = max(
                self.slicing_sample_min_size,
                getattr(self, "temporal_downsample_factor", 1),
            )
            x_slices = list(x[:, :, 1:].split(split_size=split_size, dim=2))   # _encode moves each to the device
            min_active_len = getattr(self, "temporal_downsample_factor", 1)
            if len(x_slices) > 1 and x_slices[-1].shape[2] < min_active_len:
                x_slices[-2] = torch.cat((x_slices[-2], x_slices[-1]), dim=2)
                x_slices.pop()
            try:
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
            finally:
                memory_cache.free()
            return torch.cat(encoded_slices, dim=2)
        else:
            return self._encode(x.to(self.device))

    def slicing_decode(self, z: torch.Tensor, output_buffer=None) -> torch.Tensor:
        """Decode slice by slice; with ``output_buffer`` each slice is written into it (no cat, no accumulation)."""
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size:
            memory_cache = CausalMemoryCache(offload=_offload_caches_for(z.shape[-2] * z.shape[-1] * self.spatial_downsample_factor ** 2))
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size, dim=2)
            decoded_slices = []
            write_pos = 0

            def emit(decoded):
                nonlocal write_pos
                if output_buffer is None:
                    decoded_slices.append(decoded)
                    return
                frames = decoded.size(2)
                output_buffer[:, :, write_pos:write_pos + frames].copy_(
                    decoded[..., :output_buffer.size(-2), :output_buffer.size(-1)])
                write_pos += frames

            try:
                emit(self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                    memory_cache=memory_cache,
                ))
                for z_idx in range(1, len(z_slices)):
                    emit(self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE, memory_cache=memory_cache))
            finally:
                memory_cache.free()
            if output_buffer is not None:
                return output_buffer
            return torch.cat(decoded_slices, dim=2)
        decoded = self._decode(z)
        if output_buffer is None:
            return decoded
        output_buffer.copy_(decoded[..., :output_buffer.size(-2), :output_buffer.size(-1)])
        return output_buffer

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

    def _encode_with_raw_latent(self, x, device=None):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        # sd.py leaves the pixels wherever they were when it uses the chunked-io protocol; the
        # slicing below moves each slice as it encodes, so the whole clip never sits on the GPU.
        self.device = x.device if device is None else device
        p = super().encode(x)
        z = p.squeeze(2)
        return z, p

    def encode(self, x, device=None):
        z, _ = self._encode_with_raw_latent(x, device=device)
        return z

    # sd.py preallocates the output on its device/dtype and hands slices of it to decode(); the
    # decoded frames then never accumulate on the GPU (see slicing_decode).
    comfy_has_chunked_io = True

    def decode_output_shape(self, input_shape):
        b, c, t, h, w = self._latent_dims(input_shape)
        frames = max(1, (t - 1) * BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE + 1)
        ph, pw = h * self.spatial_downsample_factor, w * self.spatial_downsample_factor
        return (b, 3, frames, ph - ph % 2, pw - pw % 2)

    @staticmethod
    def _latent_dims(shape):
        """(b, c, t, h, w) of a 5-D latent or the collapsed 4-D (B, 16*T, H, W) form."""
        if len(shape) == 5:
            return tuple(shape)
        b, tc, h, w = shape
        return b, SEEDVR2_LATENT_CHANNELS, tc // SEEDVR2_LATENT_CHANNELS, h, w

    def decode(self, z, seedvr2_tiling=None, output_buffer=None):
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
            if output_buffer is not None:
                return super().decode_(latent, output_buffer=output_buffer)
            x = super().decode_(latent)

        h, w = x.shape[-2:]
        w2 = w - (w % 2)
        h2 = h - (h % 2)
        x = x[..., :h2, :w2]

        if output_buffer is not None:   # the tiled path: one copy into the caller's frames
            output_buffer.copy_(x)
            return output_buffer
        return x

    # comfy_memory_used_decode is fitted against measured peaks, so a decode it says will not fit
    # can be tiled without trying first. See preferred_decode_tile for the sizing that follows.
    comfy_decode_estimate_is_reliable = True

    def preferred_decode_tile(self, free_memory):
        """Largest square tile, in latent units, whose decode fits ``free_memory``, from the same per-pixel figure as the estimate."""
        budget = free_memory * SEEDVR2_TILE_MEM_HEADROOM - SEEDVR2_DECODE_FIXED_BYTES
        if budget <= 0:
            return SEEDVR2_MIN_TILE_LATENT
        area = budget / SEEDVR2_DECODE_BYTES_PER_FRAME_PIXEL
        side = int(math.sqrt(max(area, 1.0))) // self.spatial_downsample_factor
        side = (side // 8) * 8  # keep tiles a whole number of latent blocks
        return max(SEEDVR2_MIN_TILE_LATENT, min(SEEDVR2_MAX_TILE_LATENT, side))

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

    def comfy_memory_used_encode(self, shape):
        """Peak device memory of an encode of (b, c, t, h, w) pixels, in bytes: one frame's area sets it."""
        return shape[-2] * shape[-1] * SEEDVR2_ENCODE_BYTES_PER_PIXEL + SEEDVR2_ENCODE_FIXED_BYTES

    def comfy_memory_used_decode(self, shape):
        """Peak device memory of a decode, in bytes: one frame's area sets it, the clip barely moves it."""

        def frame_area(latent_h, latent_w):
            return latent_h * 8 * latent_w * 8

        def output_frames(latent_t):
            return max(1, (latent_t - 1) * 4 + 1)

        if len(shape) == 5:
            candidates = []
            if shape[1] == SEEDVR2_LATENT_CHANNELS:
                candidates.append((shape[2], shape[3], shape[4]))
            if shape[-1] == SEEDVR2_LATENT_CHANNELS:
                candidates.append((shape[1], shape[2], shape[3]))
            if len(candidates) == 0:
                candidates.append((shape[2], shape[3], shape[4]))
            latent_t, latent_h, latent_w = max(candidates, key=lambda c: frame_area(c[1], c[2]))
        elif len(shape) == 4:
            latent_t = max(1, (shape[1] + SEEDVR2_LATENT_CHANNELS - 1) // SEEDVR2_LATENT_CHANNELS)
            latent_h, latent_w = shape[2], shape[3]
        else:
            latent_t, latent_h, latent_w = 1, shape[-2], shape[-1]

        area = frame_area(latent_h, latent_w)
        frames = output_frames(latent_t)
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
        return int(max(decode_peak, colour_transfer))

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
