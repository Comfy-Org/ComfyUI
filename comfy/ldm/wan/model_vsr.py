# Wan video super-resolution
# RelayVSR https://github.com/kopperx/RelayVSR (Apache-2.0): one-step Wan keyframes guiding a small per-frame transformer
import math

import comfy_kitchen
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.wan.model import WanAttentionBlock, WanModel, WanSelfAttention, modulate, sinusoidal_embedding_1d
import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.utils

ops = comfy.ops.disable_weight_init

RELAY_KEYFRAME_INTERVAL = 15
RELAY_WAN_CONTEXT = 4
RELAY_WAN_WINDOW = (22, 40)
RELAY_WAN_CHUNK_TOKENS = 32768
# the per-layer cache needs about as much VRAM as this many keyframes of a single pass
RELAY_WAN_CACHE_KEYFRAMES = 7
RELAY_FLASH_CONDITION_WINDOW = (48, 48)
RELAY_FLASH_VIDEO_WINDOW = (24, 24)


def relay_keyframes(frames):
    return list(range(0, frames - 1, RELAY_KEYFRAME_INTERVAL)) + [frames - 1]


def relay_latent_size(height, width, latent_stride):
    # whole 2x2 patches of the model's pixel stride
    stride = latent_stride * 2
    return math.ceil(height / stride) * 2, math.ceil(width / stride) * 2


class RelayLQProjection(nn.Module):
    def __init__(self, upscale_factor, hidden_dim=768, hidden_dim2=3072, out_dim=3072, eps=1e-6, device=None, dtype=None, operations=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = operations.Conv2d(3 * (4 * upscale_factor) ** 2, hidden_dim, 1, device=device, dtype=dtype)
        self.norm1 = operations.RMSNorm(hidden_dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.conv2 = operations.Conv2d(hidden_dim, hidden_dim2, 3, stride=2, padding=1, device=device, dtype=dtype)
        self.norm2 = operations.RMSNorm(hidden_dim2, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.linear = operations.Linear(hidden_dim2, out_dim, device=device, dtype=dtype)

    def forward(self, video):
        b, c, f, h, w = video.shape
        x = video.movedim(2, 1).reshape(b * f, c, h, w)
        if self.upscale_factor == 4:
            x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.conv1(F.pixel_unshuffle(x, 4 * self.upscale_factor))
        x = F.silu(self.norm1(x.movedim(1, -1))).movedim(-1, 1)
        x = F.silu(self.norm2(self.conv2(x).flatten(2).transpose(1, 2)))
        return self.linear(x).view(b, -1, self.linear.out_features)


class RelayVSRSelfAttention(WanSelfAttention):
    def forward(self, x, freqs, cache, start, store, transformer_options={}):
        # causal over the last keyframes, per-query windows slid to stay inside the frame
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        frames, height, width = transformer_options["grid_sizes"]
        grid = (b, frames, height, width, n, d)
        q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs).view(grid)
        k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs).view(grid)
        v = self.v(x).view(grid)
        # keyframes before this chunk come from the cache, which holds the last ones in order as int8 rows with a scale each
        past = min(start, RELAY_WAN_CONTEXT - 1)
        keys, values = k, v
        if past > 0:
            cached, scales = cache
            k_past, v_past = (cached[:, :, :past].to(k.dtype) * scales[:, :, :past]).unbind(0)
            keys, values = torch.cat((k_past, k), dim=1), torch.cat((v_past, v), dim=1)
        x = comfy_kitchen.window_attention(q, keys, values, RELAY_WAN_WINDOW, temporal_window=RELAY_WAN_CONTEXT)
        if store:
            # the cache moves on to the last keyframes for the next chunk, older cached ones first when this one is short
            cached, scales = cache
            new = min(frames, RELAY_WAN_CONTEXT - 1)
            old = min(past, RELAY_WAN_CONTEXT - 1 - new)
            if old > 0:
                cached[:, :, :old] = cached[:, :, past - old:past].clone()
                scales[:, :, :old] = scales[:, :, past - old:past].clone()
            for i, rows in enumerate((k[:, frames - new:], v[:, frames - new:])):
                scale = (rows.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_(min=torch.finfo(rows.dtype).tiny)
                scales[i, :, old:old + new] = scale
                cached[i, :, old:old + new] = (rows.float() / scale.float()).round_().clamp_(-127.0, 127.0)
        return self.o(x.view(b, s, n * d))


class RelayVSRAttentionBlock(WanAttentionBlock):
    def forward(self, x, e, freqs, context, cache, start, store, transformer_options={}):
        # residual updated in place, nothing allocated in the block outlives its memory compiler scope
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)
        x.addcmul_(self.self_attn(modulate(x, self.norm1, e[0], e[1]), freqs, cache, start, store, transformer_options=transformer_options), e[2])
        x.add_(self.cross_attn(self.norm3(x), context, transformer_options=transformer_options))
        x.addcmul_(self.ffn(modulate(x, self.norm2, e[3], e[4])), e[5])
        return x


class RelayVSRWanModel(WanModel):
    def __init__(self, model_type=None, upscale_factor=4, operations=None, device=None, dtype=None, **kwargs):
        super().__init__(model_type="t2v", wan_attn_block_class=RelayVSRAttentionBlock, operations=operations, device=device, dtype=dtype, **kwargs)
        self.register_buffer("positive_conditioning", torch.empty((self.text_len, self.text_dim), device=device, dtype=dtype))
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}
        for block in self.blocks:
            block.self_attn = RelayVSRSelfAttention(self.dim, self.num_heads, eps=self.eps, operation_settings=operation_settings)
        self.lq_proj_in = RelayLQProjection(upscale_factor, out_dim=self.dim, device=device, dtype=dtype, operations=operations)
        self.latent_stride = 16 // upscale_factor

    def forward_orig(self, x, t, context, freqs=None, lq_pixels=None, transformer_options={}, **kwargs):
        b, frames = x.shape[0], x.shape[2]
        height, width = x.shape[3] // self.patch_size[1], x.shape[4] // self.patch_size[2]
        tokens = height * width
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x.dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)

        # keyframes go through in chunks, every layer's cache carries the keys and values of the last ones across
        chunk = max(1, RELAY_WAN_CHUNK_TOKENS // tokens)
        if frames <= chunk + RELAY_WAN_CACHE_KEYFRAMES:
            chunk = frames
        shape = (2, b, RELAY_WAN_CONTEXT - 1, height, width, self.num_heads, self.dim // self.num_heads)
        caches = [(torch.empty(shape, device=x.device, dtype=torch.int8), torch.empty((*shape[:-1], 1), device=x.device, dtype=x.dtype)) if frames > chunk else None for _ in self.blocks]
        out = torch.empty((b, self.out_dim, *x.shape[2:]), device=x.device, dtype=x.dtype)
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "double"
        starts = range(0, frames, chunk)
        # the sampler's bars only count the one step
        progress = comfy.utils.ProgressBar(len(starts) * len(self.blocks))
        with comfy.utils.model_trange(len(starts) * len(self.blocks), desc="RelayVSR", unit="block") as pbar:
            for start in starts:
                end = min(start + chunk, frames)
                # only a following chunk reads the cache
                store = end < frames
                grid_sizes = (end - start, height, width)
                transformer_options["grid_sizes"] = grid_sizes
                # the projected LQ frames go first so the sum is row-major, the blocks update it in place
                hidden = self.lq_proj_in(lq_pixels[:, :, start:end]) + self.patch_embedding(x[:, :, start:end].float()).to(x.dtype).flatten(2).transpose(1, 2)
                chunk_freqs = freqs[:, start * tokens:end * tokens]
                prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
                comfy.model_prefetch.malloc_graph_begin(x.device)
                for i, block in enumerate(self.blocks):
                    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype, malloc_scope="block")
                    transformer_options["block_index"] = i
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args):
                            return {"img": block(args["img"], args["vec"], args["pe"], args["txt"], caches[i], start, store, transformer_options=args["transformer_options"])}
                        hidden = blocks_replace[("double_block", i)]({"img": hidden, "txt": context, "vec": e0, "pe": chunk_freqs, "transformer_options": transformer_options}, {"original_block": block_wrap})["img"]
                    else:
                        hidden = block(hidden, e0, chunk_freqs, context, caches[i], start, store, transformer_options=transformer_options)
                    pbar.update(1)
                    progress.update(1)
                comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None, malloc_scope="block")
                comfy.model_prefetch.malloc_graph_end()
                out[:, :, start:end] = self.unpatchify(self.head(hidden, e), grid_sizes)
        return out

    def _forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, lq_pixels=None, **kwargs):
        if lq_pixels is None:
            raise ValueError("RelayVSR needs the low-quality video, connect the model through the RelayVSR Conditioning node.")
        t, h, w = x.shape[2:]
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size, padding_mode="replicate")
        tp, hp, wp = x.shape[2:]
        lq_height, lq_width = hp * self.latent_stride, wp * self.latent_stride
        lq_pixels = lq_pixels[..., :lq_height, :lq_width]
        lq_pixels = F.pad(lq_pixels, (0, lq_width - lq_pixels.shape[-1], 0, lq_height - lq_pixels.shape[-2], 0, 0), mode="replicate")
        freqs = self.rope_encode(tp, hp, wp, device=x.device, dtype=x.dtype, transformer_options=transformer_options)
        return self.forward_orig(x, timestep, context, freqs=freqs, lq_pixels=lq_pixels, transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]


class RelayFlashAttention(nn.Module):
    def __init__(self, dim, num_heads, kv_groups, eps, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.head_dim = dim // num_heads
        self.query = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.key = ops.Linear(dim, kv_groups * self.head_dim, device=device, dtype=dtype)
        self.value = ops.Linear(dim, kv_groups * self.head_dim, device=device, dtype=dtype)
        self.condition_key = ops.Linear(dim, kv_groups * self.head_dim, device=device, dtype=dtype)
        self.condition_value = ops.Linear(dim, kv_groups * self.head_dim, device=device, dtype=dtype)
        self.key_norm = ops.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.value_norm = ops.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.output = ops.Linear(dim, dim, device=device, dtype=dtype)

    def project(self, x, key, value, freqs):
        b, s, _ = x.shape
        q = apply_rope1(self.query(x).view(b, s, self.num_heads, self.head_dim), freqs)
        k = self.key_norm(key(x).view(b, s, self.kv_groups, self.head_dim))
        v = self.value_norm(value(x).view(b, s, self.kv_groups, self.head_dim))
        return q, k, v

    def condition(self, x, freqs, height, width):
        # the rotated condition keys are reused as they are by every frame of the window
        b, s, c = x.shape
        q, k, v = self.project(x, self.condition_key, self.condition_value, freqs)
        k = apply_rope1(k, freqs).view(b, 1, height, width, self.kv_groups, self.head_dim)
        v = v.view(k.shape)
        x = comfy_kitchen.window_attention(q.view(b, 1, height, width, self.num_heads, self.head_dim), k, v, RELAY_FLASH_CONDITION_WINDOW, clamp=False)
        return self.output(x.view(b, s, c)), (k, v)

    def video(self, x, freqs, freqs_past, height, width, condition, past):
        b, s, c = x.shape
        plane = (b, 1, height, width, self.kv_groups, self.head_dim)
        q, k, v = self.project(x, self.key, self.value, freqs)
        # both condition keyframes, then the previous frame and this one
        (k_start, v_start), (k_end, v_end) = condition
        keys, values = [k_start, k_end], [v_start, v_end]
        if past is not None:
            keys.append(apply_rope1(past[0], freqs_past).view(plane))
            values.append(past[1].view(plane))
        keys.append(apply_rope1(k, freqs).view(plane))
        values.append(v.view(plane))
        windows = [RELAY_FLASH_CONDITION_WINDOW] * 2 + [RELAY_FLASH_VIDEO_WINDOW] * (len(keys) - 2)
        x = comfy_kitchen.window_attention(q.view(b, 1, height, width, self.num_heads, self.head_dim), torch.cat(keys, dim=1), torch.cat(values, dim=1), windows, clamp=False)
        return self.output(x.view(b, s, c)), (k, v)


class RelayFlashFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, device=None, dtype=None):
        super().__init__()
        self.gate_up_proj = ops.Linear(dim, 2 * hidden_dim, bias=False, device=device, dtype=dtype)
        self.down = ops.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.down, self.gate_up_proj(x), "swiglu")


class RelayFlashBlock(nn.Module):
    def __init__(self, dim, num_heads, kv_groups, hidden_dim, eps, device=None, dtype=None):
        super().__init__()
        self.attention_norm = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.attention = RelayFlashAttention(dim, num_heads, kv_groups, eps, device=device, dtype=dtype)
        self.feed_forward_norm = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)
        self.feed_forward = RelayFlashFeedForward(dim, hidden_dim, device=device, dtype=dtype)

    def condition(self, x, *args):
        y, kv = self.attention.condition(self.attention_norm(x), *args)
        x = x + y
        return x + self.feed_forward(self.feed_forward_norm(x)), kv

    def video(self, x, *args):
        y, kv = self.attention.video(self.attention_norm(x), *args)
        x = x + y
        return x + self.feed_forward(self.feed_forward_norm(x)), kv


class RelayFlashEncoder(nn.Module):
    def __init__(self, last_stride, dim, eps, device=None, dtype=None):
        super().__init__()
        self.layers = nn.Sequential(
            ops.Conv2d(3, 64, 3, stride=2, padding=1, device=device, dtype=dtype), nn.SiLU(),
            ops.Conv2d(64, 128, 3, stride=2, padding=1, device=device, dtype=dtype), nn.SiLU(),
            ops.Conv2d(128, 128, 3, stride=last_stride, padding=1, device=device, dtype=dtype), nn.SiLU(),
            ops.Conv2d(128, dim, 1, device=device, dtype=dtype))
        self.norm = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.norm(self.layers(x).flatten(2).transpose(1, 2))


class RelayFlashConditionEncoder(nn.Module):
    def __init__(self, in_channels, dim, eps, device=None, dtype=None):
        super().__init__()
        self.projection = ops.Conv2d(in_channels, dim, 1, device=device, dtype=dtype)
        self.norm = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.norm(self.projection(x).flatten(2).transpose(1, 2))


class RelayFlashBackbone(nn.Module):
    def __init__(self, depth, dim, num_heads, kv_groups, hidden_dim, eps, device=None, dtype=None):
        super().__init__()
        self.blocks = nn.ModuleList([RelayFlashBlock(dim, num_heads, kv_groups, hidden_dim, eps, device=device, dtype=dtype) for _ in range(depth)])


class RelayFlash(nn.Module):
    """Per-frame decoder conditioned on the keyframe latents on either side of the frame, VAE slot of RelayVSR."""

    comfy_has_chunked_io = True

    def __init__(self, upscale_factor, latent_channels=48, dim=512, depth=10, num_heads=8, kv_groups=2, hidden_dim=1408, head_dim=1024, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.latent_stride = 16 // upscale_factor
        self.lr_encoder = RelayFlashEncoder(self.latent_stride // 4, dim, eps, device=device, dtype=dtype)
        self.condition_encoder = RelayFlashConditionEncoder(latent_channels, dim, eps, device=device, dtype=dtype)
        self.video_type_embedding = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))
        self.condition_type_embedding = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))
        self.backbone = RelayFlashBackbone(depth, dim, num_heads, kv_groups, hidden_dim, eps, device=device, dtype=dtype)
        self.spatial_head = nn.Sequential(ops.Linear(dim, head_dim, device=device, dtype=dtype), nn.SiLU())
        self.residual_projection = ops.Linear(head_dim, 768, device=device, dtype=dtype)
        self.rope_embedder = EmbedND(dim=dim // num_heads, theta=10000.0, axes_dim=[16, 24, 24])

    def grid_rope(self, t, height, width, device):
        ids = torch.zeros((height, width, 3), device=device, dtype=torch.float32)
        ids[..., 0] = t
        ids[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None]
        ids[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None]
        return self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)

    def prepare_condition(self, latent, freqs):
        height, width = latent.shape[-2:]
        x = self.condition_encoder(latent) + comfy.ops.cast_to_input(self.condition_type_embedding, latent)
        out = []
        for block in self.backbone.blocks:
            x, kv = block.condition(x, freqs, height, width)
            out.append(kv)
        return out

    def decode_frame(self, lr, dtype, freqs, freqs_past, condition, history, detail):
        height, width = lr.shape[-2] // self.latent_stride, lr.shape[-1] // self.latent_stride
        x = self.lr_encoder(lr.to(dtype))
        x = x + comfy.ops.cast_to_input(self.video_type_embedding, x)
        next_history = []
        for block, cond, past in zip(self.backbone.blocks, condition, history):
            x, kv = block.video(x, freqs, freqs_past, height, width, cond, past)
            next_history.append(kv)
        residual = F.pixel_shuffle(self.residual_projection(self.spatial_head(x)).transpose(1, 2).reshape(lr.shape[0], 768, height, width), 16)
        if detail != 1.0:
            residual = residual * detail
        return F.interpolate(lr, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False) + residual, next_history

    def decode_output_shape(self, input_shape, lq_images=None, detail=1.0):
        if lq_images is None:
            raise ValueError("The RelayVSR Flash decoder needs the low-quality frames, decode with the VSR Decode node.")
        frames, height, width = lq_images.shape[:3]
        # the x2 and x4 decoders load from the same weight shapes, a latent of the other factor would decode garbage or fail on the copy
        if tuple(input_shape[2:]) != (len(relay_keyframes(frames)), *relay_latent_size(height, width, self.latent_stride)):
            raise ValueError(f"The latent {tuple(input_shape[2:])} doesn't fit the x{self.upscale_factor} RelayVSR Flash decoder and these {frames} {width}x{height} frames: use the RelayVSR model and Flash decoder of the same factor, and the same video in RelayVSR Conditioning and VSR Decode.")
        return (input_shape[0], 3, frames, height * self.upscale_factor, width * self.upscale_factor)

    def decode(self, x, lq_images=None, output_buffer=None, detail=1.0):
        # detail scales what the model adds to the bicubic upscale of each frame
        out = output_buffer
        if out is None:
            out = torch.empty(self.decode_output_shape(x.shape, lq_images), device=comfy.model_management.intermediate_device())
        b, _, _, height, width = x.shape
        frames, lq_height, lq_width = lq_images.shape[:3]
        depth = len(self.backbone.blocks)
        keyframes = relay_keyframes(frames)
        device = x.device
        freqs, freqs_past = self.grid_rope(0, height, width, device), self.grid_rope(-1, height, width, device)
        pbar = comfy.utils.ProgressBar(frames)
        # finished frames go to the output without the GPU waiting on each copy
        with comfy.model_management.HostWriter(device, out, (b, 3, *out.shape[-2:])) as writer, comfy.utils.model_trange(frames, desc="RelayVSR Flash decode", unit="frame") as console:
            start = self.prepare_condition(x[:, :, 0], freqs)
            for window, (first, last) in enumerate(zip(keyframes, keyframes[1:] or keyframes)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                end = self.prepare_condition(x[:, :, min(window + 1, len(keyframes) - 1)], freqs)
                condition = list(zip(start, end))
                history = [None] * depth
                # one upload per window, an upload from pageable memory waits for the whole stream
                lr_window = lq_images[first:last + 1].to(device=device)[..., :3].movedim(-1, 1) * 2.0 - 1.0
                lr_window = F.pad(lr_window, (0, width * self.latent_stride - lq_width, 0, height * self.latent_stride - lq_height), mode="replicate")
                for i, frame in enumerate(range(first, last + 1)):
                    decoded, history = self.decode_frame(lr_window[i:i + 1].expand(b, -1, -1, -1), x.dtype, freqs, freqs_past, condition, history, detail)
                    # the shared keyframe was already output by the previous window, here it only seeds the history
                    if window == 0 or frame > first:
                        # finished [0, 1] pixels on the GPU, the VAE's process_output is the identity
                        writer.write(out[:, :, frame], decoded[..., :out.shape[-2], :out.shape[-1]].add_(1.0).div_(2.0).clamp_(0.0, 1.0).to(out.dtype))
                        pbar.update(1)
                        console.update(1)
                start = end
        return out
