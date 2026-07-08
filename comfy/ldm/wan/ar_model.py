"""
CausalWanModel: Wan 2.1 backbone with KV-cached causal self-attention for
autoregressive (frame-by-frame) video generation via Causal Forcing.

Weight-compatible with the standard WanModel -- same layer names, same shapes.
The difference is purely in the forward pass: this model processes one temporal
block at a time and maintains a KV cache across blocks.

Reference: https://github.com/thu-ml/Causal-Forcing
"""

import torch
import torch.nn as nn

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.wan.model import (
    sinusoidal_embedding_1d,
    repeat_e,
    WanModel,
    WanAttentionBlock,
)
import comfy.ldm.common_dit
import comfy.model_management


class CausalWanSelfAttention(nn.Module):
    """Self-attention with KV cache support for autoregressive inference."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True,
                 eps=1e-6, operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        ops = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.q = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.k = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.v = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.o = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()
        self.norm_k = ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype) if qk_norm else nn.Identity()

    def forward(self, x, freqs, kv_cache=None, transformer_options={}):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
        k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v.view(b, s, n * d),
                heads=self.num_heads,
                transformer_options=transformer_options,
            )
        else:
            end = kv_cache["end"]
            new_end = end + s

            # Roped K and plain V go into cache
            kv_cache["k"][:, end:new_end] = k
            kv_cache["v"][:, end:new_end] = v
            kv_cache["end"] = new_end

            x = optimized_attention(
                q.view(b, s, n * d),
                kv_cache["k"][:, :new_end].view(b, new_end, n * d),
                kv_cache["v"][:, :new_end].view(b, new_end, n * d),
                heads=self.num_heads,
                transformer_options=transformer_options,
            )

        x = self.o(x)
        return x


class CausalWanAttentionBlock(WanAttentionBlock):
    """Transformer block with KV-cached self-attention and cross-attention caching."""

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads,
                         window_size, qk_norm, cross_attn_norm, eps,
                         operation_settings=operation_settings)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            operation_settings=operation_settings)

    def forward(self, x, e, freqs, context, context_img_len=257,
                kv_cache=None, crossattn_cache=None, transformer_options={}):
        if e.ndim < 4:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        else:
            e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)

        # Self-attention with optional KV cache
        x = x.contiguous()
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs, kv_cache=kv_cache, transformer_options=transformer_options)
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # Cross-attention with optional caching
        if crossattn_cache is not None and crossattn_cache.get("is_init"):
            q = self.cross_attn.norm_q(self.cross_attn.q(self.norm3(x)))
            x_ca = optimized_attention(
                q, crossattn_cache["k"], crossattn_cache["v"],
                heads=self.num_heads, transformer_options=transformer_options)
            x = x + self.cross_attn.o(x_ca)
        else:
            x = x + self.cross_attn(self.norm3(x), context, context_img_len=context_img_len, transformer_options=transformer_options)
            if crossattn_cache is not None:
                crossattn_cache["k"] = self.cross_attn.norm_k(self.cross_attn.k(context))
                crossattn_cache["v"] = self.cross_attn.v(context)
                crossattn_cache["is_init"] = True

        # FFN
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class CausalWanModel(WanModel):
    """
    Wan 2.1 diffusion backbone with causal KV-cache support.

    Same weight structure as WanModel -- loads identical state dicts.
    Adds forward_block() for frame-by-frame autoregressive inference.
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None):
        super().__init__(
            model_type=model_type, patch_size=patch_size, text_len=text_len,
            in_dim=in_dim, dim=dim, ffn_dim=ffn_dim, freq_dim=freq_dim,
            text_dim=text_dim, out_dim=out_dim, num_heads=num_heads,
            num_layers=num_layers, window_size=window_size, qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm, eps=eps, image_model=image_model,
            wan_attn_block_class=CausalWanAttentionBlock,
            device=device, dtype=dtype, operations=operations)

    def forward_block(self, x, timestep, context, start_frame,
                      kv_caches, crossattn_caches, clip_fea=None):
        """
        Forward one temporal block for autoregressive inference.

        Args:
            x: [B, C, block_frames, H, W] input latent for the current block
            timestep: [B, block_frames] per-frame timesteps
            context: [B, L, text_dim] raw text embeddings (pre-text_embedding)
            start_frame: temporal frame index for RoPE offset
            kv_caches: list of per-layer KV cache dicts
            crossattn_caches: list of per-layer cross-attention cache dicts
            clip_fea: optional CLIP features for I2V

        Returns:
            flow_pred: [B, C_out, block_frames, H, W] flow prediction
        """
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        bs, c, t, h, w = x.shape

        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # Per-frame time embedding
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype=x.dtype))
        e = e.reshape(timestep.shape[0], -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        # Text embedding (reuses crossattn_cache after first block)
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        # RoPE for current block's temporal position
        freqs = self.rope_encode(t, h, w, t_start=start_frame, device=x.device, dtype=x.dtype)

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, e=e0, freqs=freqs, context=context,
                      context_img_len=context_img_len,
                      kv_cache=kv_caches[i],
                      crossattn_cache=crossattn_caches[i])

        # Head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x[:, :, :t, :h, :w]

    def init_kv_caches(self, batch_size, max_seq_len, device, dtype):
        """Create fresh KV caches for all layers."""
        caches = []
        for _ in range(self.num_layers):
            caches.append({
                "k": torch.zeros(batch_size, max_seq_len, self.num_heads, self.head_dim, device=device, dtype=dtype),
                "v": torch.zeros(batch_size, max_seq_len, self.num_heads, self.head_dim, device=device, dtype=dtype),
                "end": 0,
            })
        return caches

    def init_crossattn_caches(self, batch_size, device, dtype):
        """Create fresh cross-attention caches for all layers."""
        caches = []
        for _ in range(self.num_layers):
            caches.append({"is_init": False})
        return caches

    def reset_kv_caches(self, kv_caches):
        """Reset KV caches to empty (reuse allocated memory)."""
        for cache in kv_caches:
            cache["end"] = 0

    def reset_crossattn_caches(self, crossattn_caches):
        """Reset cross-attention caches."""
        for cache in crossattn_caches:
            cache["is_init"] = False

    @property
    def head_dim(self):
        return self.dim // self.num_heads

    def forward(self, x, timestep, context, clip_fea=None, time_dim_concat=None, transformer_options={}, **kwargs):
        ar_state = transformer_options.get("ar_state")
        if ar_state is not None:
            bs = x.shape[0]
            block_frames = x.shape[2]
            t_per_frame = timestep.unsqueeze(1).expand(bs, block_frames)
            return self.forward_block(
                x=x, timestep=t_per_frame, context=context,
                start_frame=ar_state["start_frame"],
                kv_caches=ar_state["kv_caches"],
                crossattn_caches=ar_state["crossattn_caches"],
                clip_fea=clip_fea,
            )

        return super().forward(x, timestep, context, clip_fea=clip_fea,
                               time_dim_concat=time_dim_concat,
                               transformer_options=transformer_options, **kwargs)
