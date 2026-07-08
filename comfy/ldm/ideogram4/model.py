"""
The Ideogram 4 transformer is a NextDiT/Lumina2-family single-stream model
consumes Qwen3-VL hidden-state features (concatenated from 13 layers -> 53248 dims)
packs ``[text tokens, image tokens]`` into one sequence with block-diagonal segment attention and 3D interleaved MRoPE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.patcher_extension
from comfy.ldm.lumina.model import FeedForward
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.text_encoders.llama import apply_rope, precompute_freqs_cis

# Per-token role indicators
SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
# Image grid coordinates are offset so they never collide with text positions
IMAGE_POSITION_OFFSET = 65536


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, eps=1e-5, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.qkv = operations.Linear(hidden_size, hidden_size * 3, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(self.head_dim, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.o = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x, attn_mask, freqs_cis, transformer_options={}):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        # (B, heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cis)

        out = optimized_attention_masked(q, k, v, self.num_heads, attn_mask, skip_reshape=True, transformer_options=transformer_options)
        return self.o(out)


class Ideogram4TransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, norm_eps, adaln_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5, dtype=dtype, device=device, operations=operations)
        self.feed_forward = FeedForward(
            dim=hidden_size, hidden_dim=intermediate_size, multiple_of=1, ffn_dim_multiplier=None,
            operation_settings={"operations": operations, "dtype": dtype, "device": device},
        )

        self.attention_norm1 = operations.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.ffn_norm1 = operations.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.attention_norm2 = operations.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)
        self.ffn_norm2 = operations.RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device)

        self.adaln_modulation = operations.Linear(adaln_dim, 4 * hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask, freqs_cis, transformer_options=transformer_options)
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(t, dim, scale=1e4):
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim, input_range=(0.0, 1.0), dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)
        self.mlp_out = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, dtype):
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, adaln_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm_final = operations.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)
        self.adaln_modulation = operations.Linear(adaln_dim, hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, x, c):
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
    """A single Ideogram 4 backbone operating on a packed token sequence."""

    def __init__(self, emb_dim, num_layers, num_heads, intermediate_size, adaln_dim,
                 in_channels, llm_features_dim, rope_theta, mrope_section, norm_eps,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.head_dim = emb_dim // num_heads
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)

        self.input_proj = operations.Linear(in_channels, emb_dim, bias=True, dtype=dtype, device=device)
        self.llm_cond_norm = operations.RMSNorm(llm_features_dim, eps=1e-6, elementwise_affine=True, dtype=dtype, device=device)
        self.llm_cond_proj = operations.Linear(llm_features_dim, emb_dim, bias=True, dtype=dtype, device=device)
        self.t_embedding = Ideogram4EmbedScalar(emb_dim, input_range=(0.0, 1.0), dtype=dtype, device=device, operations=operations)
        self.adaln_proj = operations.Linear(emb_dim, adaln_dim, bias=True, dtype=dtype, device=device)

        self.embed_image_indicator = operations.Embedding(2, emb_dim, dtype=dtype, device=device)

        self.layers = nn.ModuleList([
            Ideogram4TransformerBlock(emb_dim, intermediate_size, num_heads, norm_eps, adaln_dim,
                                      dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)
        ])

        self.final_layer = Ideogram4FinalLayer(emb_dim, in_channels, adaln_dim, dtype=dtype, device=device, operations=operations)

    def _backbone(self, llm_features, x, t, position_ids, attn_mask, indicator, transformer_options={}):
        indicator = indicator.to(torch.long)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        x = x * output_image_mask
        h = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t, dtype=x.dtype)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        # h is zero on the text rows (content lives only on image rows), add writes the text features in place
        if llm_features is not None:
            L_text = llm_features.shape[1]
            text_mask = (indicator[:, :L_text] == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
            llm = self.llm_cond_norm(llm_features * text_mask)
            llm = self.llm_cond_proj(llm) * text_mask
            h[:, :L_text] = h[:, :L_text] + llm

        h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long), out_dtype=h.dtype)

        # Qwen3-VL interleaved MRoPE; position_ids (B, L, 3) -> (3, L) (same across batch).
        freqs_cis = precompute_freqs_cis(
            self.head_dim, position_ids[0].transpose(0, 1), self.rope_theta,
            rope_dims=self.mrope_section, interleaved_mrope=True, device=position_ids.device,
        )

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=h.dtype).masked_fill_(~attn_mask, -torch.finfo(h.dtype).max)

        for layer in self.layers:
            h = layer(h, attn_mask, freqs_cis, adaln_input, transformer_options=transformer_options)

        return self.final_layer(h, adaln_input)


class Ideogram4Transformer2DModel(Ideogram4Transformer):
    """Ideogram 4 single-stream DiT.

    Runs a packed ``[text, image]`` sequence when text context is supplied, or an image-only sequence when ``context is None``.
    """

    def __init__(self, image_model=None, in_channels=128, num_layers=34, num_attention_heads=18, attention_head_dim=256, intermediate_size=12288,
                 adaln_dim=512, llm_features_dim=53248, rope_theta=5000000, mrope_section=(24, 20, 20), norm_eps=1e-5,
                 dtype=None, device=None, operations=None, **kwargs):
        emb_dim = num_attention_heads * attention_head_dim
        super().__init__(
            emb_dim=emb_dim, num_layers=num_layers, num_heads=num_attention_heads,
            intermediate_size=intermediate_size, adaln_dim=adaln_dim, in_channels=in_channels,
            llm_features_dim=llm_features_dim, rope_theta=rope_theta, mrope_section=mrope_section,
            norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = in_channels
        # 128-dim token = patch (2x2) * ae_channels (32).
        self.patch_size = 2
        self.ae_channels = in_channels // (self.patch_size * self.patch_size)

    def _img_to_tokens(self, x):
        B, C, gh, gw = x.shape
        x = x.view(B, self.ae_channels, self.patch_size, self.patch_size, gh, gw)
        x = x.permute(0, 4, 5, 2, 3, 1)  # (B, gh, gw, pi, pj, c)
        return x.reshape(B, gh * gw, C)

    def _tokens_to_img(self, tokens, gh, gw):
        B = tokens.shape[0]
        C = tokens.shape[-1]
        x = tokens.reshape(B, gh, gw, self.patch_size, self.patch_size, self.ae_channels)
        x = x.permute(0, 5, 3, 4, 1, 2)  # (B, c, pi, pj, gh, gw)
        return x.reshape(B, C, gh, gw)

    def _image_position_ids(self, gh, gw, device):
        h_idx = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
        w_idx = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        return torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET  # (L_img, 3)

    def _run_conditional(self, x_chunk, context_chunk, attn_mask_chunk, t_chunk, gh, gw, transformer_options):
        B = x_chunk.shape[0]
        device = x_chunk.device
        img_tokens = self._img_to_tokens(x_chunk)
        L_img = img_tokens.shape[1]
        L_text = context_chunk.shape[1]
        L = L_text + L_img
        latent_dim = img_tokens.shape[-1]

        x_full = torch.zeros(B, L, latent_dim, dtype=img_tokens.dtype, device=device)
        x_full[:, L_text:] = img_tokens

        text_pos = torch.arange(L_text, device=device).view(-1, 1).expand(L_text, 3)
        img_pos = self._image_position_ids(gh, gw, device)
        position_ids = torch.cat([text_pos, img_pos], dim=0).unsqueeze(0).expand(B, L, 3)

        indicator = torch.empty(B, L, dtype=torch.long, device=device)
        indicator[:, :L_text] = LLM_TOKEN_INDICATOR
        indicator[:, L_text:] = OUTPUT_IMAGE_INDICATOR

        attn_mask = None
        if attn_mask_chunk is not None:
            segment_ids = torch.ones(B, L, dtype=torch.long, device=device)
            pad = (attn_mask_chunk == 0)
            segment_ids[:, :L_text][pad] = SEQUENCE_PADDING_INDICATOR
            indicator[:, :L_text][pad] = 0
            # Block-diagonal mask from segment ids: (B, 1, L, L), True = attend.
            attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        out = self._backbone(context_chunk, x_full, t_chunk, position_ids, attn_mask, indicator,
                             transformer_options=transformer_options)
        return self._tokens_to_img(out[:, L_text:], gh, gw)

    def _run_image_only(self, x_chunk, t_chunk, gh, gw, transformer_options):
        B = x_chunk.shape[0]
        device = x_chunk.device
        img_tokens = self._img_to_tokens(x_chunk)
        L_img = img_tokens.shape[1]

        position_ids = self._image_position_ids(gh, gw, device).unsqueeze(0).expand(B, L_img, 3)
        indicator = torch.full((B, L_img), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=device)

        # Image-only sequence is a single segment -> no mask, full attention, no LLM context.
        out = self._backbone(None, img_tokens, t_chunk, position_ids, None, indicator, transformer_options=transformer_options)
        return self._tokens_to_img(out, gh, gw)

    def forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    def _forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={}, **kwargs):
        bs, c, gh, gw = x.shape

        timesteps = 1.0 - timesteps

        # unconditional pass
        if context is None:
            return -self._run_image_only(x, timesteps, gh, gw, transformer_options)

        return -self._run_conditional(x, context, attention_mask, timesteps, gh, gw, transformer_options)
