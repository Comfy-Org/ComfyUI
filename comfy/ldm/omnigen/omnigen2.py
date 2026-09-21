# Original code: https://github.com/VectorSpaceLab/OmniGen2

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from comfy.ldm.lightricks.model import Timesteps
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import optimized_attention_masked
import comfy.model_management
import comfy.ldm.common_dit


def apply_rotary_emb(x, freqs_cis):
    if x.shape[1] == 0:
        return x

    return apply_rope1(x, freqs_cis)


def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x, inplace=True).mul_(y)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_channels, time_embed_dim, dtype=dtype, device=device)
        self.act = nn.SiLU()
        self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class LuminaRMSNormZero(nn.Module):
    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5, dtype=None, device=None, operations=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(min(embedding_dim, 1024), 4 * embedding_dim, dtype=dtype, device=device)
        self.norm = operations.RMSNorm(embedding_dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, elementwise_affine: bool = False, eps: float = 1e-6, out_dim: Optional[int] = None, dtype=None, device=None, operations=None):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_1 = operations.Linear(conditioning_embedding_dim, embedding_dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, eps, elementwise_affine, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(embedding_dim, out_dim, bias=True, dtype=dtype, device=device) if out_dim is not None else None

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        x = self.norm(x) * (1 + emb)[:, None, :]
        if self.linear_2 is not None:
            x = self.linear_2(x)
        return x


class LuminaFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, multiple_of: int = 256, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)
        self.linear_1 = operations.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(inner_dim, dim, bias=False, dtype=dtype, device=device)
        self.linear_3 = operations.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(swiglu(h1, h2))


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(self, hidden_size: int = 4096, text_feat_dim: int = 2048, frequency_embedding_size: int = 256, norm_eps: float = 1e-5, timestep_scale: float = 1.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0, scale=timestep_scale)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024), dtype=dtype, device=device, operations=operations)
        self.caption_embedder = nn.Sequential(
            operations.RMSNorm(text_feat_dim, eps=norm_eps, dtype=dtype, device=device),
            operations.Linear(text_feat_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, timestep: torch.Tensor, text_hidden_states: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(text_hidden_states)
        return time_embed, caption_embed


class Attention(nn.Module):
    def __init__(self, query_dim: int, dim_head: int, heads: int, kv_heads: int, eps: float = 1e-5, bias: bool = False, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = operations.Linear(query_dim, heads * dim_head, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, kv_heads * dim_head, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, kv_heads * dim_head, bias=bias, dtype=dtype, device=device)

        self.norm_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            operations.Linear(heads * dim_head, query_dim, bias=bias, dtype=dtype, device=device),
            nn.Dropout(0.0)
        )

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, image_rotary_emb: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, self.heads, self.dim_head)
        key = key.view(batch_size, -1, self.kv_heads, self.dim_head)
        value = value.view(batch_size, -1, self.kv_heads, self.dim_head)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        gqa_kwargs = {"enable_gqa": True} if self.kv_heads < self.heads else {}
        hidden_states = optimized_attention_masked(query, key, value, self.heads, attention_mask, skip_reshape=True, transformer_options=transformer_options, **gqa_kwargs)
        hidden_states = self.to_out[0](hidden_states)
        return hidden_states


class OmniGen2TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, num_kv_heads: int, multiple_of: int, ffn_dim_multiplier: float, norm_eps: float, modulation: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.modulation = modulation

        self.attn = Attention(
            query_dim=dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            dtype=dtype, device=device, operations=operations,
        )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            dtype=dtype, device=device, operations=operations
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        else:
            self.norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

        self.ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, image_rotary_emb: torch.Tensor, temb: Optional[torch.Tensor] = None, transformer_options={}) -> torch.Tensor:
        if self.modulation:
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
            attn_output = self.attn(norm_hidden_states, norm_hidden_states, attention_mask, image_rotary_emb, transformer_options=transformer_options)
            hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
            hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
        else:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn(norm_hidden_states, norm_hidden_states, attention_mask, image_rotary_emb, transformer_options=transformer_options)
            hidden_states = hidden_states + self.norm2(attn_output)
            mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
            hidden_states = hidden_states + self.ffn_norm2(mlp_output)
        return hidden_states


class OmniGen2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int] = (300, 512, 512), patch_size: int = 2):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size
        self.rope_embedder = EmbedND(dim=sum(axes_dim), theta=self.theta, axes_dim=axes_dim)

    def forward(self, batch_size, encoder_seq_len, l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, ref_img_sizes, img_sizes, device):
        p = self.patch_size

        seq_lengths = [cap_len + sum(ref_img_len) + img_len for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            position_ids[i, :cap_seq_len] = repeat(torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3")

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p

                    row_ids = repeat(torch.arange(ref_H_tokens, dtype=torch.int32, device=device), "h -> h w", w=ref_W_tokens).flatten()
                    col_ids = repeat(torch.arange(ref_W_tokens, dtype=torch.int32, device=device), "w -> h w", h=ref_H_tokens).flatten()
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p

            row_ids = repeat(torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens).flatten()

            position_ids[i, pe_shift_len: seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len: seq_len, 1] = row_ids
            position_ids[i, pe_shift_len: seq_len, 2] = col_ids

        freqs_cis = self.rope_embedder(position_ids).movedim(1, 2)

        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = encoder_seq_len
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        ref_img_freqs_cis_shape = list(freqs_cis.shape)
        ref_img_freqs_cis_shape[1] = max_ref_img_len
        ref_img_freqs_cis = torch.zeros(*ref_img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[i, cap_seq_len:cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len]

        return cap_freqs_cis, ref_img_freqs_cis, img_freqs_cis, freqs_cis, l_effective_cap_len, seq_lengths


class OmniGen2Transformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (32, 32, 32),
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.dtype = dtype

        self.rope_embedder = OmniGen2RotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = operations.Linear(patch_size * patch_size * in_channels, hidden_size, dtype=dtype, device=device)
        self.ref_image_patch_embedder = operations.Linear(patch_size * patch_size * in_channels, hidden_size, dtype=dtype, device=device)

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale, dtype=dtype, device=device, operations=operations
        )

        self.noise_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations
            ) for _ in range(num_refiner_layers)
        ])

        self.ref_image_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations
            ) for _ in range(num_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=False, dtype=dtype, device=device, operations=operations
            ) for _ in range(num_refiner_layers)
        ])

        self.layers = nn.ModuleList([
            OmniGen2TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations
            ) for _ in range(num_layers)
        ])

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            out_dim=patch_size * patch_size * self.out_channels, dtype=dtype, device=device, operations=operations
        )

        self.image_index_embedding = nn.Parameter(torch.empty(5, hidden_size, device=device, dtype=dtype))

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        batch_size = len(hidden_states)
        p = self.patch_size

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_image_hidden_states = list(map(lambda ref: comfy.ldm.common_dit.pad_to_patch_size(ref, (p, p)), ref_image_hidden_states))
            ref_img_sizes = [[(imgs.size(2), imgs.size(3)) if imgs is not None else None for imgs in ref_image_hidden_states]] * batch_size
            l_effective_ref_img_len = [[(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes] if _ref_img_sizes is not None else [0] for _ref_img_sizes in ref_img_sizes]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        flat_ref_img_hidden_states = None
        if ref_image_hidden_states is not None:
            imgs = []
            for ref_img in ref_image_hidden_states:
                B, C, H, W = ref_img.size()
                ref_img = rearrange(ref_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
                imgs.append(ref_img)
            flat_ref_img_hidden_states = torch.cat(imgs, dim=1)

        img = hidden_states
        B, C, H, W = img.size()
        flat_hidden_states = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

        return (
            flat_hidden_states, flat_ref_img_hidden_states,
            None, None,
            l_effective_ref_img_len, l_effective_img_len,
            ref_img_sizes, img_sizes,
        )

    def img_patch_embed_and_refine(self, hidden_states, ref_image_hidden_states, padded_img_mask, padded_ref_img_mask, noise_rotary_emb, ref_img_rotary_emb, l_effective_ref_img_len, l_effective_img_len, temb, transformer_options={}):
        batch_size = len(hidden_states)

        hidden_states = self.x_embedder(hidden_states)
        if ref_image_hidden_states is not None:
            ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)
            image_index_embedding = comfy.model_management.cast_to(self.image_index_embedding, dtype=hidden_states.dtype, device=hidden_states.device)

            for i in range(batch_size):
                shift = 0
                for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                    ref_image_hidden_states[i, shift:shift + ref_img_len, :] = ref_image_hidden_states[i, shift:shift + ref_img_len, :] + image_index_embedding[j]
                    shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb, transformer_options=transformer_options)

        if ref_image_hidden_states is not None:
            for layer in self.ref_image_refiner:
                ref_image_hidden_states = layer(ref_image_hidden_states, padded_ref_img_mask, ref_img_rotary_emb, temb, transformer_options=transformer_options)

            hidden_states = torch.cat([ref_image_hidden_states, hidden_states], dim=1)

        return hidden_states

    def forward(self, x, timesteps, context, num_tokens, ref_latents=None, attention_mask=None, transformer_options={}, **kwargs):
        B, C, H, W = x.shape
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        _, _, H_padded, W_padded = hidden_states.shape
        timestep = 1.0 - timesteps
        text_hidden_states = context
        text_attention_mask = attention_mask
        ref_image_hidden_states = ref_latents
        device = hidden_states.device

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

        (
            hidden_states, ref_image_hidden_states,
            img_mask, ref_img_mask,
            l_effective_ref_img_len, l_effective_img_len,
            ref_img_sizes, img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        (
            context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb,
            rotary_emb, encoder_seq_lengths, seq_lengths,
        ) = self.rope_embedder(
            hidden_states.shape[0], text_hidden_states.shape[1], [num_tokens] * text_hidden_states.shape[0],
            l_effective_ref_img_len, l_effective_img_len,
            ref_img_sizes, img_sizes, device,
        )

        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb, transformer_options=transformer_options)

        img_len = hidden_states.shape[1]
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states, ref_image_hidden_states,
            img_mask, ref_img_mask,
            noise_rotary_emb, ref_img_rotary_emb,
            l_effective_ref_img_len, l_effective_img_len,
            temb,
            transformer_options=transformer_options,
        )

        hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)
        attention_mask = None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb, transformer_options=transformer_options)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.patch_size
        output = rearrange(hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',  h=H_padded // p, w=W_padded// p, p1=p, p2=p)[:, :, :H, :W]

        return -output
