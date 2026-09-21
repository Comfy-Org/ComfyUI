# Boogu-Image-0.1 transformer
# Architecture is an OmniGen2 derivative (see comfy/ldm/omnigen/omnigen2.py) with an
# added dual-stream ("double_stream") stage before the single-stream layers, conditioned
# by a Qwen3-VL multimodal LLM. Reuses the OmniGen2/Lumina building blocks and the Flux
# RoPE core, the only new component is the double-stream block + the hybrid forward order.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

import comfy.ldm.common_dit
import comfy.ldm.omnigen.omnigen2
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.omnigen.omnigen2 import (
    OmniGen2RotaryPosEmbed,
    Lumina2CombinedTimestepCaptionEmbedding,
    LuminaRMSNormZero,
    LuminaLayerNormContinuous,
    LuminaFeedForward,
    Attention,
    OmniGen2TransformerBlock,
    apply_rotary_emb,
)

class BooguDoubleStreamProcessor(nn.Module):
    # Joint attention over [instruct ; img] with separate per-stream q/k/v and output projections.
    def __init__(self, dim, head_dim, heads, kv_heads, dtype=None, device=None, operations=None):
        super().__init__()
        query_dim = head_dim * heads
        kv_dim = head_dim * kv_heads

        self.img_to_q = operations.Linear(query_dim, query_dim, bias=False, dtype=dtype, device=device)
        self.img_to_k = operations.Linear(query_dim, kv_dim, bias=False, dtype=dtype, device=device)
        self.img_to_v = operations.Linear(query_dim, kv_dim, bias=False, dtype=dtype, device=device)

        self.instruct_to_q = operations.Linear(query_dim, query_dim, bias=False, dtype=dtype, device=device)
        self.instruct_to_k = operations.Linear(query_dim, kv_dim, bias=False, dtype=dtype, device=device)
        self.instruct_to_v = operations.Linear(query_dim, kv_dim, bias=False, dtype=dtype, device=device)

        self.instruct_out = operations.Linear(query_dim, query_dim, bias=False, dtype=dtype, device=device)
        self.img_out = operations.Linear(query_dim, query_dim, bias=False, dtype=dtype, device=device)

    def forward(self, attn, img_hidden_states, instruct_hidden_states, rotary_emb, attention_mask=None, transformer_options={}):
        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]

        img_q = self.img_to_q(img_hidden_states)
        img_k = self.img_to_k(img_hidden_states)
        img_v = self.img_to_v(img_hidden_states)

        instruct_q = self.instruct_to_q(instruct_hidden_states)
        instruct_k = self.instruct_to_k(instruct_hidden_states)
        instruct_v = self.instruct_to_v(instruct_hidden_states)

        # Concatenate instruction first, then image (matches reference processor order).
        query = torch.cat([instruct_q, img_q], dim=1)
        key = torch.cat([instruct_k, img_k], dim=1)
        value = torch.cat([instruct_v, img_v], dim=1)

        query = query.view(batch_size, -1, attn.heads, attn.dim_head)
        key = key.view(batch_size, -1, attn.kv_heads, attn.dim_head)
        value = value.view(batch_size, -1, attn.kv_heads, attn.dim_head)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        gqa_kwargs = {"enable_gqa": True} if attn.kv_heads < attn.heads else {}
        hidden_states = optimized_attention_masked(query, key, value, attn.heads, attention_mask, skip_reshape=True, transformer_options=transformer_options, **gqa_kwargs)

        # Split back to instruction/image, apply per-stream output projections, recombine.
        instruct_hidden_states = self.instruct_out(hidden_states[:, :L_instruct])
        img_hidden_states = self.img_out(hidden_states[:, L_instruct:])
        hidden_states = torch.cat([instruct_hidden_states, img_hidden_states], dim=1)

        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


class BooguJointAttention(nn.Module):
    # Holds the shared q/k RMSNorm + final output projection
    def __init__(self, dim, head_dim, heads, kv_heads, eps=1e-5, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = head_dim
        self.scale = head_dim ** -0.5

        self.norm_q = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.to_out = nn.Sequential(
            operations.Linear(heads * head_dim, dim, bias=False, dtype=dtype, device=device),
            nn.Dropout(0.0),
        )
        self.processor = BooguDoubleStreamProcessor(dim, head_dim, heads, kv_heads, dtype=dtype, device=device, operations=operations)

    def forward(self, img_hidden_states, instruct_hidden_states, rotary_emb, attention_mask=None, transformer_options={}):
        return self.processor(self, img_hidden_states, instruct_hidden_states, rotary_emb, attention_mask, transformer_options=transformer_options)


class BooguDoubleStreamBlock(nn.Module):
    # Dual-stream block: joint attention over [instruct ; img] + image self-attention, each stream with its own modulation/MLP.
    def __init__(self, dim, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, dtype=None, device=None, operations=None):
        super().__init__()
        head_dim = dim // num_attention_heads

        self.img_instruct_attn = BooguJointAttention(dim, head_dim, num_attention_heads, num_kv_heads, eps=1e-5, dtype=dtype, device=device, operations=operations)
        self.img_self_attn = Attention(
            query_dim=dim, dim_head=head_dim, heads=num_attention_heads, kv_heads=num_kv_heads,
            eps=1e-5, bias=False, dtype=dtype, device=device, operations=operations,
        )

        self.img_feed_forward = LuminaFeedForward(dim=dim, inner_dim=4 * dim, multiple_of=multiple_of, dtype=dtype, device=device, operations=operations)
        self.instruct_feed_forward = LuminaFeedForward(dim=dim, inner_dim=4 * dim, multiple_of=multiple_of, dtype=dtype, device=device, operations=operations)

        self.img_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        self.img_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        self.img_norm3 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        self.instruct_norm1 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)
        self.instruct_norm2 = LuminaRMSNormZero(embedding_dim=dim, norm_eps=norm_eps, dtype=dtype, device=device, operations=operations)

        self.img_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.img_self_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.img_ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.img_ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

        self.instruct_attn_norm = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.instruct_ffn_norm1 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)
        self.instruct_ffn_norm2 = operations.RMSNorm(dim, eps=norm_eps, dtype=dtype, device=device)

    def forward(self, img_hidden_states, instruct_hidden_states, joint_rotary_emb, img_rotary_emb, temb, joint_attention_mask=None, img_attention_mask=None, transformer_options={}):
        L_instruct = instruct_hidden_states.shape[1]

        img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(img_hidden_states, temb)
        img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
        img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

        instruct_norm1_out, instruct_gate_msa, instruct_scale_mlp, instruct_gate_mlp = self.instruct_norm1(instruct_hidden_states, temb)
        instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(instruct_hidden_states, temb)

        joint_attn_out = self.img_instruct_attn(img_norm1_out, instruct_norm1_out, joint_rotary_emb, joint_attention_mask, transformer_options=transformer_options)
        instruct_attn_out = joint_attn_out[:, :L_instruct]
        img_attn_out = joint_attn_out[:, L_instruct:]

        img_self_attn_out = self.img_self_attn(img_norm3_out, img_norm3_out, img_attention_mask, img_rotary_emb, transformer_options=transformer_options)

        img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(1).tanh() * self.img_attn_norm(img_attn_out)
        img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(1).tanh() * self.img_self_attn_norm(img_self_attn_out)
        img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
        img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
        img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(1).tanh() * self.img_ffn_norm2(img_mlp_out)

        instruct_hidden_states = instruct_hidden_states + instruct_gate_msa.unsqueeze(1).tanh() * self.instruct_attn_norm(instruct_attn_out)
        instruct_mlp_input = (1 + instruct_scale_mlp.unsqueeze(1)) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
        instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
        instruct_hidden_states = instruct_hidden_states + instruct_gate_mlp.unsqueeze(1).tanh() * self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


class BooguTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 3360,
        num_layers: int = 32,
        num_double_stream_layers: int = 8,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 28,
        num_kv_heads: int = 7,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),
        axes_lens: Tuple[int, int, int] = (2048, 1664, 1664),
        instruction_feat_dim: int = 4096,
        timestep_scale: float = 1000.0,
        image_model=None,
        device=None, dtype=None, operations=None,
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
            text_feat_dim=instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale, dtype=dtype, device=device, operations=operations
        )

        self.noise_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations)
            for _ in range(num_refiner_layers)
        ])

        self.ref_image_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations)
            for _ in range(num_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList([
            OmniGen2TransformerBlock(hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, modulation=False, dtype=dtype, device=device, operations=operations)
            for _ in range(num_refiner_layers)
        ])

        self.double_stream_layers = nn.ModuleList([
            BooguDoubleStreamBlock(hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, dtype=dtype, device=device, operations=operations)
            for _ in range(num_double_stream_layers)
        ])

        self.single_stream_layers = nn.ModuleList([
            OmniGen2TransformerBlock(hidden_size, num_attention_heads, num_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, modulation=True, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)
        ])

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            out_dim=patch_size * patch_size * self.out_channels, dtype=dtype, device=device, operations=operations
        )

        self.image_index_embedding = nn.Parameter(torch.empty(5, hidden_size, device=device, dtype=dtype))

    # Patchify/refine helpers are identical to OmniGen2; reuse via bound methods.
    flat_and_pad_to_seq = comfy.ldm.omnigen.omnigen2.OmniGen2Transformer2DModel.flat_and_pad_to_seq
    img_patch_embed_and_refine = comfy.ldm.omnigen.omnigen2.OmniGen2Transformer2DModel.img_patch_embed_and_refine

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

        # Double-stream stage: the image self-attention only sees the [ref ; noise] tokens,
        # which sit after the instruction tokens in the joint rope.
        L_instruct = text_hidden_states.shape[1]
        combined_img_rotary_emb = rotary_emb[:, L_instruct:]
        for layer in self.double_stream_layers:
            combined_img_hidden_states, text_hidden_states = layer(
                combined_img_hidden_states, text_hidden_states,
                rotary_emb, combined_img_rotary_emb, temb,
                joint_attention_mask=None, img_attention_mask=None,
                transformer_options=transformer_options,
            )

        hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)

        for layer in self.single_stream_layers:
            hidden_states = layer(hidden_states, None, rotary_emb, temb, transformer_options=transformer_options)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.patch_size
        output = rearrange(hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=H_padded // p, w=W_padded // p, p1=p, p2=p)[:, :, :H, :W]

        return -output
