from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import einops
from einops import repeat

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
import torch.nn.functional as F

from comfy.ldm.flux.math import apply_rope, rope
from comfy.ldm.flux.layers import LastLayer

from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import comfy.model_management
import comfy.patcher_extension
import comfy.ldm.common_dit


# Copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = operations.Linear(in_channels * patch_size * patch_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, latent):
        latent = self.proj(latent)
        return latent


class PooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, pooled_embed):
        return self.pooled_embedder(pooled_embed)


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size, dtype=dtype, device=device, operations=operations)

    def forward(self, timesteps, wdtype):
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, transformer_options={}, preferred_attention=None):
    if isinstance(query, AttentionTensorContainer):
        query, key, value = query.take(), key.take(), value.take()
    heads = query.shape[2]
    query, key, value = AttentionTensorContainer(query.flatten(2)), AttentionTensorContainer(key.flatten(2)), AttentionTensorContainer(value.flatten(2))
    return optimized_attention(query, key, value, heads, preferred_attention=preferred_attention, transformer_options=transformer_options)


class HiDreamAttnProcessor_flashattn:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        transformer_options={},
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
            del query_t, key_t, value_t
        else:
            query = query_i
            key = key_i
            value = value_i

        del query_i, key_i, value_i
        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)
            del query_1, query_2, key_1, key_2

        query, key, value = AttentionTensorContainer(query), AttentionTensorContainer(key), AttentionTensorContainer(value)
        hidden_states = attention(query, key, value, transformer_options=transformer_options, preferred_attention=attn.comfy_attention)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states

class HiDreamAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor = None,
        out_dim: int = None,
        single: bool = False,
        dtype=None, device=None, operations=None
    ):
        # super(Attention, self).__init__()
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        linear_cls = operations.Linear
        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_k = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_v = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
        self.to_out = linear_cls(self.inner_dim, self.out_dim, dtype=dtype, device=device)
        self.q_rms_norm = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)
        self.k_rms_norm = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)

        if not single:
            self.to_q_t = linear_cls(query_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_k_t = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_v_t = linear_cls(self.inner_dim, self.inner_dim, dtype=dtype, device=device)
            self.to_out_t = linear_cls(self.inner_dim, self.out_dim, dtype=dtype, device=device)
            self.q_rms_norm_t = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)
            self.k_rms_norm_t = operations.RMSNorm(self.inner_dim, eps, dtype=dtype, device=device)

        self.processor = processor

    def forward(
        self,
        norm_image_tokens: torch.FloatTensor,
        image_tokens_masks: torch.FloatTensor = None,
        norm_text_tokens: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
        transformer_options={},
    ) -> torch.Tensor:
        return self.processor(
            self,
            image_tokens = norm_image_tokens,
            image_tokens_masks = image_tokens_masks,
            text_tokens = norm_text_tokens,
            rope = rope,
            transformer_options=transformer_options,
        )


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01, dtype=None, device=None, operations=None):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass
        # import torch.nn.init  as init
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, comfy.model_management.cast_to(self.weight, dtype=hidden_states.dtype, device=hidden_states.device), None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = None
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.shared_experts = FeedForwardSwiGLU(dim, hidden_dim // 2, dtype=dtype, device=device, operations=operations)
        self.experts = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim, dtype=dtype, device=device, operations=operations) for i in range(num_routed_experts)])
        self.gate = MoEGate(
            embed_dim = dim,
            num_routed_experts = num_routed_experts,
            num_activated_experts = num_activated_experts,
            dtype=dtype, device=device, operations=operations
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if True:  # self.training: # TODO: check which branch performs faster
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape).to(dtype=wtype)
            #y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear = operations.Linear(in_features=in_features, out_features=hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


class BlockType:
    TransformerBlock = 1
    SingleTransformerBlock = 2


class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device)
        )

        # 1. Attention
        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = True,
            dtype=dtype, device=device, operations=operations
        )

        # 3. Feed-forward
        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                dtype=dtype, device=device, operations=operations
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        transformer_options={},
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            rope = rope,
            transformer_options=transformer_options,
        )
        image_tokens = gate_msa_i * attn_output_i + image_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens.to(dtype=wtype))
        image_tokens = ff_output_i + image_tokens
        return image_tokens


class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 12 * dim, bias=True, dtype=dtype, device=device)
        )
        # nn.init.zeros_(self.adaLN_modulation[1].weight)
        # nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm1_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        self.norm1_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor = HiDreamAttnProcessor_flashattn(),
            single = False,
            dtype=dtype, device=device, operations=operations
        )

        # 3. Feed-forward
        self.norm3_i = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False, dtype=dtype, device=device)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim = dim,
                hidden_dim = 4 * dim,
                num_routed_experts = num_routed_experts,
                num_activated_experts = num_activated_experts,
                dtype=dtype, device=device, operations=operations
            )
        else:
            self.ff_i = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)
        self.norm3_t = operations.LayerNorm(dim, eps = 1e-06, elementwise_affine = False)
        self.ff_t = FeedForwardSwiGLU(dim = dim, hidden_dim = 4 * dim, dtype=dtype, device=device, operations=operations)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        transformer_options={},
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i, \
        shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = \
            self.adaLN_modulation(adaln_input)[:,None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm1_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_i) + shift_msa_i
        norm_text_tokens = self.norm1_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            rope = rope,
            transformer_options=transformer_options,
        )

        image_tokens = gate_msa_i * attn_output_i + image_tokens
        text_tokens = gate_msa_t * attn_output_t + text_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm3_i(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_i) + shift_mlp_i
        norm_text_tokens = self.norm3_t(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i * self.ff_i(norm_image_tokens)
        ff_output_t = gate_mlp_t * self.ff_t(norm_text_tokens)
        image_tokens = ff_output_i + image_tokens
        text_tokens = ff_output_t + text_tokens
        return image_tokens, text_tokens


class HiDreamImageBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        block_type: BlockType = BlockType.TransformerBlock,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        block_classes = {
            BlockType.TransformerBlock: HiDreamImageTransformerBlock,
            BlockType.SingleTransformerBlock: HiDreamImageSingleTransformerBlock,
        }
        self.block = block_classes[block_type](
            dim,
            num_attention_heads,
            attention_head_dim,
            num_routed_experts,
            num_activated_experts,
            dtype=dtype, device=device, operations=operations
        )

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: torch.FloatTensor = None,
        rope: torch.FloatTensor = None,
        transformer_options={},
    ) -> torch.FloatTensor:
        return self.block(
            image_tokens,
            image_tokens_masks,
            text_tokens,
            adaln_input,
            rope,
            transformer_options=transformer_options,
        )


class HiDreamImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
        image_model=None,
        dtype=None, device=None, operations=None
    ):
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers

        self.gradient_checkpointing = False

        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.llama_layers = llama_layers

        self.t_embedder = TimestepEmbed(self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.p_embedder = PooledEmbed(text_emb_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.x_embedder = PatchEmbed(
            patch_size = patch_size,
            in_channels = in_channels,
            out_channels = self.inner_dim,
            dtype=dtype, device=device, operations=operations
        )
        self.pe_embedder = EmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.num_attention_heads,
                    attention_head_dim = self.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.TransformerBlock,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamImageBlock(
                    dim = self.inner_dim,
                    num_attention_heads = self.num_attention_heads,
                    attention_head_dim = self.attention_head_dim,
                    num_routed_experts = num_routed_experts,
                    num_activated_experts = num_activated_experts,
                    block_type = BlockType.SingleTransformerBlock,
                    dtype=dtype, device=device, operations=operations
                )
                for i in range(self.num_single_layers)
            ]
        )

        self.final_layer = LastLayer(self.inner_dim, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations)

        caption_channels = [caption_channels[1], ] * (num_layers + num_single_layers) + [caption_channels[0], ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim, dtype=dtype, device=device, operations=operations))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

    def expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        return timesteps

    def unpatchify(self, x: torch.Tensor, img_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        x_arr = []
        for i, img_size in enumerate(img_sizes):
            pH, pW = img_size
            x_arr.append(
                einops.rearrange(x[i, :pH*pW].reshape(1, pH, pW, -1), 'B H W (p1 p2 C) -> B C (H p1) (W p2)',
                    p1=self.patch_size, p2=self.patch_size)
            )
        x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, x, max_seq, img_sizes=None):
        pz2 = self.patch_size * self.patch_size
        if isinstance(x, torch.Tensor):
            B = x.shape[0]
            device = x.device
            dtype = x.dtype
        else:
            B = len(x)
            device = x[0].device
            dtype = x[0].dtype
        x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)

        if img_sizes is not None:
            for i, img_size in enumerate(img_sizes):
                x_masks[i, 0:img_size[0] * img_size[1]] = 1
            x = einops.rearrange(x, 'B C S p -> B S (p C)', p=pz2)
        elif isinstance(x, torch.Tensor):
            pH, pW = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
            x = einops.rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
            img_sizes = [[pH, pW]] * B
            x_masks = None
        else:
            raise NotImplementedError
        return x, x_masks, img_sizes

    def forward(self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        encoder_hidden_states_llama3=None,
        image_cond=None,
        control = None,
        transformer_options = {},
    ):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, t, y, context, encoder_hidden_states_llama3, image_cond, control, transformer_options)

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        encoder_hidden_states_llama3=None,
        image_cond=None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        bs, c, h, w = x.shape
        if image_cond is not None:
            x = torch.cat([x, image_cond], dim=-1)
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        timesteps = t
        pooled_embeds = y
        T5_encoder_hidden_states = context

        img_sizes = None

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder

        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        hidden_states = self.x_embedder(hidden_states)

        # T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        rope = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens = hidden_states,
                image_tokens_masks = image_tokens_masks,
                text_tokens = cur_encoder_hidden_states,
                adaln_input = adaln_input,
                rope = rope,
                transformer_options=transformer_options,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
                transformer_options=transformer_options,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(output, img_sizes)
        return -output[:, :, :h, :w]
