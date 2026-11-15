# Adapted from https://github.com/inclusionAI/Ming-3.0-tts/tree/main/ming_tiny
# (MIT, Copyright 2025 inclusionAI).
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

import comfy_kitchen
import comfy.model_prefetch
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.text_encoders.llama import MLP

def recurrent_kda(q, k, v, decay, beta, state):
    # KDA decays each key channel; Kitchen's GatedDeltaNet kernel has one decay per head.
    q, k = q.float(), k.float()
    q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).to(v.dtype).float()
    k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).to(v.dtype).float()
    q = q * q.shape[-1] ** -0.5
    output = []
    for i in range(q.shape[1]):
        state = state * decay[:, i, :, :, None]
        key = k[:, i, :, :, None]
        delta = v[:, i].float() - (key * state).sum(-2)
        state = state + key * (beta[:, i, :, None] * delta).unsqueeze(-2)
        output.append((q[:, i, :, :, None] * state).sum(-2))
    return torch.stack(output, dim=1).to(v.dtype), state


class KimiDeltaAttention(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.width = self.heads * self.head_dim
        self.conv_size = config.short_conv_kernel_size
        self.eps = config.rms_norm_eps
        self.safe_gate = config.kda_safe_gate
        self.lower_bound = config.kda_lower_bound
        for name in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
            setattr(self, name, operations.Linear(config.hidden_size, self.width, bias=False, device=device, dtype=dtype))
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            setattr(self, name, operations.Conv1d(self.width, self.width, self.conv_size, groups=self.width, bias=False, device=device, dtype=dtype))
        self.b_proj = operations.Linear(config.hidden_size, self.heads, bias=False, device=device, dtype=dtype)
        self.o_proj = operations.Linear(self.width, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.o_norm = operations.RMSNorm(self.head_dim, eps=self.eps, device=device, dtype=dtype)
        self.A_log = nn.Parameter(torch.empty(self.heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.width, device=device, dtype=torch.float32))

    def init_cache(self, batch, device, dtype):
        conv = torch.zeros((3, batch, self.width, self.conv_size - 1), device=device, dtype=dtype)
        state = torch.zeros((batch, self.heads, self.head_dim, self.head_dim), device=device, dtype=torch.float32)
        return (conv, state, torch.empty_like(conv), torch.empty_like(state))

    def forward(self, x, rope, positions, mask, cache):
        conv, state, next_conv, next_state = cache
        qkv = []
        for index, name in enumerate(("q", "k", "v")):
            projected = getattr(self, name + "_proj")(x)
            module = getattr(self, name + "_conv1d")
            with comfy.ops.CastBiasWeightContext(module, projected, offloadable=True) as (weight, bias):
                if x.shape[1] == 1 and comfy_kitchen.gated_delta_decode_is_available(x.device):
                    next_conv[index].copy_(conv[index])
                    value = comfy_kitchen.deltanet_conv_step(projected, next_conv[index], weight, bias).transpose(1, 2)
                else:
                    history = torch.cat((conv[index], projected.transpose(1, 2)), dim=-1)
                    next_conv[index].copy_(history[..., 1 - self.conv_size:])
                    value = F.silu(F.conv1d(history, weight, bias, groups=self.width)).transpose(1, 2)
            qkv.append(value.unflatten(-1, (self.heads, self.head_dim)))
        gate = self.f_proj(x).float().unflatten(-1, (self.heads, self.head_dim))
        beta = self.b_proj(x).float().sigmoid()
        gate = gate + comfy.ops.cast_to_input(self.dt_bias, gate).view(self.heads, self.head_dim)
        rate = comfy.ops.cast_to_input(self.A_log, gate).exp()[:, None]
        gate = self.lower_bound * (rate * gate).sigmoid() if self.safe_gate else -rate * F.softplus(gate)
        out, final_state = recurrent_kda(*qkv, gate.exp(), beta, state)
        next_state.copy_(final_state)
        g = self.g_proj(x).unflatten(-1, (self.heads, self.head_dim))
        out = (self.o_norm(out.float()) * g.float().sigmoid()).to(x.dtype)
        return self.o_proj(out.flatten(-2))

    def commit(self, cache):
        # Capture and its first replay must read the same previous recurrent state.
        cache[0].copy_(cache[2])
        cache[1].copy_(cache[3])


class MultiLatentAttention(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.heads = config.num_attention_heads
        self.rope_dim = config.qk_rope_head_dim
        self.nope_dim = config.qk_nope_head_dim
        self.value_dim = config.v_head_dim
        self.kv_rank = config.kv_lora_rank
        self.head_dim = self.nope_dim + self.rope_dim
        self.q_a_proj = operations.Linear(config.hidden_size, config.q_lora_rank, bias=False, device=device, dtype=dtype)
        self.q_a_layernorm = operations.RMSNorm(config.q_lora_rank, eps=1e-6, device=device, dtype=dtype)
        self.q_b_proj = operations.Linear(config.q_lora_rank, self.heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.kv_a_proj_with_mqa = operations.Linear(config.hidden_size, self.kv_rank + self.rope_dim, bias=False, device=device, dtype=dtype)
        self.kv_a_layernorm = operations.RMSNorm(self.kv_rank, eps=1e-6, device=device, dtype=dtype)
        self.kv_b_proj = operations.Linear(self.kv_rank, self.heads * (self.nope_dim + self.value_dim), bias=False, device=device, dtype=dtype)
        self.g_proj = operations.Linear(config.hidden_size, self.heads, bias=False, device=device, dtype=dtype)
        self.dense = operations.Linear(self.heads * self.value_dim, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.comfy_attention = ComfyAttention()

    def init_cache(self, batch, capacity, device, dtype):
        key = torch.zeros((batch, self.heads, capacity, self.head_dim), device=device, dtype=dtype)
        return key, torch.zeros_like(key)

    def forward(self, x, rope, positions, mask, cache):
        b, length, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(b, length, self.heads, self.head_dim).transpose(1, 2)
        q_pass, q_rot = q.split((self.nope_dim, self.rope_dim), dim=-1)
        compressed, k_rot = self.kv_a_proj_with_mqa(x).split((self.kv_rank, self.rope_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed)).view(b, length, self.heads, -1).transpose(1, 2)
        k_pass, value = kv.split((self.nope_dim, self.value_dim), dim=-1)
        q_rot, k_rot = apply_rope(q_rot, k_rot[:, None], rope)
        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot.expand(-1, self.heads, -1, -1)), dim=-1)
        # Equal head widths keep the selected ComfyUI attention backend's shape contract.
        value = F.pad(value, (0, self.head_dim - self.value_dim))
        cache[0].index_copy_(2, positions, k)
        cache[1].index_copy_(2, positions, value)
        # SDPA's capture-only workspace must stay outside the allocation replay.
        # CUDA graphs still capture and replay the attention kernels.
        with comfy.model_prefetch.pause_malloc_graph():
            out = optimized_attention(
                AttentionTensorContainer(q), AttentionTensorContainer(cache[0]), AttentionTensorContainer(cache[1]),
                self.heads, mask=mask, skip_reshape=True, preferred_attention=self.comfy_attention,
            ).view(b, length, self.heads, self.head_dim)[..., :self.value_dim]
        gate = self.g_proj(x).float().sigmoid().to(x.dtype)
        return self.dense((out * gate[..., None]).flatten(-2))

    def commit(self, cache):
        pass


class SparseMoE(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.groups = config.n_group
        self.top_groups = config.topk_group
        self.scale = config.routed_scaling_factor
        self.gate = operations.Linear(config.hidden_size, self.num_experts, bias=False, device=device, dtype=dtype)
        self.gate.register_buffer("expert_bias", torch.empty(self.num_experts, device=device, dtype=torch.float32))
        self.shared_experts = MLP(config, device=device, dtype=dtype, ops=operations, intermediate_size=config.moe_shared_expert_intermediate_size)
        self.experts = nn.Module()
        banks = comfy.ops.mixed_precision_ops({}, dtype)
        self.experts.gate_up_proj = banks.MoEExperts(self.num_experts, config.hidden_size, 2 * config.moe_intermediate_size, bias=False, device=device, dtype=dtype)
        self.experts.down_proj = banks.MoEExperts(self.num_experts, config.moe_intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def route(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = self.gate(x.float()).sigmoid()
        biased = scores + comfy.ops.cast_to_input(self.gate.expert_bias, scores)
        groups = biased.view(-1, self.groups, self.num_experts // self.groups).topk(2, dim=-1).values.sum(-1)
        selected = groups.topk(self.top_groups, dim=-1, sorted=False).indices
        keep = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, selected, True)
        keep = keep[:, :, None].expand(-1, -1, self.num_experts // self.groups).flatten(1)
        indices = biased.masked_fill(~keep, -torch.inf).topk(self.top_k, dim=-1).indices
        weight = scores.gather(1, indices)
        weight = weight / (weight.sum(-1, keepdim=True) + 1e-20) * self.scale
        return indices, weight

    @staticmethod
    def activation(x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    @staticmethod
    def selected_linear(module, x, indices):
        with module._cast_bank(x) as (bank, _):
            if isinstance(bank, QuantizedTensor):
                rows = indices[:, None] * module.out_features + torch.arange(module.out_features, device=x.device)
                data = bank._qdata[rows.flatten()]
                params = replace(bank._params, scale=bank._params.scale[rows.flatten()],
                                 orig_shape=(rows.numel(), module.in_features))
                bank = QuantizedTensor(data, bank._layout_cls, params).dequantize().view(-1, module.out_features, module.in_features)
            else:
                bank = bank[indices]
            return torch.bmm(bank, x[..., None]).squeeze(-1)

    def forward(self, x):
        flat = x.flatten(0, 1)
        indices, weight = self.route(flat)
        gate_up, down = self.experts.gate_up_proj, self.experts.down_proj
        if flat.shape[0] == 1:
            # Device-side expert selection keeps token decoding capture-safe.
            hidden = self.selected_linear(gate_up, flat.expand(self.top_k, -1), indices[0])
            hidden = self.activation(hidden)
            out = self.selected_linear(down, hidden, indices[0])
            out = (out.float() * weight[0, :, None]).sum(0, keepdim=True).to(x.dtype)
        else:
            order = indices.flatten().argsort()
            counts = torch.bincount(indices.flatten(), minlength=self.num_experts).tolist()
            sorted_x = flat[order // self.top_k]
            sorted_out = torch.empty_like(sorted_x)
            start = 0
            with gate_up.bank_resident(flat), down.bank_resident(flat):
                for expert, count in enumerate(counts):
                    if count:
                        hidden = self.activation(gate_up.expert_linear(sorted_x[start:start + count], expert))
                        sorted_out[start:start + count] = down.expert_linear(hidden, expert)
                        start += count
            restored = torch.empty_like(sorted_out)
            restored[order] = sorted_out
            # The reference weights and reduces expert outputs in FP32 before the final cast.
            out = (restored.view(*indices.shape, -1).float() * weight[..., None]).sum(1).to(x.dtype)
        return out.view_as(x) + self.shared_experts(x)


class TinyBlock(nn.Module):
    def __init__(self, config, index, device, dtype, operations):
        super().__init__()
        self.is_kda = (index + 1) % config.layer_group_size != 0
        attention = KimiDeltaAttention if self.is_kda else MultiLatentAttention
        self.attention = attention(config, device, dtype, operations)
        self.mlp = SparseMoE(config, device, dtype, operations) if index >= config.first_k_dense_replace else MLP(config, device=device, dtype=dtype, ops=operations)
        self.input_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = operations.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, rope, positions, mask, cache):
        x = x + self.attention(self.input_layernorm(x), rope, positions, mask, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyBackbone(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.config = SimpleNamespace(**config, mlp_activation="silu")
        self.word_embeddings = operations.Embedding(config["vocab_size"], config["hidden_size"], device=device, dtype=dtype)
        self.layers = nn.ModuleList([TinyBlock(self.config, i, device, dtype, operations) for i in range(config["num_hidden_layers"])])
        self.norm = operations.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"], device=device, dtype=dtype)

    @property
    def embed_tokens(self):
        return self.word_embeddings

    def init_kv_cache(self, batch, capacity, device, dtype):
        caches = [
            layer.attention.init_cache(batch, device, dtype) if layer.is_kda
            else layer.attention.init_cache(batch, capacity, device, dtype)
            for layer in self.layers
        ]
        return {"layers": caches, "length": 0, "capacity": capacity,
                "decode_mask": torch.empty((1, capacity), device=device, dtype=dtype)}

    def init_decode_buffers(self, batch, device, dtype):
        dim = self.config.qk_rope_head_dim
        return (
            torch.empty((batch, 1, self.config.hidden_size), device=device, dtype=dtype),
            torch.empty((1, 1, 1, dim // 2, 2, 2), device=device, dtype=dtype),
            torch.empty((1,), device=device, dtype=torch.long),
        )

    def forward(self, x, embeds, past_key_values, decode_buffers=None):
        x = embeds
        length, capacity = past_key_values["length"], past_key_values["capacity"]
        positions = torch.arange(length, length + x.shape[1], device=x.device)
        freq = 1 / self.config.rope_theta ** (torch.arange(0, self.config.qk_rope_head_dim, 2, device=x.device).float() / self.config.qk_rope_head_dim)
        phase = positions.float()[:, None] * freq
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        rope = torch.stack((cos, -sin, sin, cos), dim=-1).view(1, 1, x.shape[1], -1, 2, 2)
        if length and decode_buffers is not None:
            decode_buffers[0].copy_(x)
            decode_buffers[1].copy_(rope)
            decode_buffers[2].copy_(positions)
            x, rope, positions = decode_buffers
        mask = torch.arange(capacity, device=x.device)[None, :] > positions[:, None]
        mask = torch.zeros(mask.shape, device=x.device, dtype=x.dtype).masked_fill_(mask, -torch.inf)
        if length and decode_buffers is not None:
            past_key_values["decode_mask"].copy_(mask)
            mask = past_key_values["decode_mask"]
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {"prefetch_dynamic_vbars": True})
        for index, (layer, cache) in enumerate(zip(self.layers, past_key_values["layers"])):
            def core():
                x.copy_(layer(x, rope, positions, mask, cache))
            # KDA, MLA and dense/MoE blocks have different allocation sequences.
            scope = f"ming_tiny_{index}"
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, core=core, enable_graph=decode_buffers is not None and length > 0, malloc_scope=scope)
            layer.attention.commit(cache)
            comfy.model_prefetch.prefetch_queue_pop(None, x.device, None, malloc_scope=scope)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None)
        past_key_values["length"] += x.shape[1]
        return self.norm(x), None, past_key_values
