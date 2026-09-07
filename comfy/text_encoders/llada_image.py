# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# The LLaDA2 portion is adapted from inclusionAI/LLaDA-Image's pinned
# modeling_llada2uni_moe.py implementation for ComfyUI-native operations.

"""Native LLaDA-Image text, VQ, and query conditioning stack."""

from dataclasses import dataclass, fields

import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import comfy.model_management
import comfy.ops
from comfy import sd1_clip
from comfy.ldm.llada_image.conditioning import QueryFormer, SigVQ, TextProjection
from comfy.ldm.modules.attention import optimized_attention


@dataclass
class LLaDA2Config:
    vocab_size: int = 173568
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    rms_norm_eps: float = 1e-6
    rope_theta: float = 600000.0
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 16384
    pad_token_id: int = 156892
    mask_token_id: int = 156895
    end_of_image_token_id: int = 156902
    image_token_offset: int = 157184

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "LLaDA2 hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "LLaDA2 num_attention_heads must be divisible by num_key_value_heads"
            )
        rope_dim = int(self.head_dim * self.partial_rotary_factor)
        if rope_dim <= 0 or rope_dim % 2:
            raise ValueError("LLaDA2 partial rotary dimension must be positive and even")
        if self.num_experts % self.n_group:
            raise ValueError("LLaDA2 num_experts must be divisible by n_group")
        experts_per_group = self.num_experts // self.n_group
        if experts_per_group < 2:
            raise ValueError("LLaDA2 routing requires at least two experts per group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("LLaDA2 topk_group must be between one and n_group")
        if not 1 <= self.num_experts_per_tok <= self.topk_group * experts_per_group:
            raise ValueError(
                "LLaDA2 num_experts_per_tok exceeds the selected expert groups"
            )
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError(
                "LLaDA2 first_k_dense_replace must be within the layer count"
            )


def _known_config(config, keys):
    config = config or {}
    return {key: config[key] for key in keys if key in config}


def _require_config_values(component, config, expected):
    config = config or {}
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(
                f"Unsupported LLaDA-Image {component} configuration: "
                f"expected {key}={value!r}, got {config[key]!r}"
            )


def _rms_norm(hidden_states, norm):
    # LLaDA rounds normalized activations before applying the affine weight.
    with comfy.ops.CastBiasWeightContext(norm, hidden_states, offloadable=True) as (weight, _):
        return F.rms_norm(hidden_states, norm.normalized_shape, None, norm.eps) * weight


class LLaDA2MLP(nn.Module):
    def __init__(
        self, hidden_size, intermediate_size, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.gate_proj = operations.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = operations.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = operations.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype, device=device
        )

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class LLaDA2Gate(nn.Module):
    def __init__(self, config, dtype=None, device=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(
            torch.empty(
                config.num_experts, config.hidden_size, dtype=dtype, device=device
            )
        )
        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=dtype, device=device),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        weight = comfy.ops.cast_to_input(self.weight, hidden_states, copy=False)
        expert_bias = comfy.ops.cast_to_input(
            self.expert_bias, hidden_states, copy=False
        )
        logits = F.linear(hidden_states.float(), weight.float())
        scores = torch.sigmoid(logits.float()).to(logits.dtype)

        scores_for_routing = scores + expert_bias.float()
        group_scores = scores_for_routing.view(hidden_states.shape[0], self.n_group, -1)
        group_scores = group_scores.topk(2, dim=-1)[0].sum(dim=-1)
        group_indices = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_indices, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                hidden_states.shape[0], self.n_group, self.num_experts // self.n_group
            )
            .reshape(hidden_states.shape[0], -1)
        )
        masked_scores = scores_for_routing.masked_fill(
            ~score_mask.bool(), float("-inf")
        )
        topk_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)[1]

        topk_weight = torch.gather(scores, dim=1, index=topk_indices)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weight * self.routed_scaling_factor


class LLaDA2Experts(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = operations.MoEExperts(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.up_proj = operations.MoEExperts(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = operations.MoEExperts(
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, hidden_states, routing_weights, selected_experts):
        token_count = hidden_states.shape[0]
        top_k = selected_experts.shape[-1]
        intermediate = torch.empty(
            token_count * top_k,
            self.intermediate_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output = torch.zeros(
            token_count * top_k,
            self.hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )
        active_experts = (
            torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
        )
        routes = []
        for expert_tensor in active_experts:
            expert_index = int(expert_tensor.item())
            topk_position, token_index = torch.where(expert_mask[expert_index])
            routes.append(
                (
                    expert_index,
                    topk_position,
                    token_index,
                    token_index * top_k + topk_position,
                )
            )

        with self.gate_proj.bank_resident(hidden_states) as gate_bank:
            for expert_index, _, token_index, route_index in routes:
                current = hidden_states[token_index]
                gated = F.silu(gate_bank.expert_linear(current, expert_index))
                intermediate[route_index] = gated.to(intermediate.dtype)

        with self.up_proj.bank_resident(hidden_states) as up_bank:
            for expert_index, topk_position, token_index, route_index in routes:
                current = hidden_states[token_index]
                gated = intermediate[route_index]
                gated = gated * up_bank.expert_linear(current, expert_index)
                gated = gated * routing_weights[
                    token_index, topk_position, None
                ].to(gated)
                intermediate[route_index] = gated.to(intermediate.dtype)

        with self.down_proj.bank_resident(hidden_states) as down_bank:
            for expert_index, _, _, route_index in routes:
                gated = intermediate[route_index]
                expert_output = down_bank.expert_linear(gated, expert_index)
                output[route_index] = expert_output.to(output.dtype)
        return (
            output.view(token_count, top_k, self.hidden_size)
            .sum(dim=1, dtype=torch.float32)
            .to(output.dtype)
        )


class LLaDA2SparseMoeBlock(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.gate = LLaDA2Gate(config, dtype=dtype, device=device)
        self.experts = LLaDA2Experts(
            config, dtype=dtype, device=device, operations=operations
        )
        self.shared_experts = LLaDA2MLP(
            config.hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        selected_experts, routing_weights = self.gate(hidden_states)
        routed = self.experts(
            hidden_states.reshape(-1, hidden), routing_weights, selected_experts
        ).reshape(batch, sequence, hidden)
        return routed + self.shared_experts(hidden_states)


def _rotary_embeddings(position_ids, rope_dim, theta, dtype):
    device = position_ids.device
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    embedding = torch.cat((freqs, freqs), dim=-1)
    return embedding.cos().to(dtype).unsqueeze(1), embedding.sin().to(dtype).unsqueeze(
        1
    )


def _rotate_half(hidden_states):
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_partial_rope(query, key, cos, sin):
    rope_dim = cos.shape[-1]
    query_rotary, query_pass = query[..., :rope_dim], query[..., rope_dim:]
    key_rotary, key_pass = key[..., :rope_dim], key[..., rope_dim:]
    query_rotary = query_rotary * cos + _rotate_half(query_rotary) * sin
    key_rotary = key_rotary * cos + _rotate_half(key_rotary) * sin
    return torch.cat((query_rotary, query_pass), dim=-1), torch.cat(
        (key_rotary, key_pass), dim=-1
    )


def _repeat_kv(hidden_states, repeats):
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, head_dim)
        .reshape(batch, heads * repeats, sequence, head_dim)
    )


def _attention_bias(attention_mask, sequence, dtype, device):
    minimum = torch.finfo(dtype).min
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        key_valid = attention_mask.bool()[:, None, None, :]
        causal = torch.ones(sequence, sequence, dtype=torch.bool, device=device).tril()
        attention_mask = key_valid & causal[None, None]
    if attention_mask.dtype == torch.bool:
        return torch.where(
            attention_mask,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), minimum, dtype=dtype, device=device),
        )
    return attention_mask.to(device=device, dtype=dtype)


class LLaDA2Attention(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rope_theta = config.rope_theta
        self.query_key_value = operations.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.query_layernorm = operations.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device
        )
        self.key_layernorm = operations.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, dtype=dtype, device=device
        )
        self.dense = operations.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, hidden_states, attention_mask, position_ids, transformer_options=None
    ):
        batch, sequence, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states).view(
            batch,
            sequence,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query, key, value = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query = _rms_norm(query.transpose(1, 2), self.query_layernorm)
        key = _rms_norm(key.transpose(1, 2), self.key_layernorm)
        value = value.transpose(1, 2)
        cos, sin = _rotary_embeddings(
            position_ids, self.rope_dim, self.rope_theta, query.dtype
        )
        query, key = _apply_partial_rope(query, key, cos, sin)
        key_value_groups = self.num_heads // self.num_key_value_heads
        key = _repeat_kv(key, key_value_groups)
        value = _repeat_kv(value, key_value_groups)

        mask = _attention_bias(attention_mask, sequence, query.dtype, query.device)
        hidden_states = optimized_attention(
            query,
            key,
            value,
            self.num_heads,
            mask=mask,
            skip_reshape=True,
            transformer_options={}
            if transformer_options is None
            else transformer_options,
        )
        return self.dense(hidden_states)


class LLaDA2DecoderLayer(nn.Module):
    def __init__(self, config, layer_index, dtype=None, device=None, operations=None):
        super().__init__()
        self.attention = LLaDA2Attention(config, dtype, device, operations)
        if layer_index >= config.first_k_dense_replace:
            self.mlp = LLaDA2SparseMoeBlock(config, dtype, device, operations)
        else:
            self.mlp = LLaDA2MLP(
                config.hidden_size, config.intermediate_size, dtype, device, operations
            )
        self.input_layernorm = operations.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.post_attention_layernorm = operations.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, hidden_states, attention_mask, position_ids, transformer_options=None
    ):
        residual = hidden_states
        hidden_states = self.attention(
            _rms_norm(hidden_states, self.input_layernorm),
            attention_mask,
            position_ids,
            transformer_options,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(_rms_norm(hidden_states, self.post_attention_layernorm))


class LLaDA2LanguageModel(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        self.word_embeddings = operations.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                LLaDA2DecoderLayer(config, index, dtype, device, operations)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = operations.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
        )

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        transformer_options=None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        batch, sequence = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = (
                torch.arange(sequence, device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, position_ids, transformer_options
            )
        return _rms_norm(hidden_states, self.norm)


class LLaDA2Backbone(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.language_model = LLaDA2LanguageModel(config, dtype, device, operations)
        self.lm_head = operations.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class _LLaDARawTokenizer:
    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            if (
                tokenizer_json_bytes.dtype != torch.uint8
                or tokenizer_json_bytes.ndim != 1
            ):
                raise ValueError(
                    "LLaDA-Image tokenizer_json must be a one-dimensional uint8 tensor"
                )
            tokenizer_json_bytes = (
                tokenizer_json_bytes.detach().cpu().contiguous().numpy().tobytes()
            )
        if tokenizer_json_bytes is None:
            raise ValueError(
                "LLaDA-Image requires the tokenizer_json byte tensor embedded in the AIO checkpoint"
            )
        self.tokenizer = Tokenizer.from_str(tokenizer_json_bytes.decode("utf-8"))

    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        return cls(tokenizer_json_bytes=tokenizer_data, **kwargs)

    def __call__(self, text, add_special_tokens=True):
        return {
            "input_ids": self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            ).ids
        }

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(
            ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
        )

    def get_vocab(self):
        return self.tokenizer.get_vocab()


class LLaDAImageRawTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json")
        self.tokenizer_json_data = tokenizer_json
        super().__init__(
            tokenizer_json,
            embedding_directory=embedding_directory,
            pad_with_end=False,
            embedding_size=2048,
            embedding_key="llada2",
            tokenizer_class=_LLaDARawTokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=2048,
            min_length=1,
            pad_token=156892,
            disable_weights=True,
            tokenizer_data=tokenizer_data,
        )

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=True)["input_ids"]

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        if text is None or not text.strip():
            formatted = "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
        else:
            formatted = (
                f"<role>HUMAN</role> Generate an image: {text.strip()}\n"
                "<role>ASSISTANT</role>\n<IMAGE1>"
            )
        token_ids = self._encode(formatted)[:2048]
        return [[(int(token), 1.0) for token in token_ids]]

    def tokenize_vq(self, prompt, height, width):
        frontend_scale = max(max(height, width) / 512, 1.0)
        vq_height = int(height / frontend_scale) // 16
        vq_width = int(width / frontend_scale) // 16
        system_prompt = "You are a text-to-image generation assistant."
        conditional = (
            f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role>{prompt}"
            "<role>ASSISTANT</role>"
        )
        unconditional = (
            f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role><uncondition>"
            "<role>ASSISTANT</role>"
        )
        image_info = self._encode(
            f"<|image|><|reserved_token_{vq_height}|><|reserved_token_{vq_width}|><boi><|/image|>"
        )
        return (
            self._encode(conditional) + image_info[:-1],
            self._encode(unconditional) + image_info[:-1],
            vq_height,
            vq_width,
        )

    def state_dict(self):
        if self.tokenizer_json_data is None:
            return {}
        return {"tokenizer_json": self.tokenizer_json_data}


class LLaDAImageTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="llada2",
            tokenizer=LLaDAImageRawTokenizer,
        )

    def tokenize_vq(self, prompt, height, width):
        return self.llada2.tokenize_vq(prompt, height, width)


class LLaDAImageClipModel(nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        super().__init__()
        model_options = dict(model_options or {})
        operations = model_options.get("custom_operations")
        if operations is None or not hasattr(operations, "MoEExperts"):
            operations = comfy.ops.mixed_precision_ops(
                model_options.get("quantization_metadata") or {},
                dtype,
                full_precision_mm=True,
            )
        self.dtype = dtype
        self.execution_device = None
        _require_config_values(
            "text encoder",
            model_options.get("llada2_config"),
            {
                "attention_dropout": 0.0,
                "embedding_dropout": 0.0,
                "hidden_act": "silu",
                "moe_router_enable_expert_bias": True,
                "norm_topk_prob": True,
                "output_dropout": 0.0,
                "output_router_logits": False,
                "router_dtype": "fp32",
                "rope_scaling": {
                    "mrope_section": [16, 24, 24],
                    "rope_type": "default",
                    "type": "default",
                },
                "score_function": "sigmoid",
                "sliding_window": None,
                "tie_word_embeddings": False,
                "use_cache": False,
                "use_bias": False,
                "use_qk_norm": True,
                "use_qkv_bias": False,
            },
        )
        _require_config_values(
            "QueryFormer", model_options.get("queryformer_config"), {"dropout": 0.0}
        )
        _require_config_values(
            "text projection",
            model_options.get("text_projection_config"),
            {"attention_dropout": 0.0},
        )
        _require_config_values(
            "SigVQ",
            model_options.get("sigvq_config"),
            {"attention_dropout": 0.0},
        )
        self.config = LLaDA2Config(
            **_known_config(
                model_options.get("llada2_config"),
                {field.name for field in fields(LLaDA2Config)},
            )
        )
        self.model = LLaDA2Backbone(self.config, dtype, device, operations)
        self.queryformer = QueryFormer(
            **_known_config(
                model_options.get("queryformer_config"),
                {
                    "num_queries",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "intermediate_size",
                    "norm_eps",
                },
            ),
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.text_projection = TextProjection(
            **_known_config(
                model_options.get("text_projection_config"),
                {
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "projection_dim",
                    "norm_eps",
                },
            ),
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.sigvq = SigVQ(
            **_known_config(
                model_options.get("sigvq_config"),
                {
                    "image_size",
                    "patch_size",
                    "in_channels",
                    "hidden_size",
                    "intermediate_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "attention_bias",
                    "norm_eps",
                    "codebook_size",
                    "codebook_embed_dim",
                    "semantic_embed_dim",
                },
            ),
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.num_layers = self.config.num_hidden_layers

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def _batch_tokens(self, token_weight_pairs):
        token_ids = [[int(pair[0]) for pair in row] for row in token_weight_pairs]
        maximum = max(len(row) for row in token_ids)
        device = self.execution_device
        ids = torch.full(
            (len(token_ids), maximum),
            self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros(len(token_ids), maximum, dtype=torch.bool, device=device)
        for index, row in enumerate(token_ids):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[index, : len(row)] = True
        return ids, mask

    def encode_token_weights(self, token_weight_pairs):
        input_ids, attention_mask = self._batch_tokens(token_weight_pairs)
        inputs_embeds = self.model.get_input_embeddings()(
            input_ids, out_dtype=self.dtype
        )
        query_embeds = self.queryformer(inputs_embeds, attention_mask)
        text_length = inputs_embeds.shape[1]
        inputs_embeds = torch.cat(
            (inputs_embeds, query_embeds.to(inputs_embeds)), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    query_embeds.shape[1],
                    dtype=torch.bool,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)
        backbone_mask = (
            attention_mask[:, None, None, :]
            .expand(-1, 1, attention_mask.shape[1], -1)
            .clone()
        )
        backbone_mask[:, :, :text_length, text_length:] = False
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=backbone_mask,
            position_ids=position_ids,
        )
        prompt_embeds = self.text_projection(hidden_states)
        return prompt_embeds, None, {"attention_mask": attention_mask}

    def forward_logits(self, input_ids, attention_mask, position_ids, logits_to_keep=0):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.model.lm_head(hidden_states[:, -logits_to_keep:])

    @staticmethod
    def _num_transfer_tokens(block_length, steps):
        count, remainder = divmod(block_length, steps)
        return [count + (index < remainder) for index in range(steps)]

    def generate_vq_tokens(
        self, input_ids, unconditional_ids, image_token_count, cfg_scale=2.0
    ):
        block_length = 32
        steps = min(8, image_token_count)
        prompt_length = input_ids.shape[1]
        num_blocks = (
            prompt_length + image_token_count + block_length - 1
        ) // block_length
        total_length = num_blocks * block_length
        device = input_ids.device
        block_mask = torch.ones(
            num_blocks, num_blocks, dtype=torch.bool, device=device
        ).tril()
        full_attention_mask = block_mask.repeat_interleave(
            block_length, 0
        ).repeat_interleave(block_length, 1)[None, None]
        position_ids = torch.arange(total_length, device=device).unsqueeze(0)
        tokens = torch.full(
            (1, total_length),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        tokens[:, :prompt_length] = input_ids
        prefill_blocks = prompt_length // block_length
        schedule = self._num_transfer_tokens(block_length, steps)

        unconditional_ids = unconditional_ids.flatten()
        padding = prompt_length - len(unconditional_ids)
        if padding < 0:
            raise ValueError(
                "The unconditional LLaDA-Image prompt is longer than the prompt"
            )
        unconditional_input = torch.full(
            (1, prompt_length),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        unconditional_input[0, -len(unconditional_ids) :] = unconditional_ids
        unconditional_mask = full_attention_mask.clone()
        unconditional_mask[:, :, :, :padding] = False
        unconditional_positions = torch.cat(
            (
                torch.zeros(padding, dtype=torch.long, device=device),
                torch.arange(total_length - padding, device=device),
            )
        ).unsqueeze(0)

        for block_index in range(prefill_blocks, num_blocks):
            window_end = (block_index + 1) * block_length
            current = tokens[:, :window_end]
            current_mask = full_attention_mask[:, :, :window_end, :window_end]
            current_positions = position_ids[:, :window_end]
            for step_index in range(steps):
                active = current[:, -block_length:] == self.config.mask_token_id
                if not active.any():
                    break
                unconditional = current.clone()
                unconditional[:, :prompt_length] = unconditional_input
                combined_ids = torch.cat((current, unconditional), dim=0)
                combined_positions = torch.cat(
                    (current_positions, unconditional_positions[:, :window_end]), dim=0
                )
                combined_mask = torch.cat(
                    (
                        current_mask,
                        unconditional_mask[:, :, :window_end, :window_end],
                    ),
                    dim=0,
                )
                logits = self.forward_logits(
                    combined_ids, combined_mask, combined_positions,
                    logits_to_keep=block_length,
                )
                conditional_logits, unconditional_logits = logits.chunk(2, dim=0)
                active_logits = unconditional_logits[:, -block_length:] + cfg_scale * (
                    conditional_logits[:, -block_length:]
                    - unconditional_logits[:, -block_length:]
                )
                probabilities = F.softmax(active_logits, dim=-1)
                sampled = active_logits.argmax(dim=-1)
                confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                count = schedule[step_index]
                scores = torch.where(active, confidence, -torch.inf)
                selected = torch.zeros_like(sampled, dtype=torch.bool)
                high_confidence = scores[0] > 0.95
                if int(high_confidence.sum().item()) >= count:
                    selected[0] = high_confidence
                else:
                    _, indices = torch.topk(
                        scores[0], k=min(count, int(active.sum().item()))
                    )
                    selected[0, indices] = True
                current[:, -block_length:][selected] = sampled[selected]

                stop_positions = (
                    current[0, prompt_length:] == self.config.end_of_image_token_id
                ).nonzero(as_tuple=True)[0]
                if len(stop_positions) > 0:
                    stop_position = int(stop_positions[0].item()) + prompt_length
                    if (
                        current[0, prompt_length:stop_position]
                        != self.config.mask_token_id
                    ).all():
                        tokens[:, :window_end] = current
                        return (
                            tokens[:, prompt_length:stop_position]
                            - self.config.image_token_offset
                        )
            tokens[:, :window_end] = current
        return (
            tokens[:, prompt_length : prompt_length + image_token_count]
            - self.config.image_token_offset
        )

    def encode_sigvq(self, pixel_values=None, token_ids=None):
        return self.sigvq(pixel_values=pixel_values, token_ids=token_ids)

    def load_sd(self, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop("tokenizer_json", None)
        return self.load_state_dict(
            state_dict, strict=False, assign=getattr(self, "can_assign_sd", False)
        )


class LLaDAImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(
        self,
        device="cpu",
        dtype=None,
        model_options=None,
        llada2_config=None,
        queryformer_config=None,
        text_projection_config=None,
        sigvq_config=None,
    ):
        model_options = dict(model_options or {})
        checkpoint_configs = {
            "llada2_config": llada2_config,
            "queryformer_config": queryformer_config,
            "text_projection_config": text_projection_config,
            "sigvq_config": sigvq_config,
        }
        for key, value in checkpoint_configs.items():
            if value is not None:
                model_options.setdefault(key, value)
        super().__init__(
            device=device,
            dtype=dtype,
            model_options=model_options,
            name="llada2",
            clip_model=LLaDAImageClipModel,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = dict(state_dict)
        state_dict.pop("tokenizer_json", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def vq_memory_estimation(self, sequence_length):
        config = self.llada2.config
        sequence_length = (sequence_length + 31) // 32 * 32
        tokens = 2 * sequence_length
        dtype_size = comfy.model_management.dtype_size(self.llada2.dtype)
        # Only the active 32-token block is projected into vocabulary logits.
        logits = 2 * 2 * 32 * config.vocab_size * dtype_size
        routes = tokens * config.num_experts_per_tok * (
            (config.moe_intermediate_size + config.hidden_size) * dtype_size
            + config.num_experts * 8 + 32
        )
        hidden = tokens * (12 * config.hidden_size + 3 * config.intermediate_size) * dtype_size
        attention = 2 * tokens * config.num_attention_heads * sequence_length * 4
        sampling = 2 * 32 * config.vocab_size * (3 * dtype_size + 2 * 4)
        return logits + routes + hidden + attention + sampling

    def generate_vq_tokens(
        self, input_ids, unconditional_ids, image_token_count, cfg_scale=2.0
    ):
        return self.llada2.generate_vq_tokens(
            input_ids, unconditional_ids, image_token_count, cfg_scale
        )

    def encode_sigvq(self, pixel_values=None, token_ids=None):
        return self.llada2.encode_sigvq(pixel_values=pixel_values, token_ids=token_ids)


def te(dtype_llada=None):
    class LLaDAImageTEModel_(LLaDAImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
            if dtype_llada is not None:
                dtype = dtype_llada
            super().__init__(device=device, dtype=dtype, model_options=model_options, **kwargs)
    return LLaDAImageTEModel_
