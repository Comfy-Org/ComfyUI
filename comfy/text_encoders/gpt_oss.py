"""GPT-OSS text encoder for Lens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import RMSNorm, apply_rope


@dataclass
class GptOss20BConfig:
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    sliding_window: int = 128
    original_max_position_embeddings: int = 4096
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_truncate: bool = False
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    layer_types: Optional[List[str]] = None
    moe_alpha: float = 1.702
    moe_limit: float = 7.0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 2 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


def _yarn_inv_freq(head_dim: int, base: float, factor: float, beta_fast: float, beta_slow: float,
    original_max_position_embeddings: int, truncate: bool, device=None) -> tuple[torch.Tensor, float]:
    """YARN inv_freq + attention scaling (matches transformers)."""
    dim = head_dim

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def find_correction_range() -> tuple[float, float]:
        low = find_correction_dim(beta_fast)
        high = find_correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_: float, max_: float, n: int) -> torch.Tensor:
        if min_ == max_:
            max_ += 0.001
        linear = (torch.arange(n, dtype=torch.float32, device=device) - min_) / (max_ - min_)
        return torch.clamp(linear, 0, 1)

    def get_mscale(scale: float) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    attention_scaling = get_mscale(factor)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range()
    extrap_factor = 1 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = inv_freq_interpolation * (1 - extrap_factor) + inv_freq_extrapolation * extrap_factor
    return inv_freq, attention_scaling


def _build_freqs_cis(inv_freq: torch.Tensor, attention_scaling: float, position_ids: torch.Tensor, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inv_freq_e = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    pos_e = position_ids[:, None, :].float()
    freqs = (inv_freq_e @ pos_e).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).to(dtype).unsqueeze(1)
    sin = (emb.sin() * attention_scaling).to(dtype).unsqueeze(1)
    sin_split = sin.shape[-1] // 2
    return cos, sin[..., :sin_split], -sin[..., sin_split:]


def _attention_with_sinks(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: torch.Tensor,
    attention_mask: Optional[torch.Tensor], num_heads: int, num_kv_groups: int) -> torch.Tensor:
    """Attention with per-head sinks.

    Sinks add a learned term to each row's softmax denominator but contribute
    nothing to the output. We fake this by appending one zero k/v position and
    putting the sink logit in the mask at that column.
    """

    B, _, S_q, D = q.shape
    H_kv = k.shape[1]
    S_kv = k.shape[-2]

    k = torch.cat([k, k.new_zeros(B, H_kv, 1, D)], dim=-2)
    v = torch.cat([v, v.new_zeros(B, H_kv, 1, D)], dim=-2)

    sinks_col = sinks.to(q.dtype).view(1, num_heads, 1, 1).expand(B, num_heads, S_q, 1)
    if attention_mask is not None:
        mask_left = attention_mask[..., :S_kv].expand(B, num_heads, S_q, S_kv)
    else:
        mask_left = q.new_zeros(B, num_heads, S_q, S_kv)
    mask = torch.cat([mask_left, sinks_col], dim=-1)

    op = optimized_attention_for_device(q.device, mask=True, small_input=True)
    return op(q, k, v, num_heads, mask=mask, skip_reshape=True, enable_gqa=True)


class GptOssAttention(nn.Module):
    def __init__(self, config: GptOss20BConfig, layer_idx: int, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        bias = config.attention_bias
        self.q_proj = ops.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias, device=device, dtype=dtype)
        self.sinks = nn.Parameter(torch.empty(self.num_heads, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor], freqs_cis) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cis)

        out = _attention_with_sinks(q, k, v, self.sinks, attention_mask, self.num_heads, self.num_kv_groups)
        return self.o_proj(out)


# Mixture of Experts

class GptOssTopKRouter(nn.Module):
    def __init__(self, config: GptOss20BConfig, device=None, dtype=None):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.weight = nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(config.num_local_experts, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = comfy.ops.cast_to_input(self.weight, hidden_states, copy=False)
        bias = comfy.ops.cast_to_input(self.bias, hidden_states, copy=False)
        logits = F.linear(hidden_states, weight, bias)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        # Softmax over top-k slice only
        scores = F.softmax(top_vals, dim=-1, dtype=top_vals.dtype)
        return scores, top_idx


class GptOssExperts(nn.Module):
    def __init__(self, config: GptOss20BConfig, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.alpha = config.moe_alpha
        self.limit = config.moe_limit

        E = self.num_experts
        H = self.hidden_size
        I = self.intermediate_size

        self.gate_up_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=2 * I, bias=True, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts=E, in_features=I, out_features=H, bias=True, device=device, dtype=dtype)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return torch.addcmul(glu, up, glu)

    def forward(self, hidden_states: torch.Tensor, router_indices: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        N = hidden_states.shape[0]
        top_k = router_indices.shape[-1]
        H = hidden_states.shape[-1]

        per_pair = torch.zeros((N * top_k, H), dtype=hidden_states.dtype, device=hidden_states.device)

        expert_mask = F.one_hot(router_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        with self.gate_up_proj.bank_resident(hidden_states) as gate_up_bank, \
             self.down_proj.bank_resident(hidden_states) as down_bank:
            for ei in expert_hit:
                expert_idx = int(ei.item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current = hidden_states[token_idx]

                gate_up = gate_up_bank.expert_linear(current, expert_idx)
                gated = self._apply_gate(gate_up)
                expert_out = down_bank.expert_linear(gated, expert_idx)

                weighted = expert_out * routing_weights[token_idx, top_k_pos, None]

                flat_idx = token_idx * top_k + top_k_pos
                per_pair[flat_idx] = weighted.to(per_pair.dtype)

        return per_pair.view(N, top_k, H).sum(dim=1)


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOss20BConfig, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.router = GptOssTopKRouter(config, device=device, dtype=dtype)
        self.experts = GptOssExperts(config, device=device, dtype=dtype, ops=ops)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        flat = hidden_states.reshape(-1, H)
        scores, idx = self.router(flat)
        out = self.experts(flat, idx, scores)
        return out.reshape(B, S, H)


# Decoder layer + model

class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOss20BConfig, layer_idx: int, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx, device=device, dtype=dtype, ops=ops)
        self.mlp = GptOssMLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.layer_type = config.layer_types[layer_idx]

    def forward(self, x: torch.Tensor, attention_masks: dict[str, Optional[torch.Tensor]], freqs_cis) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_masks[self.layer_type], freqs_cis)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


def _make_full_causal_mask(B: int, S: int, key_padding_mask: Optional[torch.Tensor], dtype, device):
    neg = torch.finfo(dtype).min
    mask = torch.full((S, S), neg, dtype=dtype, device=device).triu_(1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).contiguous()
    if key_padding_mask is not None:
        kp = key_padding_mask.to(dtype=dtype)
        kp = (1.0 - kp).reshape(B, 1, 1, S) * neg
        mask = mask + kp
    return mask


def _make_sliding_causal_mask(B: int, S: int, window: int, key_padding_mask: Optional[torch.Tensor], dtype, device):
    neg = torch.finfo(dtype).min
    i = torch.arange(S, device=device).view(-1, 1)
    j = torch.arange(S, device=device).view(1, -1)
    keep = (j <= i) & (j > i - window)
    mask = torch.where(keep, torch.zeros((), dtype=dtype, device=device), torch.full((), neg, dtype=dtype, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).contiguous()
    if key_padding_mask is not None:
        kp = key_padding_mask.to(dtype=dtype)
        kp = (1.0 - kp).reshape(B, 1, 1, S) * neg
        mask = mask + kp
    return mask


class GptOssModel(nn.Module):
    """GPT-OSS decoder with multi-layer hidden-state capture + early exit."""

    def __init__(self, config: GptOss20BConfig, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, i, device=device, dtype=dtype, ops=ops)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

        # Always build on CPU so the buffer survives meta-device construction.
        inv_freq, attn_scaling = _yarn_inv_freq(
            head_dim=config.head_dim,
            base=config.rope_theta,
            factor=config.rope_factor,
            beta_fast=config.rope_beta_fast,
            beta_slow=config.rope_beta_slow,
            original_max_position_embeddings=config.original_max_position_embeddings,
            truncate=config.rope_truncate,
            device=torch.device("cpu"),
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.rope_attention_scaling = float(attn_scaling)

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    def get_input_embeddings(self):
        return self.embed_tokens

    def _build_attention_masks(self, B: int, S: int, attention_mask: Optional[torch.Tensor], dtype: torch.dtype, device,
    ) -> dict[str, torch.Tensor]:
        full = _make_full_causal_mask(B, S, attention_mask, dtype, device)
        masks = {"full_attention": full}
        if any(t == "sliding_attention" for t in self.config.layer_types):
            masks["sliding_attention"] = _make_sliding_causal_mask(
                B, S, self.config.sliding_window, attention_mask, dtype, device
            )
        return masks

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,
                capture_layers: Optional[Sequence[int]] = None) -> dict[str, Any]:
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.dtype

        hidden_states = self.embed_tokens(input_ids, out_dtype=dtype)

        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        freqs_cis = _build_freqs_cis(self.rope_inv_freq.to(device=device), self.rope_attention_scaling, position_ids, dtype)

        attn_masks = self._build_attention_masks(B, S, attention_mask, dtype, device)

        capture_layers = list(capture_layers) if capture_layers else None
        if capture_layers:
            max_layer = max(capture_layers)
            wanted = {idx: pos for pos, idx in enumerate(capture_layers)}
            captured: List[Optional[torch.Tensor]] = [None] * len(capture_layers)
        else:
            max_layer = self.config.num_hidden_layers - 1
            wanted = None
            captured = None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attn_masks, freqs_cis)
            if wanted is not None and i in wanted:
                captured[wanted[i]] = hidden_states
            if i >= max_layer:
                break

        if captured is not None:
            return {"hidden_states": captured}
        return {"last_hidden_state": self.norm(hidden_states)}


# Lens chat-template constants (verbatim from the reference pipeline).
_LENS_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_LENS_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
LENS_TXT_OFFSET = 97
LENS_SELECTED_LAYERS = (5, 11, 17, 23)
LENS_MAX_TOKENS = 512


# The reference GPT-OSS Harmony template injects today's date here
_LENS_CHAT_DATE = "2026-05-23"


def _lens_render_chat(prompt: str) -> str:
    """Render the Lens prompt in GPT-OSS Harmony format."""
    return (
        f"<|start|>system<|message|>"
        f"You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\n"
        f"Current date: {_LENS_CHAT_DATE}\n\n"
        f"Reasoning: medium\n\n"
        f"# Valid channels: analysis, commentary, final. "
        f"Channel must be included for every message.<|end|>"
        f"<|start|>developer<|message|># Instructions\n\n"
        f"{_LENS_CHAT_SYSTEM}\n\n<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>"
        f"{_LENS_CHAT_ASSISTANT_THINKING}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>"
    )


# GPT-OSS-20B fixed token IDs (from the tokenizer's added-tokens table).
_LENS_PAD_TOKEN_ID = 199999  # <|endoftext|>


class _GptOssRawTokenizer:
    """Raw ``tokenizers.Tokenizer`` wrapper.

    The tokenizer JSON ships as a byte tensor inside the encoder checkpoint
    (``tokenizer_json`` key) rather than as a committed file. Extracted
    it in ``sd.py`` and passes it here via ``tokenizer_data``.
    """

    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        from tokenizers import Tokenizer
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            tokenizer_json_bytes = bytes(tokenizer_json_bytes.tolist())
        if tokenizer_json_bytes is None:
            raise ValueError(
                "Lens tokenizer requires the ``tokenizer_json`` byte tensor in the "
                "encoder state dict. Re-bundle the encoder via bundle_te.py so it "
                "embeds the tokenizer."
            )
        self.tokenizer = Tokenizer.from_str(tokenizer_json_bytes.decode("utf-8"))

    @classmethod
    def from_pretrained(cls, tokenizer_data, **kwargs):
        return cls(tokenizer_json_bytes=tokenizer_data, **kwargs)

    def __call__(self, text):
        return {"input_ids": self.tokenizer.encode(text, add_special_tokens=False).ids}

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.token_to_id(t) for t in tokens]

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, skip_special_tokens=kwargs.get("skip_special_tokens", False))


class LensGptOssTokenizer(sd1_clip.SDTokenizer):
    tokenizer_json_data = None

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json", None)
        self.tokenizer_json_data = tokenizer_json
        super().__init__(
            tokenizer_json,
            embedding_directory=embedding_directory,
            pad_with_end=False,
            embedding_size=2880,
            embedding_key="gpt_oss",
            tokenizer_class=_GptOssRawTokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_left=False,
            disable_weights=True,
            tokenizer_data=tokenizer_data,
        )
        self.pad_token_id = _LENS_PAD_TOKEN_ID

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        # Empty prompt -> empty list; encode_token_weights returns zeros (uncond).
        if not text or not text.strip():
            return [[]]
        rendered = _lens_render_chat(text)
        ids = self.tokenizer(rendered)["input_ids"]
        if len(ids) > LENS_MAX_TOKENS:
            ids = ids[:LENS_MAX_TOKENS]
        return [[(int(t), 1.0) for t in ids]]

    def state_dict(self):
        if self.tokenizer_json_data is not None:
            return {"tokenizer_json": self.tokenizer_json_data}
        return {}


class LensTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
            name="gpt_oss",
            tokenizer=LensGptOssTokenizer,
        )


class LensGptOssClipModel(nn.Module):
    """SDClipModel-shaped Lens GPT-OSS encoder (multi-layer feature extractor)."""

    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        super().__init__()
        model_options = dict(model_options or {})

        operations = model_options.get("custom_operations")
        if operations is None:
            quant_config = model_options.get("quantization_metadata") or {}
            operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)
        self.operations = operations

        cfg_overrides = model_options.get("gpt_oss_config", {})
        self.config = GptOss20BConfig(**cfg_overrides)
        self.selected_layers = tuple(model_options.get("selected_layers", LENS_SELECTED_LAYERS))
        self.txt_offset = int(model_options.get("txt_offset", LENS_TXT_OFFSET))

        self.transformer = GptOssModel(self.config, device=device, dtype=dtype, ops=operations)
        self.num_layers = self.config.num_hidden_layers
        self.dtype = dtype
        self.execution_device = None
        self._pad_token_id = _LENS_PAD_TOKEN_ID

    def set_clip_options(self, options):
        self.execution_device = options.get("execution_device", self.execution_device)

    def reset_clip_options(self):
        self.execution_device = None

    def _gather_tokens(self, token_weight_pairs):
        ids_list = [[int(t[0]) for t in batch] for batch in token_weight_pairs]
        pad_id = self._pad_token_id
        max_len = max(len(x) for x in ids_list)
        device = self.execution_device
        ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(ids_list), max_len), dtype=torch.long, device=device)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x, dtype=torch.long, device=device)
            mask[i, : len(x)] = 1
        return ids, mask

    def encode_token_weights(self, token_weight_pairs):
        # Empty negative: emit zero-length features + zero mask
        if all(len(batch) == 0 for batch in token_weight_pairs):
            device = self.execution_device
            B = len(token_weight_pairs)
            L = len(self.selected_layers)
            H = self.config.hidden_size
            flat = torch.zeros(B, 0, L * H, dtype=self.dtype, device=device)
            mask = torch.zeros(B, 0, dtype=torch.long, device=device)
            return flat, None, {"attention_mask": mask, "num_layers_stacked": L}

        input_ids, attn_mask = self._gather_tokens(token_weight_pairs)
        out = self.transformer(input_ids, attention_mask=attn_mask, capture_layers=self.selected_layers)
        layers = out["hidden_states"]  # list of L × [B, S, H]
        stacked = torch.stack(layers, dim=2)  # [B, S, L, H]

        offset = self.txt_offset
        if stacked.shape[1] > offset:
            stacked = stacked[:, offset:].contiguous()
            mask_trim = attn_mask[:, offset:]
        else:
            stacked = stacked[:, :0]
            mask_trim = attn_mask[:, :0]

        B, S, L, H = stacked.shape
        flat = stacked.reshape(B, S, L * H)
        extra = {"attention_mask": mask_trim, "num_layers_stacked": L}
        return flat, None, extra

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False, assign=True)


class LensTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None):
        super().__init__(device=device, dtype=dtype, name="gpt_oss", clip_model=LensGptOssClipModel, model_options=model_options or {})


def lens_te(dtype_llama=None, llama_quantization_metadata=None):
    class LensTEModel_(LensTEModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            mo = dict(model_options or {})
            if llama_quantization_metadata is not None:
                mo["quantization_metadata"] = llama_quantization_metadata
            if dtype is None and dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=mo)

    return LensTEModel_
