import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import math

from comfy import sd1_clip
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.rmsnorm import rms_norm
from comfy.text_encoders.llama import RMSNorm, MLP, BaseLlama, BaseGenerate, _make_scaled_embedding


# Intentional minor divergences from transformers -reference implementation:
# - Embedding sqrt(hidden_size) scale applied as a Python scalar (full precision) instead of dtype-matched buffer tensor.
# - RMSNorm uses torch fused F.rms_norm, very slight numerical differences, but considerably faster
# - Input image and audio resizing/resampling slightly different numerically


GEMMA4_VISION_CONFIG = {"hidden_size": 768, "image_size": 896, "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 16, "patch_size": 16, "head_dim": 64, "rms_norm_eps": 1e-6, "position_embedding_size": 10240, "pooling_kernel_size": 3}
GEMMA4_VISION_31B_CONFIG = {"hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 16, "head_dim": 72, "rms_norm_eps": 1e-6, "position_embedding_size": 10240, "pooling_kernel_size": 3}
GEMMA4_AUDIO_CONFIG = {"hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 8, "intermediate_size": 4096, "conv_kernel_size": 5, "attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0, "attention_logit_cap": 50.0, "output_proj_dims": 1536, "rms_norm_eps": 1e-6, "residual_weight": 0.5}

@dataclass
class Gemma4Config:
    vocab_size: int = 262144
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 42
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma4"
    head_dim = 256
    global_head_dim = 512
    rms_norm_add = False
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [512, 512, 512, 512, 512, False]
    rope_scale = None
    partial_rotary_factor: float = 0.25
    final_norm: bool = True
    lm_head: bool = False
    final_logit_softcapping: float = 30.0
    hidden_size_per_layer_input: int = 256
    num_kv_shared_layers: int = 18
    use_double_wide_mlp: bool = False
    stop_tokens = [1, 50, 106]
    vision_config = GEMMA4_VISION_CONFIG
    audio_config = GEMMA4_AUDIO_CONFIG
    mm_tokens_per_image = 280

@dataclass
class Gemma4_E2B_Config(Gemma4Config):
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_key_value_heads: int = 1
    sliding_attention = [512, 512, 512, 512, False]
    num_kv_shared_layers: int = 20
    use_double_wide_mlp: bool = True

@dataclass
class Gemma4_31B_Config(Gemma4Config):
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    hidden_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    audio_config = None
    vision_config = GEMMA4_VISION_31B_CONFIG


# unfused RoPE as addcmul_ RoPE diverges from reference code
def _apply_rotary_pos_emb(x, freqs_cis):
    cos, sin = freqs_cis[0], freqs_cis[1]
    half = x.shape[-1] // 2
    out = x * cos
    out[..., :half] -= x[..., half:] * sin[..., :half]
    out[..., half:] += x[..., :half] * sin[..., half:]
    return out

class Gemma4Attention(nn.Module):
    def __init__(self, config, head_dim, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.inner_size = self.num_heads * head_dim

        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = None
        self.k_norm = None
        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        freqs_cis=None,
        past_key_value=None,
        sliding_window=None,
        shared_kv=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            xq = self.q_norm(xq)

        if shared_kv is not None:
            xk, xv = shared_kv
            # Apply RoPE to Q only (K already has RoPE from source layer)
            xq = _apply_rotary_pos_emb(xq, freqs_cis)
            present_key_value = None
            shareable_kv = None
        else:
            xk = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
            xv = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
            if self.k_norm is not None:
                xk = self.k_norm(xk)
            xv = rms_norm(xv)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            xq = _apply_rotary_pos_emb(xq, freqs_cis)
            xk = _apply_rotary_pos_emb(xk, freqs_cis)

            present_key_value = None
            if past_key_value is not None:
                cumulative_len = 0
                if len(past_key_value) > 0:
                    past_key, past_value, cumulative_len = past_key_value
                    xk = torch.cat((past_key, xk), dim=2)
                    xv = torch.cat((past_value, xv), dim=2)
                new_cumulative = cumulative_len + seq_length
                if sliding_window is not None and xk.shape[2] > sliding_window - 1:
                    cache_k = xk[:, :, -(sliding_window - 1):]
                    cache_v = xv[:, :, -(sliding_window - 1):]
                else:
                    cache_k = xk
                    cache_v = xv
                present_key_value = (cache_k, cache_v, new_cumulative)

            # KV for sharing: full xk/xv that SDPA sees (not evicted cache)
            shareable_kv = (xk, xv)

        # GQA: pass unexpanded KV with enable_gqa when no sliding mask,
        # expand heads when sliding mask is present
        # has to be done within SDPA itself to match the reference code, pre-scaling expansion causes numerical differences
        expand_kv = (self.num_heads != self.num_kv_heads and
                     sliding_window is not None and
                     xk.shape[2] >= sliding_window)
        if expand_kv:
            xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        gqa_kwargs = {} if expand_kv else ({"enable_gqa": True} if self.num_heads != self.num_kv_heads else {})
        output = optimized_attention_for_device(xq.device, mask=attention_mask is not None, small_input=True)(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, scale=1.0, **gqa_kwargs)

        return self.o_proj(output), present_key_value, shareable_kv


class TransformerBlockGemma4(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        head_dim = config.head_dim if self.sliding_attention else config.global_head_dim

        self.self_attn = Gemma4Attention(config, head_dim=head_dim, device=device, dtype=dtype, ops=ops)

        num_kv_shared = config.num_kv_shared_layers
        first_kv_shared = config.num_hidden_layers - num_kv_shared
        mlp_size = config.intermediate_size * 2 if config.use_double_wide_mlp and index >= first_kv_shared else None
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops, intermediate_size=mlp_size)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = ops.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False, device=device, dtype=dtype)
            self.per_layer_projection = ops.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False, device=device, dtype=dtype)
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
            self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))
        else:
            self.layer_scalar = None

    def forward(self, x, attention_mask=None, freqs_cis=None, past_key_value=None, per_layer_input=None, shared_kv=None):
        sliding_window = None
        if self.sliding_attention:
            sliding_window = self.sliding_attention
            # For prefill > sliding window, add sliding window restriction to the causal mask.
            if x.shape[1] > self.sliding_attention:
                sw_mask = torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
                sw_mask.masked_fill_(torch.ones_like(sw_mask, dtype=torch.bool).tril_(-self.sliding_attention), torch.finfo(x.dtype).min)
                attention_mask = attention_mask + sw_mask if attention_mask is not None else sw_mask
            freqs_cis = freqs_cis[1]
        else:
            freqs_cis = freqs_cis[0]

        residual = x
        x = self.input_layernorm(x)
        x, present_key_value, shareable_kv = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, freqs_cis=freqs_cis,
            past_key_value=past_key_value, sliding_window=sliding_window, shared_kv=shared_kv,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = x
            x = self.per_layer_input_gate(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            x = x * per_layer_input
            x = self.per_layer_projection(x)
            x = self.post_per_layer_input_norm(x)
            x = residual + x

        if self.layer_scalar is not None:
            x = x * self.layer_scalar

        return x, present_key_value, shareable_kv


class Gemma4Transformer(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config

        self.embed_tokens = _make_scaled_embedding(ops, config.vocab_size, config.hidden_size, config.hidden_size ** 0.5, device, dtype)

        self.layers = nn.ModuleList([
            TransformerBlockGemma4(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype) if config.final_norm else None

        # Precompute RoPE inv_freq on CPU to match reference code's exact value
        rope_angles_global = int(config.partial_rotary_factor * config.global_head_dim // 2)
        nope_global = config.global_head_dim // 2 - rope_angles_global
        global_inv = 1.0 / (config.rope_theta[0] ** (torch.arange(0, 2 * rope_angles_global, 2).float() / config.global_head_dim))
        if nope_global > 0:
            global_inv = torch.cat([global_inv, torch.zeros(nope_global)])
        self.register_buffer("_global_inv_freq", global_inv, persistent=False)

        sliding_inv = 1.0 / (config.rope_theta[1] ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("_sliding_inv_freq", sliding_inv, persistent=False)

        # Per-layer input mechanism
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = _make_scaled_embedding(ops, config.vocab_size, config.num_hidden_layers * self.hidden_size_per_layer_input, self.hidden_size_per_layer_input ** 0.5, device, dtype)
            self.per_layer_model_projection = ops.Linear(
                config.hidden_size, config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False, device=device, dtype=dtype)
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input, eps=config.rms_norm_eps,
                device=device, dtype=dtype)

    def get_past_len(self, past_key_values):
        for kv in past_key_values:
            if len(kv) >= 3:
                return kv[2]
        return 0

    def _freqs_from_inv(self, inv_freq, position_ids, device, dtype):
        """Compute cos/sin from stored inv_freq"""
        inv_exp = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_exp @ pos_exp).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)

    def compute_freqs_cis(self, position_ids, device, dtype=None):
        global_freqs = self._freqs_from_inv(self._global_inv_freq, position_ids, device, dtype)
        sliding_freqs = self._freqs_from_inv(self._sliding_inv_freq, position_ids, device, dtype)
        return [global_freqs, sliding_freqs]

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=None,
                past_key_values=None, input_ids=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.compute_freqs_cis(position_ids, x.device, dtype=x.dtype)

        mask = None
        min_val = torch.finfo(x.dtype).min
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), min_val)

        if seq_len > 1:
            causal_mask = torch.zeros(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device)
            causal_mask.masked_fill_(torch.ones_like(causal_mask, dtype=torch.bool).triu_(1), min_val)
            mask = mask + causal_mask if mask is not None else causal_mask

        # Per-layer inputs
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            num_layers = self.config.num_hidden_layers
            hpl = self.hidden_size_per_layer_input
            per_layer_proj = self.per_layer_model_projection(x) * (1.0 / (self.config.hidden_size ** 0.5))
            per_layer_proj = self.per_layer_projection_norm(per_layer_proj.reshape(*x.shape[:-1], num_layers, hpl))
            if input_ids is not None and input_ids.shape[1] == x.shape[1]:
                per_layer_emb = self.embed_tokens_per_layer(input_ids).reshape(*input_ids.shape, num_layers, hpl)
                per_layer_inputs = (per_layer_proj + per_layer_emb) * (0.5 ** 0.5)
            else:
                per_layer_inputs = per_layer_proj

        # KV sharing: later layers reuse KV from the last non-shared sliding/global layer
        num_kv_shared = self.config.num_kv_shared_layers
        first_kv_shared = self.config.num_hidden_layers - num_kv_shared if num_kv_shared > 0 else self.config.num_hidden_layers
        shared_sliding_kv = None  # KV from last non-shared sliding layer
        shared_global_kv = None   # KV from last non-shared global layer

        intermediate = None
        next_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None

            layer_kwargs = {}
            if per_layer_inputs is not None:
                layer_kwargs['per_layer_input'] = per_layer_inputs[:, :, i, :]

            is_sliding = hasattr(layer, 'sliding_attention') and layer.sliding_attention
            if i >= first_kv_shared and num_kv_shared > 0:
                shared = shared_sliding_kv if is_sliding else shared_global_kv
                if shared is not None:
                    layer_kwargs['shared_kv'] = shared

            x, current_kv, shareable_kv = layer(x=x, attention_mask=mask, freqs_cis=freqs_cis, past_key_value=past_kv, **layer_kwargs)

            next_key_values.append(current_kv if current_kv is not None else ())

            # Only track the last sliding/global before the sharing boundary
            if i < first_kv_shared and shareable_kv is not None:
                if is_sliding:
                    shared_sliding_kv = shareable_kv
                else:
                    shared_global_kv = shareable_kv

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        return x, intermediate


class Gemma4Base(BaseLlama, BaseGenerate, torch.nn.Module):
    """Common base for all Gemma4 variants: text model + vision."""
    def _init_model(self, config, dtype, device, operations):
        self.num_layers = config.num_hidden_layers
        self.model = Gemma4Transformer(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype
        self.multi_modal_projector = Gemma4MultiModalProjector(config, dtype=dtype, device=device, ops=operations)
        self.vision_model = Gemma4VisionEncoder(config.vision_config, dtype=dtype, device=device, ops=operations)

    def logits(self, x):
        logits = super().logits(x)
        cap = self.model.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        past_key_values = []
        for _ in range(self.model.config.num_hidden_layers):
            past_key_values.append(())
        return past_key_values

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = embed.pop("data").movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            max_soft_tokens = embed.get("max_soft_tokens", None)
            vision_out = self.vision_model(image.to(device, dtype=torch.float32), max_soft_tokens=max_soft_tokens)
            return self.multi_modal_projector(vision_out), None
        return None, None


class Gemma4AudioMixin:
    """Adds audio support to a Gemma4 model."""
    def _init_audio(self, config, dtype, device, operations):
        self.audio_model = Gemma4AudioEncoder(config.audio_config, dtype=dtype, device=device, ops=operations)
        self.audio_projector = Gemma4AudioProjector({"audio_output_proj_dims": config.audio_config["output_proj_dims"], "text_hidden_size": config.hidden_size, "rms_norm_eps": config.rms_norm_eps}, dtype=dtype, device=device, ops=operations)

    def preprocess_embed(self, embed, device):
        result, extra = super().preprocess_embed(embed, device)
        if result is not None:
            return result, extra
        if embed["type"] == "audio":
            audio = embed.pop("data").to(device, dtype=torch.float32)
            audio_mask = embed.pop("mask", None)
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            audio_out = self.audio_model(audio, audio_mask=audio_mask)
            return self.audio_projector(audio_out), None
        return None, None


# Vision Encoder

def _compute_vision_2d_rope(head_dim, pixel_position_ids, theta=100.0, device=None):
    """Compute 2D RoPE for vision: separate frequencies for x and y dimensions.

    Args:
        head_dim: dimension per head (e.g. 64)
        pixel_position_ids: [batch, num_patches, 2] with (x, y) coords
        theta: RoPE base frequency
    Returns:
        (cos, sin) each of shape [batch, num_patches, head_dim]
    """
    rotary_dim_per_axis = head_dim // 2
    freq_indices = torch.arange(0, rotary_dim_per_axis, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (freq_indices / rotary_dim_per_axis))

    all_cos, all_sin = [], []
    for i in range(2):  # x and y
        dim_positions = pixel_position_ids[:, :, i].float()  # [batch, num_patches]
        freqs = torch.einsum('bi,j->bij', dim_positions, inv_freq.to(device))  # [batch, num_patches, rotary_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, num_patches, rotary_dim]
        all_cos.append(emb.cos())
        all_sin.append(emb.sin())

    cos = torch.cat(all_cos, dim=-1).to(pixel_position_ids.device)  # [batch, num_patches, head_dim]
    sin = torch.cat(all_sin, dim=-1).to(pixel_position_ids.device)
    return cos, sin


def _apply_vision_2d_rope(x, freqs):
    """Apply 2D RoPE (multidimensional) to vision query/key states.

    Splits x and cos/sin into ndim=2 parts, applies 1D RoPE to each independently.

    x: [batch, heads, seq, head_dim]
    freqs: (cos, sin) each [batch, seq, head_dim]
    """
    cos = freqs[0].unsqueeze(1)  # [batch, 1, seq, head_dim]
    sin = freqs[1].unsqueeze(1)
    half = x.shape[-1] // 2
    a = _apply_rotary_pos_emb(x[..., :half], (cos[..., :half], sin[..., :half]))
    b = _apply_rotary_pos_emb(x[..., half:], (cos[..., half:], sin[..., half:]))
    return torch.cat([a, b], dim=-1)


class ClippedLinear(nn.Module):
    """Linear layer with activation clipping (from quantization-aware training).

    Stores input_max/min and output_max/min as buffers loaded from checkpoint.
    """
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None, ops=None):
        super().__init__()
        self.linear = ops.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('input_max', torch.tensor(float('inf'), device=device, dtype=dtype))
        self.register_buffer('input_min', torch.tensor(float('-inf'), device=device, dtype=dtype))
        self.register_buffer('output_max', torch.tensor(float('inf'), device=device, dtype=dtype))
        self.register_buffer('output_min', torch.tensor(float('-inf'), device=device, dtype=dtype))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        x = x.clamp(min=self.input_min, max=self.input_max)
        x = self.linear(x)
        return x.clamp_(min=self.output_min, max=self.output_max)


class Gemma4VisionMLP(nn.Module):
    """SwiGLU MLP matching gate_proj/up_proj/down_proj structure."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.gate_proj = ClippedLinear(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)
        self.up_proj = ClippedLinear(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)
        self.down_proj = ClippedLinear(intermediate_size, hidden_size, device=device, dtype=dtype, ops=ops)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)

        self.q_proj = ClippedLinear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, ops=ops)
        self.k_proj = ClippedLinear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, ops=ops)
        self.v_proj = ClippedLinear(self.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype, ops=ops)
        self.o_proj = ClippedLinear(self.num_heads * self.head_dim, self.hidden_size, device=device, dtype=dtype, ops=ops)

        self.q_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"], device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"], device=device, dtype=dtype)

    def forward(self, x, freqs, attention_mask=None):
        batch_size, seq_length, _ = x.shape

        xq = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        xv = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk).transpose(1, 2)
        xv = rms_norm(xv)

        xq = _apply_vision_2d_rope(xq, freqs)
        xk = _apply_vision_2d_rope(xk, freqs)

        xv = xv.to(xq.dtype).transpose(1, 2)

        output = optimized_attention_for_device(xq.device, mask=attention_mask is not None, small_input=True)(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, scale=1.0)
        return self.o_proj(output)


class Gemma4VisionLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.self_attn = Gemma4VisionAttention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = Gemma4VisionMLP(config, device=device, dtype=dtype, ops=ops)
        norm_kwargs = dict(eps=config["rms_norm_eps"], device=device, dtype=dtype)
        hidden = config["hidden_size"]
        self.input_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.post_attention_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(hidden, **norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(hidden, **norm_kwargs)

    def forward(self, x, freqs, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, freqs, attention_mask=attention_mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class Gemma4PatchEmbedder(nn.Module):
    """Patch embedding with learned 2D position embeddings via one-hot lookup."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]
        self.patch_size = patch_size
        self.position_embedding_size = config.get("position_embedding_size", 10240)

        self.input_proj = ops.Linear(3 * patch_size * patch_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.position_embedding_table = nn.Parameter(
            torch.empty(2, self.position_embedding_size, hidden_size, device=device, dtype=dtype)
        )

    def forward(self, patches, pixel_position_ids):
        """
        patches: [B, num_patches, 3*patch_size²] in [0,1] range (normalized to [-1,1] inside, matching HF)
        pixel_position_ids: [B, num_patches, 2] with (x,y) positions, (-1,-1) for padding
        """
        hidden_states = self.input_proj((2.0 * (patches - 0.5)).to(self.input_proj.weight.dtype))

        clamped_positions = pixel_position_ids.clamp(min=0)
        pos_table = comfy.model_management.cast_to_device(self.position_embedding_table, hidden_states.device, hidden_states.dtype)
        position_embeddings = pos_table[0][clamped_positions[..., 0]] + pos_table[1][clamped_positions[..., 1]]

        # Zero out position embeddings for padding patches (matching HF)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

        return hidden_states + position_embeddings


class Gemma4VisionEncoderLayers(nn.Module):
    """Wrapper to produce state dict keys as encoder.layers.X.*"""
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__()
        self.layers = nn.ModuleList([
            Gemma4VisionLayer(config, device=device, dtype=dtype, ops=ops)
            for _ in range(config["num_hidden_layers"])
        ])


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.patch_size = config["patch_size"]
        self.pooling_kernel_size = config.get("pooling_kernel_size", 3)
        self.root_hidden_size = self.hidden_size ** 0.5

        self.patch_embedder = Gemma4PatchEmbedder(config, device=device, dtype=dtype, ops=ops)
        self.encoder = Gemma4VisionEncoderLayers(config, dtype=dtype, device=device, ops=ops)

    def forward(self, pixel_values, max_soft_tokens=None):
        """
        pixel_values: [B, C, H, W] in [0,1] range
        max_soft_tokens: if provided, pad to max_soft_tokens * k² total patches
        """
        batch_size, _, height, width = pixel_values.shape
        ps = self.patch_size
        k = self.pooling_kernel_size
        patches_h, patches_w = height // ps, width // ps
        num_patches = patches_h * patches_w
        output_length = max_soft_tokens if max_soft_tokens is not None else num_patches // (k * k)
        n_padding = output_length * k * k - num_patches

        # Patchify and build position grid
        patches = pixel_values.reshape(batch_size, -1, patches_h, ps, patches_w, ps)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, num_patches, -1)
        grid_y, grid_x = torch.meshgrid(torch.arange(patches_h, device=pixel_values.device), torch.arange(patches_w, device=pixel_values.device), indexing='ij')
        position_ids = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        # Append zero-pixel padding with (-1,-1) positions
        if n_padding > 0:
            patches = torch.cat([patches, patches.new_zeros(batch_size, n_padding, patches.shape[-1])], dim=1)
            position_ids = torch.cat([position_ids, position_ids.new_full((batch_size, n_padding, 2), -1)], dim=1)

        padding = (position_ids == -1).all(dim=-1)

        # Embed, encode, pool
        x = self.patch_embedder(patches, position_ids)
        freqs = _compute_vision_2d_rope(self.head_dim, position_ids, device=pixel_values.device)
        freqs = tuple(t.to(x.dtype) for t in freqs)
        if n_padding > 0:
            mask = padding.unsqueeze(1).unsqueeze(2).expand(-1, 1, position_ids.shape[1], -1)
            mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill_(mask, torch.finfo(x.dtype).min)
        else:
            mask = None

        for layer in self.encoder.layers:
            x = layer(x, freqs, attention_mask=mask)

        if n_padding > 0:
            x = x.masked_fill(padding.unsqueeze(-1), 0.0)

        # Average pool by spatial position
        clamped = position_ids.clamp(min=0)
        max_x = clamped[:, :, 0].max(dim=-1, keepdim=True)[0] + 1
        ki = torch.div(clamped, k, rounding_mode="floor")
        ki = ki[:, :, 0] + (max_x // k) * ki[:, :, 1]
        weights = torch.nn.functional.one_hot(ki.long(), output_length).float() / (k * k)
        x = (weights.transpose(1, 2) @ x.float()).to(x.dtype)

        # Strip empty output tokens
        valid_out = ~((weights == 0).all(dim=1))
        if valid_out.any() and not valid_out.all():
            x = x[:, valid_out[0]] if batch_size > 1 else x[valid_out].unsqueeze(0)

        return x * self.root_hidden_size


class Gemma4RMSNormProjector(nn.Module):
    """Shared projector: parameterless RMSNorm → linear. Used for both vision and audio."""
    def __init__(self, in_dim, out_dim, dtype=None, device=None, ops=None):
        super().__init__()
        self.embedding_projection = ops.Linear(in_dim, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.embedding_projection(rms_norm(x))


class Gemma4MultiModalProjector(Gemma4RMSNormProjector):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__(config.vision_config["hidden_size"], config.hidden_size, dtype=dtype, device=device, ops=ops)


# Audio Encoder

class Gemma4AudioConvSubsampler(nn.Module):
    """2D convolution subsampling for audio features"""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        eps = config["rms_norm_eps"]
        self.layer0 = nn.ModuleDict({
            'conv': ops.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False, device=device, dtype=dtype),
            'norm': ops.LayerNorm(128, eps=eps, elementwise_affine=True, bias=False, device=device, dtype=dtype),
        })
        self.layer1 = nn.ModuleDict({
            'conv': ops.Conv2d(128, 32, kernel_size=3, stride=2, padding=1, bias=False, device=device, dtype=dtype),
            'norm': ops.LayerNorm(32, eps=eps, elementwise_affine=True, bias=False, device=device, dtype=dtype),
        })
        # proj_input_dim = (128 // 4) * 32 = 1024
        self.input_proj_linear = ops.Linear(1024, config["hidden_size"], bias=False, device=device, dtype=dtype)

    def _conv_layer(self, x, layer, mask):
        if mask is not None:
            x = x * mask[:, None, :, None].to(x.device)
        x = layer['conv'](x.to(layer['conv'].weight.dtype))
        x = torch.relu(layer['norm'](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        if mask is not None:
            mask = mask[:, ::2]
        return x, mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x, mask = self._conv_layer(x, self.layer0, mask)
        x, mask = self._conv_layer(x, self.layer1, mask)
        batch_size, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        return self.input_proj_linear(x), mask


class Gemma4AudioFeedForward(nn.Module):
    """Conformer feed-forward with residual scaling."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        self.pre_layer_norm = RMSNorm(hidden_size, eps=config["rms_norm_eps"], device=device, dtype=dtype)
        self.ffw_layer_1 = ClippedLinear(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)
        self.ffw_layer_2 = ClippedLinear(intermediate_size, hidden_size, device=device, dtype=dtype, ops=ops)
        self.post_layer_norm = RMSNorm(hidden_size, eps=config["rms_norm_eps"], device=device, dtype=dtype)
        self.post_layer_scale = config.get("residual_weight", 0.5)

    def forward(self, x):
        residual = x
        x = self.pre_layer_norm(x)
        x = torch.nn.functional.silu(self.ffw_layer_1(x))
        x = self.ffw_layer_2(x)
        x = self.post_layer_norm(x)
        x = x * self.post_layer_scale
        return x + residual


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for audio attention."""
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        context_left = config.get("attention_context_left", 13)
        context_right = config.get("attention_context_right", 0)
        self.chunk_size = config.get("attention_chunk_size", 12)
        self.context_size = self.chunk_size + context_left - 1 + context_right

        num_timescales = hidden_size // 2
        log_inc = math.log(10000.0) / max(num_timescales - 1, 1)
        inv_timescales = torch.exp(torch.arange(num_timescales) * -log_inc).to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    def forward(self, hidden_states):
        positions = torch.arange(self.chunk_size, -1, -1, device=hidden_states.device).unsqueeze(-1)
        scaled = positions * self.inv_timescales.to(device=hidden_states.device)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    """Chunked block attention with relative position bias and softcap."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.chunk_size = config.get("attention_chunk_size", 12)
        self.max_past_horizon = config.get("attention_context_left", 13) - 1
        self.max_future_horizon = config.get("attention_context_right", 0)
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_scale = (self.head_dim ** -0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.register_buffer("softcap", torch.tensor(config.get("attention_logit_cap", 50.0), dtype=dtype), persistent=False)

        self.q_proj = ClippedLinear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, ops=ops)
        self.k_proj = ClippedLinear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, ops=ops)
        self.v_proj = ClippedLinear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, ops=ops)
        self.post = ClippedLinear(self.hidden_size, self.hidden_size, device=device, dtype=dtype, ops=ops)
        self.per_dim_scale = nn.Parameter(torch.empty(self.head_dim, device=device, dtype=dtype))
        self.relative_k_proj = ops.Linear(self.hidden_size, self.hidden_size, bias=False, device=device, dtype=dtype)

    def _convert_to_block(self, x):
        B, S, H, D = x.shape
        num_blocks = (S + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - S
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
        return x.reshape(B, num_blocks, self.chunk_size, H, D).contiguous()

    def _extract_block_context(self, x):
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1))
        x = x.unfold(1, self.context_size, self.chunk_size)
        return torch.movedim(x, -1, 2).contiguous()

    def _rel_shift(self, x):
        B, H, NB, BS, PL = x.shape
        CS = self.context_size
        x = torch.nn.functional.pad(x, (0, CS + 1 - PL))
        x = x.view(B, H, NB, BS * (CS + 1))
        x = x[..., :BS * CS]
        return x.view(B, H, NB, BS, CS)

    def _build_blocked_mask(self, seq_len, num_blocks, device, audio_mask=None):
        """Build 5D boolean blocked attention mask (True=attend, False=mask)"""
        q = torch.arange(seq_len, device=device)
        dist = q[:, None] - q[None, :]
        mask = (dist >= 0) & (dist < self.max_past_horizon)
        if self.max_future_horizon > 0:
            mask = mask | ((dist < 0) & ((-dist) < self.max_future_horizon))
        if audio_mask is not None:
            mask = mask & audio_mask[0, None, :].bool()
        m = mask[None, None]
        # Reshape to blocked 5D matching reference code
        p = num_blocks * self.chunk_size - seq_len
        m = torch.nn.functional.pad(m, (0, p, 0, p), value=False)
        m = m.reshape(1, 1, num_blocks, self.chunk_size, -1)
        m = torch.nn.functional.pad(m, (self.max_past_horizon, self.max_future_horizon), value=False)
        idx = (torch.arange(num_blocks, device=device) * self.chunk_size)[:, None] + torch.arange(self.context_size, device=device)[None, :]
        return m.gather(-1, idx[None, None, :, None, :].expand(1, 1, -1, self.chunk_size, -1))

    def forward(self, x, position_embeddings=None, attn_mask=None):
        B, S, _ = x.shape

        q = self.q_proj(x).float().view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).float().view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).float().view(B, S, self.num_heads, self.head_dim)

        q = q * self.q_scale * torch.nn.functional.softplus(self.per_dim_scale)
        k = k * self.k_scale

        q_blocks = self._convert_to_block(q)
        k_context = self._extract_block_context(k)
        v_context = self._extract_block_context(v)
        num_blocks = q_blocks.shape[1]

        rel_k = self.relative_k_proj(position_embeddings).view(-1, self.num_heads, self.head_dim).to(q.dtype)

        queries = q_blocks.permute(0, 3, 1, 2, 4)  # [B, H, NB, CS, D]
        matrix_ac = queries @ k_context.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(B, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ rel_k.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(B, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = torch.tanh(attn_weights / self.softcap) * self.softcap

        # Mask out invalid positions in chunk context (matching reference's masked_fill approach)
        if attn_mask is None:
            attn_mask = self._build_blocked_mask(S, num_blocks, x.device)
        attn_weights = attn_weights.masked_fill(attn_mask.logical_not(), -1e9)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(v.dtype)
        out = attn_weights @ v_context.permute(0, 3, 1, 2, 4)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * self.chunk_size, -1)
        out = out[:, :S].contiguous()
        return self.post(out.to(self.post.linear.weight.dtype))


class Gemma4AudioLConv1d(nn.Module):
    """Lightweight convolution with standard GLU."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden_size = config["hidden_size"]
        conv_kernel_size = config.get("conv_kernel_size", 5)
        self.pre_layer_norm = RMSNorm(hidden_size, eps=config["rms_norm_eps"], device=device, dtype=dtype)
        self.linear_start = ClippedLinear(hidden_size, hidden_size * 2, device=device, dtype=dtype, ops=ops)
        # Causal conv: left-pad only
        self.depthwise_conv1d = ops.Conv1d(hidden_size, hidden_size, kernel_size=conv_kernel_size, padding=0, groups=hidden_size, bias=False, device=device, dtype=dtype)
        self.conv_left_pad = conv_kernel_size - 1  # causal: pad left by kernel-1
        self.conv_norm = RMSNorm(hidden_size, eps=config["rms_norm_eps"], device=device, dtype=dtype)
        self.linear_end = ClippedLinear(hidden_size, hidden_size, device=device, dtype=dtype, ops=ops)

    def forward(self, x):
        residual = x
        x = self.pre_layer_norm(x)
        x = self.linear_start(x)
        x = torch.nn.functional.glu(x, dim=-1)
        x = x.transpose(1, 2)
        x = torch.nn.functional.pad(x, (self.conv_left_pad, 0))
        x = self.depthwise_conv1d(x).transpose(1, 2)
        x = self.conv_norm(x)
        x = torch.nn.functional.silu(x)
        x = self.linear_end(x)
        return x + residual


class Gemma4AudioLayer(nn.Module):
    """Conformer block: FFN1 -> Attention -> LConv -> FFN2."""
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.feed_forward1 = Gemma4AudioFeedForward(config, device=device, dtype=dtype, ops=ops)
        self.self_attn = Gemma4AudioAttention(config, device=device, dtype=dtype, ops=ops)
        norm_kwargs = dict(eps=config["rms_norm_eps"], device=device, dtype=dtype)
        hidden_size = config["hidden_size"]
        self.norm_pre_attn = RMSNorm(hidden_size, **norm_kwargs)
        self.norm_post_attn = RMSNorm(hidden_size, **norm_kwargs)
        self.lconv1d = Gemma4AudioLConv1d(config, device=device, dtype=dtype, ops=ops)
        self.feed_forward2 = Gemma4AudioFeedForward(config, device=device, dtype=dtype, ops=ops)
        self.norm_out = RMSNorm(hidden_size, **norm_kwargs)

    def forward(self, x, position_embeddings=None, attn_mask=None):
        x = self.feed_forward1(x)

        residual = x
        x = self.norm_pre_attn(x)
        x = self.self_attn(x, position_embeddings=position_embeddings, attn_mask=attn_mask)
        x = self.norm_post_attn(x)
        x = x + residual

        x = self.lconv1d(x)
        x = self.feed_forward2(x)

        x = self.norm_out(x)
        return x


class Gemma4AudioEncoder(nn.Module):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.output_proj_dims = config.get("output_proj_dims", 1536)

        self.subsample_conv_projection = Gemma4AudioConvSubsampler(config, device=device, dtype=dtype, ops=ops)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            Gemma4AudioLayer(config, device=device, dtype=dtype, ops=ops)
            for _ in range(config["num_hidden_layers"])
        ])

        self.output_proj = ops.Linear(self.hidden_size, self.output_proj_dims, bias=True, device=device, dtype=dtype)

    def forward(self, audio_features, audio_mask=None):
        x, audio_mask = self.subsample_conv_projection(audio_features, audio_mask)
        position_embeddings = self.rel_pos_enc(x)

        # Build blocked attention mask once for all layers
        attn_mask = self.layers[0].self_attn._build_blocked_mask(
            x.shape[1], (x.shape[1] + self.layers[0].self_attn.chunk_size - 1) // self.layers[0].self_attn.chunk_size,
            x.device, audio_mask=audio_mask)

        for layer in self.layers:
            x = layer(x, position_embeddings=position_embeddings, attn_mask=attn_mask)

        x = self.output_proj(x)
        return x


class Gemma4AudioProjector(Gemma4RMSNormProjector):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__(config.get("audio_output_proj_dims", 1536), config.get("text_hidden_size", 2560), dtype=dtype, device=device, ops=ops)


# Tokenizer and Wrappers

class Gemma4_Tokenizer():
    tokenizer_json_data = None

    def state_dict(self):
        if self.tokenizer_json_data is not None:
            return {"tokenizer_json": self.tokenizer_json_data}
        return {}

    def _extract_mel_spectrogram(self, waveform, sample_rate):
        """Extract 128-bin log mel spectrogram.
        Uses numpy for FFT/matmul/log to produce bit-identical results with reference code.
        """
        # Mix to mono first, then resample to 16kHz
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio = waveform.squeeze(0).float().numpy()
        if sample_rate != 16000:
            # Use scipy's resample_poly with a high-quality FIR filter to get as close as possible to librosa's resampling (while still not full match)
            from scipy.signal import resample_poly, firwin
            from math import gcd
            g = gcd(sample_rate, 16000)
            up, down = 16000 // g, sample_rate // g
            L = max(up, down)
            h = firwin(160 * L + 1, 0.96 / L, window=('kaiser', 6.5))
            audio = resample_poly(audio, up, down, window=h).astype(np.float32)
        n = len(audio)

        # Pad to multiple of 128, build sample-level mask
        if n % 128 != 0:
            audio = np.pad(audio, (0, 128 - n % 128))
        mask_raw = np.ones(len(audio), dtype=np.float32)
        mask_raw[n:] = 0.0

        # Semicausal padding: 160 zeros prepended
        audio = np.pad(audio, (160, 0))
        mask_raw = np.pad(mask_raw, (160, 0))

        # Extract 321-sample frames via stride tricks, drop last → 320
        nf = (len(audio) - 321) // 160 + 1
        strides = (audio.strides[0] * 160, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, (nf, 321), strides)[..., :-1].copy()

        # Periodic Hann window, FFT magnitude, mel filterbank, log
        window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(320) / 320)).astype(np.float32)
        magnitude = np.abs(np.fft.rfft(frames * window, n=512, axis=-1))
        mel_fb = self._build_mel_filterbank()
        log_mel = np.log(np.matmul(magnitude, mel_fb) + np.float64(0.001)).astype(np.float32)

        # Frame mask: valid when last sample in window is real audio
        mask = mask_raw[np.arange(nf) * 160 + 320].astype(bool)
        log_mel = log_mel * mask[:, None]
        return torch.from_numpy(log_mel), torch.from_numpy(mask)  # [T, 128], [T]

    @staticmethod
    def _build_mel_filterbank():
        """Build 128-bin HTK mel filterbank [257, 128] for 512-pt FFT at 16kHz."""
        mel_freqs = np.linspace(0.0, 2595.0 * np.log10(1.0 + 8000.0 / 700.0), 130)
        filter_freqs = 700.0 * (10.0 ** (mel_freqs / 2595.0) - 1.0)
        fft_freqs = np.linspace(0, 16000 // 2, 257)
        filter_diff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    def tokenize_with_weights(self, text, return_word_ids=False, image=None, audio=None, video=None, llama_template=None, skip_template=True, thinking=False, **kwargs):

        # Process audio
        audio_features = []
        if audio is not None:
            waveform = audio["waveform"].squeeze(0) if hasattr(audio, "__getitem__") else audio
            sample_rate = audio.get("sample_rate", 16000) if hasattr(audio, "get") else 16000
            mel, mel_mask = self._extract_mel_spectrogram(waveform, sample_rate)
            audio_features = [(mel.unsqueeze(0), mel_mask.unsqueeze(0))]  # ([1, T, 128], [1, T])

        # Process image/video frames
        is_video = video is not None
        source = video if is_video else image
        images = []
        if source is not None:
            samples = source.movedim(-1, 1)  # [B, C, H, W]
            num_frames = samples.shape[0]

            # Subsample video to 1fps
            if is_video:
                fps = kwargs.get("fps", 24)
                step = max(1, round(fps))
                indices = list(range(0, num_frames, step))
                if len(indices) == 0:
                    indices = [0]
                samples = samples[indices]
                num_frames = len(indices)

            h, w = samples.shape[2], samples.shape[3]
            patch_size = 16
            pooling_k = 3
            max_soft_tokens = 70 if is_video else 280  # video uses smaller token budget per frame
            max_patches = max_soft_tokens * pooling_k * pooling_k
            target_px = max_patches * patch_size * patch_size
            factor = (target_px / (h * w)) ** 0.5
            side_mult = pooling_k * patch_size
            target_h = max(int(factor * h // side_mult) * side_mult, side_mult)
            target_w = max(int(factor * w // side_mult) * side_mult, side_mult)

            import torchvision.transforms.functional as TVF
            for i in range(num_frames):
                # rescaling to match reference code
                s = (samples[i].clamp(0, 1) * 255).to(torch.uint8)  # [C, H, W] uint8
                if target_h != h or target_w != w:
                    s = TVF.resize(s, [target_h, target_w], interpolation=TVF.InterpolationMode.BICUBIC, antialias=True)
                s = s.float() * (1.0 / 255.0)
                images.append({"pixels": s.unsqueeze(0).movedim(1, -1)[:, :, :, :3], "max_soft_tokens": max_soft_tokens})

        if text.startswith('<|turn>'):
            skip_template = True

        if skip_template:
            llama_text = text
        else:
            if llama_template is not None:
                llama_text = llama_template.format(text)
            else:
                # Build template from modalities present
                system = "<|turn>system\n<|think|><turn|>\n" if thinking else ""
                media = ""
                if len(images) > 0:
                    if is_video:
                        media += "\n\n"
                        for i in range(len(images)):
                            ts = f"{int(i // 60):02d}:{int(i % 60):02d}"
                            sep = "" if i == 0 else " "
                            media += f"{sep}{ts} <|image><|video|><image|>"
                        media += "\n\n"
                    else:
                        media += "\n\n"
                        for i in range(len(images)):
                            if i > 0:
                                media += "\n\n\n\n"
                            media += "<|image><|image|><image|>"
                        media += "\n\n"
                if len(audio_features) > 0:
                    # Compute audio token count (always at 16kHz)
                    num_samples = int(waveform.shape[-1] * 16000 / sample_rate) if sample_rate != 16000 else waveform.shape[-1]
                    _fl = 320  # int(round(16000 * 20.0 / 1000.0))
                    _hl = 160  # int(round(16000 * 10.0 / 1000.0))
                    _nmel = (num_samples + _fl // 2 - (_fl + 1)) // _hl + 1
                    _t = _nmel
                    for _ in range(2):
                        _t = (_t + 2 - 3) // 2 + 1
                    n_audio_tokens = min(_t, 750)
                    media += "<|audio>" + "<|audio|>" * n_audio_tokens + "<audio|>"
                llama_text = f"{system}<|turn>user\n{media}{text}<turn|>\n<|turn>model\n"

        text_tokens = super().tokenize_with_weights(llama_text, return_word_ids)

        def _replace_placeholders(token_list, token_id, embeds):
            """Replace first placeholder with embed dict, remove remaining consecutive ones."""
            embed_idx = 0
            i = 0
            while i < len(token_list):
                if token_list[i][0] == token_id and embed_idx < len(embeds):
                    token_list[i] = (embeds[embed_idx],) + token_list[i][1:]
                    embed_idx += 1
                    i += 1
                    while i < len(token_list) and token_list[i][0] == token_id:
                        token_list.pop(i)
                else:
                    i += 1

        if len(images) > 0:
            img_token_id = 258884 if is_video else 258880
            img_embeds = [{"type": "image", "data": img["pixels"], "max_soft_tokens": img["max_soft_tokens"]} for img in images]
            for r in text_tokens:
                _replace_placeholders(r, img_token_id, img_embeds)

        if len(audio_features) > 0:
            aud_embeds = [{"type": "audio", "data": mel, "mask": mask} for mel, mask in audio_features]
            for r in text_tokens:
                _replace_placeholders(r, 258881, aud_embeds)

        return text_tokens


class _Gemma4Tokenizer:
    """Tokenizer using the tokenizers (Gemma4 doesn't come with sentencepiece model)"""
    def __init__(self, tokenizer_json_bytes=None, **kwargs):
        from tokenizers import Tokenizer
        if isinstance(tokenizer_json_bytes, torch.Tensor):
            tokenizer_json_bytes = bytes(tokenizer_json_bytes.tolist())
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


# Tokenizer
class Gemma4SDTokenizer(Gemma4_Tokenizer, sd1_clip.SDTokenizer):
    embedding_size = 2560
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_json = tokenizer_data.get("tokenizer_json", None)
        self.tokenizer_json_data = tokenizer_json
        super().__init__(tokenizer_json, pad_with_end=False, embedding_size=self.embedding_size, embedding_key='gemma4', tokenizer_class=_Gemma4Tokenizer, has_start_token=True, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_left=True, disable_weights=True, start_token=2, tokenizer_data=tokenizer_data)

    def decode(self, token_ids, **kwargs):
        text = super().decode(token_ids, skip_special_tokens=False)
        # Translate thinking channel markers to standard <think>/</think> tags
        text = text.replace("<|channel>thought\n", "<think>\n")
        text = text.replace("<channel|>", "</think>")
        # Strip remaining special tokens
        text = text.replace("<turn|>", "").replace("<eos>", "").strip()
        return text


class Gemma4Tokenizer(sd1_clip.SD1Tokenizer):
    tokenizer_class = Gemma4SDTokenizer
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma4", tokenizer=self.tokenizer_class)


# Model wrappers
class Gemma4Model(sd1_clip.SDClipModel):
    model_class = None
    def __init__(self, device="cpu", layer="all", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        self.dtypes = set()
        self.dtypes.add(dtype)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=self.model_class, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

    def process_tokens(self, tokens, device):
        embeds, _, _, _ = super().process_tokens(tokens, device)
        return embeds

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, presence_penalty=0.0):
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = sd1_clip.SDClipModel.process_tokens(self, tokens_only, self.execution_device)
        seq_len = embeds.shape[1]
        ids = [0] * seq_len
        expanded_idx = 0
        embed_map = {info["index"]: info["size"] for info in embeds_info}
        for t in tokens_only[0]:
            if expanded_idx in embed_map:
                expanded_idx += embed_map[expanded_idx]
            elif isinstance(t, int):
                if expanded_idx < seq_len:
                    ids[expanded_idx] = t
                expanded_idx += 1
            else:
                expanded_idx += 1
        initial_token_ids = [ids]
        input_ids = torch.tensor(initial_token_ids, device=self.execution_device)
        return self.transformer.generate(embeds, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed, initial_tokens=initial_token_ids[0], presence_penalty=presence_penalty, initial_input_ids=input_ids)


def gemma4_te(dtype_llama=None, llama_quantization_metadata=None, model_class=None):
    clip_model = type('Gemma4Model_', (Gemma4Model,), {'model_class': model_class})
    class Gemma4TEModel_(sd1_clip.SD1ClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name="gemma4", clip_model=clip_model, model_options=model_options)
    return Gemma4TEModel_


# Variants

def _make_variant(config_cls):
    audio = config_cls.audio_config is not None
    bases = (Gemma4AudioMixin, Gemma4Base) if audio else (Gemma4Base,)
    class Variant(*bases):
        def __init__(self, config_dict, dtype, device, operations):
            super().__init__()
            self._init_model(config_cls(**config_dict), dtype, device, operations)
            if audio:
                self._init_audio(self.model.config, dtype, device, operations)
    embedding_size = config_cls.hidden_size
    if embedding_size != Gemma4SDTokenizer.embedding_size:
        tok_cls = type('T', (Gemma4SDTokenizer,), {'embedding_size': embedding_size})
        class Tokenizer(Gemma4Tokenizer):
            tokenizer_class = tok_cls
        Variant.tokenizer = Tokenizer
    else:
        Variant.tokenizer = Gemma4Tokenizer
    return Variant

Gemma4_E4B = _make_variant(Gemma4Config)
Gemma4_E2B = _make_variant(Gemma4_E2B_Config)
Gemma4_31B = _make_variant(Gemma4_31B_Config)
