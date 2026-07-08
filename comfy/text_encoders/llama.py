import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import math
from tqdm import tqdm
import comfy.utils

from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.model_management
import comfy.ops
import comfy.ldm.common_dit
import comfy.clip_model

from . import qwen_vl

@dataclass
class Llama2Config:
    vocab_size: int = 128320
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Mistral3Small24BConfig:
    vocab_size: int = 131072
    hidden_size: int = 5120
    intermediate_size: int = 32768
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Ministral3_3BConfig:
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [2]

@dataclass
class Qwen25_3BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Qwen3_06BConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_06B_ACE15_Config:
    vocab_size: int = 151669
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_2B_ACE15_lm_Config:
    vocab_size: int = 217204
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_4B_ACE15_lm_Config:
    vocab_size: int = 217204
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_4BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_8BConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = True
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3VL_8BConfig(Qwen3_8BConfig):
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    rope_dims = [24, 20, 20]
    interleaved_mrope = True

@dataclass
class Qwen3VL_4BConfig(Qwen3VL_8BConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    lm_head: bool = False  # 4B ties word embeddings

@dataclass
class Ovis25_2BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Qwen25_7BVLI_Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Gemma2_2B_Config:
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    transformer_type: str = "gemma2"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    sliding_attention = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [1]

@dataclass
class Gemma3_4B_Config:
    vocab_size: int = 262208
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    rope_scale = [8.0, 1.0]
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [1, 106]

GEMMA3_VISION_CONFIG = {"num_channels": 3, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "model_type": "siglip_vision_model", "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14}

@dataclass
class Gemma3_4B_Vision_Config(Gemma3_4B_Config):
    vision_config = GEMMA3_VISION_CONFIG
    mm_tokens_per_image = 256

@dataclass
class Gemma3_12B_Config:
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    rope_scale = [8.0, 1.0]
    final_norm: bool = True
    lm_head: bool = False
    vision_config = GEMMA3_VISION_CONFIG
    mm_tokens_per_image = 256
    stop_tokens = [1, 106]

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, add=False, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.add = add

    def forward(self, x: torch.Tensor):
        w = self.weight
        if self.add:
            w = w + 1.0

        return comfy.ldm.common_dit.rms_norm(x, w, self.eps)



def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None, interleaved_mrope=False):
    if not isinstance(theta, list):
        theta = [theta]

    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))

        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        if rope_dims is not None and position_ids.shape[0] > 1 and interleaved_mrope:
            # Qwen3-VL interleaved MRoPE: T-freqs by default, H/W replace every 3rd dim.
            freqs_inter = freqs[0].clone()
            for axis_idx, offset in ((1, 1), (2, 2)):
                length = rope_dims[axis_idx] * 3
                idx = slice(offset, length, 3)
                freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
            emb = torch.cat((freqs_inter, freqs_inter), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
        else:
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            if rope_dims is not None and position_ids.shape[0] > 1:
                mrope_section = rope_dims * 2
                cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
                sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        sin_split = sin.shape[-1] // 2
        out.append((cos, sin[..., : sin_split], -sin[..., sin_split :]))

    if len(out) == 1:
        return out[0]

    return out

def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    nsin = freqs_cis[2]

    q_embed = (xq * cos)
    q_split = q_embed.shape[-1] // 2
    q_embed[..., : q_split].addcmul_(xq[..., q_split :], nsin)
    q_embed[..., q_split :].addcmul_(xq[..., : q_split], sin)

    k_embed = (xk * cos)
    k_split = k_embed.shape[-1] // 2
    k_embed[..., : k_split].addcmul_(xk[..., k_split :], nsin)
    k_embed[..., k_split :].addcmul_(xk[..., : k_split], sin)

    return q_embed.to(org_dtype), k_embed.to(org_dtype)


class Attention(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        ops = ops or nn
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = None
        self.k_norm = None

        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sliding_window: Optional[int] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            index = 0
            num_tokens = xk.shape[2]
            if len(past_key_value) > 0:
                past_key, past_value, index = past_key_value
                if past_key.shape[2] >= (index + num_tokens):
                    past_key[:, :, index:index + xk.shape[2]] = xk
                    past_value[:, :, index:index + xv.shape[2]] = xv
                    xk = past_key[:, :, :index + xk.shape[2]]
                    xv = past_value[:, :, :index + xv.shape[2]]
                    present_key_value = (past_key, past_value, index + num_tokens)
                else:
                    xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                    xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                    present_key_value = (xk, xv, index + num_tokens)
            else:
                present_key_value = (xk, xv, index + num_tokens)

            if sliding_window is not None and xk.shape[2] > sliding_window and seq_length == 1:
                xk = xk[:, :, -sliding_window:]
                xv = xv[:, :, -sliding_window:]
                attention_mask = attention_mask[..., -sliding_window:] if attention_mask is not None else None

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output), present_key_value

class MLP(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = ops.Linear(config.hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        if config.mlp_activation == "silu":
            self.activation = torch.nn.functional.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: torch.nn.functional.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )
        x = residual + x

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_key_value

class TransformerBlockGemma2(nn.Module):
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        self.transformer_type = config.transformer_type

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        sliding_window = None
        if self.transformer_type == 'gemma3':
            if self.sliding_attention:
                sliding_window = self.sliding_attention
                if x.shape[1] > self.sliding_attention:
                    sliding_mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
                    sliding_mask.tril_(diagonal=-self.sliding_attention)
                    if attention_mask is not None:
                        attention_mask = attention_mask + sliding_mask
                    else:
                        attention_mask = sliding_mask
                freqs_cis = freqs_cis[1]
            else:
                freqs_cis = freqs_cis[0]

        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
            sliding_window=sliding_window,
        )

        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x, present_key_value

def _make_scaled_embedding(ops, vocab_size, hidden_size, scale, device, dtype):
    class ScaledEmbedding(ops.Embedding):
        def forward(self, input_ids, out_dtype=None):
            return super().forward(input_ids, out_dtype=out_dtype) * scale
    return ScaledEmbedding(vocab_size, hidden_size, device=device, dtype=dtype)


class Llama2_(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        if self.config.transformer_type == "gemma2" or self.config.transformer_type == "gemma3":
            transformer = TransformerBlockGemma2
            self.embed_tokens = _make_scaled_embedding(ops, config.vocab_size, config.hidden_size, config.hidden_size ** 0.5, device, dtype)
        else:
            transformer = TransformerBlock
            self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            transformer(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.norm = None

        if config.lm_head:
            self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def get_past_len(self, past_key_values):
        return past_key_values[0][2]

    def compute_freqs_cis(self, position_ids, device):
        return precompute_freqs_cis(self.config.head_dim,
                                    position_ids,
                                    self.config.rope_theta,
                                    self.config.rope_scale,
                                    self.config.rope_dims,
                                    interleaved_mrope=getattr(self.config, "interleaved_mrope", False),
                                    device=device)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, position_ids=None, embeds_info=[], past_key_values=None, input_ids=None,deepstack_embeds=None, visual_pos_masks=None):
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

        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        next_key_values = []
        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i] if len(past_key_values) > 0 else []

            x, current_kv = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
                past_key_value=past_kv,
            )

            if current_kv is not None:
                next_key_values.append(current_kv)

            # DeepStack: add per-layer visual features into the first len() decoder layers at image positions (Qwen3-VL)
            if deepstack_embeds is not None and i < len(deepstack_embeds):
                x[visual_pos_masks] = x[visual_pos_masks] + deepstack_embeds[i].to(x)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            if only_layers is None or ((i + 1) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        else:
            return x, intermediate


class Gemma3MultiModalProjector(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.empty(config.vision_config["hidden_size"], config.hidden_size, device=device, dtype=dtype)
        )

        self.mm_soft_emb_norm = RMSNorm(config.vision_config["hidden_size"], eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        self.patches_per_image = int(config.vision_config["image_size"] // config.vision_config["patch_size"])
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, comfy.model_management.cast_to_device(self.mm_input_projection_weight, device=normed_vision_outputs.device, dtype=normed_vision_outputs.dtype))
        return projected_vision_outputs.type_as(vision_outputs)


class BaseLlama:
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def forward(self, input_ids, *args, **kwargs):
        return self.model(input_ids, *args, **kwargs)

class BaseGenerate:
    def logits(self, x):
        input = x[:, -1:]
        if hasattr(self.model, "lm_head"):
            module = self.model.lm_head
        else:
            module = self.model.embed_tokens

        offload_stream = None
        if module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, input, offloadable=True)
        else:
            weight = self.model.embed_tokens.weight.to(x)

        x = torch.nn.functional.linear(input, weight, None)

        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
        return x

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        model_config = self.model.config
        past_key_values = []
        for x in range(model_config.num_hidden_layers):
            past_key_values.append((torch.empty([batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim], device=device, dtype=execution_dtype),
                                    torch.empty([batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim], device=device, dtype=execution_dtype), 0))
        return past_key_values

    def generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, initial_tokens=[], execution_dtype=None, min_tokens=0, presence_penalty=0.0, initial_input_ids=None, position_ids=None, deepstack_embeds=None, visual_pos_masks=None):
        device = embeds.device

        if stop_tokens is None:
            stop_tokens = self.model.config.stop_tokens

        if execution_dtype is None:
            if comfy.model_management.should_use_bf16(device):
                execution_dtype = torch.bfloat16
            else:
                execution_dtype = torch.float32
        embeds = embeds.to(execution_dtype)

        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)

        max_cache_len = embeds.shape[1] + max_length
        past_key_values = self.init_kv_cache(embeds.shape[0], max_cache_len, device, execution_dtype)

        generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None

        generated_token_ids = []
        pbar = comfy.utils.ProgressBar(max_length)

        # MRoPE: prefill uses explicit 3D position_ids, decode continues from the last position
        next_pos = int(position_ids[:, -1].max()) + 1 if position_ids is not None else None

        # Generation loop
        current_input_ids = initial_input_ids
        for step in tqdm(range(max_length), desc="Generating tokens"):
            # DeepStack visual features are injected on the prefill only; gemma4's forward lacks these kwargs.
            extra = {}
            if step == 0 and deepstack_embeds is not None:
                extra["deepstack_embeds"] = deepstack_embeds
                extra["visual_pos_masks"] = visual_pos_masks
            x, _, past_key_values = self.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=past_key_values, input_ids=current_input_ids, position_ids=position_ids, **extra)
            logits = self.logits(x)[:, -1]
            next_token = self.sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, initial_tokens + generated_token_ids, generator, do_sample=do_sample, presence_penalty=presence_penalty)
            token_id = next_token[0].item()
            generated_token_ids.append(token_id)

            embeds = self.model.embed_tokens(next_token).to(execution_dtype)
            current_input_ids = next_token if initial_input_ids is not None else None
            if next_pos is not None:  # advance MRoPE position for the next (decode) step
                position_ids = torch.tensor([[next_pos]], device=device)
                next_pos += 1
            pbar.update(1)

            if token_id in stop_tokens:
                break

        return generated_token_ids

    def sample_token(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=True, presence_penalty=0.0):

        if not do_sample or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Sampling mode
        if repetition_penalty != 1.0:
            for i in range(logits.shape[0]):
                for token_id in set(token_history):
                    logits[i, token_id] *= repetition_penalty if logits[i, token_id] < 0 else 1/repetition_penalty

        if presence_penalty is not None and presence_penalty != 0.0:
            for i in range(logits.shape[0]):
                for token_id in set(token_history):
                    logits[i, token_id] -= presence_penalty

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        if min_p > 0.0:
            probs_before_filter = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, _ = probs_before_filter.max(dim=-1, keepdim=True)
            min_threshold = min_p * top_probs
            indices_to_remove = probs_before_filter < min_threshold
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        probs = torch.nn.functional.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1, generator=generator)

class BaseQwen3:
    def logits(self, x):
        input = x[:, -1:]
        if self.model.config.lm_head:
            return self.model.lm_head(input)

        module = self.model.embed_tokens

        offload_stream = None
        if module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, input, offloadable=True)
        else:
            weight = self.model.embed_tokens.weight.to(x)

        x = torch.nn.functional.linear(input, weight, None)

        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
        return x

class Llama2(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Llama2Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Mistral3Small24B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Mistral3Small24BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Ministral3_3B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Ministral3_3BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen25_3B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_3BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_06B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_06BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_06B_ACE15(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_06B_ACE15_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_2B_ACE15_lm(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_2B_ACE15_lm_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_4B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_4BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_4B_ACE15_lm(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_4B_ACE15_lm_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_8B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_8BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Ovis25_2B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Ovis25_2BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen25_7BVLI(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_7BVLI_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.visual = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=config.hidden_size, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

        # todo: should this be tied or not?
        #self.lm_head = operations.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = qwen_vl.process_qwen2vl_images(embed["data"])
            return self.visual(image.to(device, dtype=torch.float32), grid), grid
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") == "image":
                grid = e.get("extra", None)
                start = e.get("index")
                if position_ids is None:
                    position_ids = torch.ones((3, embeds.shape[1]), device=embeds.device, dtype=torch.long)
                    position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
                end = e.get("size") + start
                len_max = int(grid.max()) // 2
                start_next = len_max + start
                if attention_mask is not None:
                    # Assign compact sequential positions to attended tokens only,
                    # skipping over padding so post-padding tokens aren't inflated.
                    after_mask = attention_mask[0, end:]
                    text_positions = after_mask.cumsum(0) - 1 + start_next + offset
                    position_ids[:, end:] = torch.where(after_mask.bool(), text_positions, position_ids[0, end:])
                else:
                    position_ids[:, end:] = torch.arange(start_next + offset, start_next + (embeds.shape[1] - end) + offset, device=embeds.device)
                position_ids[0, start:end] = start + offset
                max_d = int(grid[0][1]) // 2
                position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
                max_d = int(grid[0][2]) // 2
                position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
                offset += len_max - (end - start)

        if grid is None:
            position_ids = None

        return super().forward(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, position_ids=position_ids)

class Gemma2_2B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma2_2B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma3_4B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_4B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma3_4B_Vision(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_4B_Vision_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype
        self.multi_modal_projector = Gemma3MultiModalProjector(config, dtype, device, operations)
        self.vision_model = comfy.clip_model.CLIPVision(config.vision_config, dtype, device, operations)
        self.image_size = config.vision_config["image_size"]

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = comfy.clip_model.clip_preprocess(embed["data"], size=self.image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True)
            return self.multi_modal_projector(self.vision_model(image.to(device, dtype=torch.float32))[0]), None
        return None, None

class Gemma3_12B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_12B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.multi_modal_projector = Gemma3MultiModalProjector(config, dtype, device, operations)
        self.vision_model = comfy.clip_model.CLIPVision(config.vision_config, dtype, device, operations)
        self.dtype = dtype
        self.image_size = config.vision_config["image_size"]

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = comfy.clip_model.clip_preprocess(embed["data"], size=self.image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True)
            return self.multi_modal_projector(self.vision_model(image.to(device, dtype=torch.float32))[0]), None
        return None, None
