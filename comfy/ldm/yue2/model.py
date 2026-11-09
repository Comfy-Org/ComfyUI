"""YuE2 acoustic transformer. Adapted from M·A·P YuE2 (Apache-2.0)."""

import torch
from torch import nn

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from comfy.text_encoders.llama import Qwen3_8BConfig, RMSNorm, TransformerBlock, precompute_freqs_cis


def model_config(**overrides):
    return Qwen3_8BConfig(**{
        "vocab_size": 184704, "hidden_size": 2048, "intermediate_size": 6144,
        "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
        "max_position_embeddings": 24576, "merged_qkv": True, "merged_mlp": True,
        **overrides,
    })


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, dtype, device, operations):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(256, hidden_size, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, dtype=dtype, device=device),
        )

    def forward(self, t, dtype):
        return self.mlp(timestep_embedding(t, 256).to(dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, frames, hidden_size, dtype, device):
        super().__init__()
        self.register_buffer("pe", torch.empty(frames, hidden_size, dtype=dtype, device=device))

    def forward(self, length, x):
        return comfy.ops.cast_to_input(self.pe[:length], x)


class YuE2(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.config = model_config(**kwargs.get("config", {}))
        config = self.config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            TransformerBlock(config, i, device=device, dtype=dtype, ops=operations)
            for i in range(config.num_hidden_layers)
        ])
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.vae2llm = operations.Linear(64, config.hidden_size, dtype=dtype, device=device)
        self.llm2vae = operations.Linear(config.hidden_size, 64, dtype=dtype, device=device)
        self.time_embedder = TimestepEmbedder(config.hidden_size, dtype, device, operations)
        self.latent_pos_embed = AudioPositionEmbedding(config.max_position_embeddings, config.hidden_size, dtype, device)

    def forward(self, x, timestep, context, yue2_chunks, transformer_options={}, **kwargs):
        batch, channels, frames = x.shape
        if frames != yue2_chunks[-1][1]:
            raise ValueError("YuE2 latent duration must match the seconds output of YuE2 Text Encode.")
        config = self.config
        time = self.time_embedder(timestep.to(x.dtype), x.dtype)[:, None]
        output = torch.empty_like(x)
        attention = optimized_attention_for_device(x.device)
        for start, end, kv_start, kv_end in yue2_chunks:
            comfy.model_management.throw_exception_if_processing_interrupted()
            ar_length = kv_end - kv_start
            length = end - start + 2
            state = torch.nn.functional.pad(x[..., start:end].transpose(1, 2), (0, 0, 1, 1))
            state = self.vae2llm(state) + time + self.latent_pos_embed(length, x)[None]
            positions = torch.arange(ar_length, ar_length + length, device=x.device)[None]
            rope = precompute_freqs_cis(config.head_dim, positions, config.rope_theta, device=x.device)
            prefix = context[:, kv_start:kv_end].reshape(batch, ar_length, config.num_hidden_layers, 2, config.num_key_value_heads, config.head_dim)
            prefix = prefix.permute(2, 3, 0, 4, 1, 5)
            prefetch = comfy.model_prefetch.make_prefetch_queue(list(self.model.layers), x.device, transformer_options)
            for index, layer in enumerate(self.model.layers):
                comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, layer, state.dtype)
                state, _ = layer(state, freqs_cis=rope, optimized_attention=attention,
                                 past_key_value=(prefix[index, 0], prefix[index, 1], ar_length))
            comfy.model_prefetch.prefetch_queue_pop(prefetch, x.device, None)
            output[..., start:end] = self.llm2vae(self.model.norm(state))[:, 1:-1].transpose(1, 2)
        return output
