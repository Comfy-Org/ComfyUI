import math

import torch
from torch import nn

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention_for_device


MAX_CONDITION_FRAMES = 200
CONDITION_HOP_FRAMES = 100


def latent_length(audio_frames):
    return max(1, int(audio_frames * 44100 / 24000 * 960 / 512))


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features // 2, in_features, dtype=dtype, device=device))

    def forward(self, value):
        weight = comfy.ops.cast_to_input(self.weight, value)
        features = 2.0 * math.pi * value @ weight.T
        return torch.cat((features.cos(), features.sin()), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim, dtype, device):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        self.register_buffer("beta", torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            comfy.ops.cast_to_input(self.gamma, x),
            comfy.ops.cast_to_input(self.beta, x),
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, dtype, device):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(dim // 2, dtype=dtype, device=device))

    def forward_from_seq_len(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, comfy.ops.cast_to_input(self.inv_freq, positions))
        frequencies = frequencies.to(dtype)
        cos, sin = frequencies.cos(), frequencies.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, length, frequencies.shape[-1], 2, 2)


def _apply_rope(x, rotation_matrix):
    x_dtype = x.dtype
    x = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(rotation_matrix.dtype)
    x = rotation_matrix[..., 0] * x[..., 0] + rotation_matrix[..., 1] * x[..., 1]
    return x.movedim(-1, -2).flatten(-2).to(x_dtype)


class Attention(nn.Module):
    def __init__(self, dim, dim_heads, dtype, device, operations):
        super().__init__()
        self.num_heads = dim // dim_heads
        self.dim_heads = dim_heads
        self.to_qkv = operations.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)
        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, rotation_matrix):
        batch, length, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, length, self.num_heads, self.dim_heads).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.dim_heads).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, self.dim_heads).transpose(1, 2)
        rotary_dims = rotation_matrix.shape[-3] * 2
        if comfy.model_management.in_training:
            q = torch.cat((_apply_rope(q[..., :rotary_dims], rotation_matrix), q[..., rotary_dims:]), dim=-1)
            k = torch.cat((_apply_rope(k[..., :rotary_dims], rotation_matrix), k[..., rotary_dims:]), dim=-1)
        else:
            rotated_q, rotated_k = comfy.quant_ops.ck.apply_rope_split_half(q[..., :rotary_dims], k[..., :rotary_dims], rotation_matrix)
            q = torch.cat((rotated_q, q[..., rotary_dims:]), dim=-1)
            k = torch.cat((rotated_k, k[..., rotary_dims:]), dim=-1)
        attention = optimized_attention_for_device(q.device)
        out = attention(q, k, v, self.num_heads, skip_reshape=True)
        return self.to_out(out)


class GLU(nn.Module):
    def __init__(self, dim, inner_dim, dtype, device, operations):
        super().__init__()
        self.proj = operations.Linear(dim, inner_dim * 2, dtype=dtype, device=device)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * torch.nn.functional.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim, dtype, device, operations):
        super().__init__()
        self.ff = nn.Sequential(
            GLU(dim, inner_dim, dtype, device, operations),
            nn.Identity(),
            operations.Linear(inner_dim, dim, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_heads, inner_dim, dtype, device, operations):
        super().__init__()
        self.pre_norm = LayerNorm(dim, dtype, device)
        self.self_attn = Attention(dim, dim_heads, dtype, device, operations)
        self.ff_norm = LayerNorm(dim, dtype, device)
        self.ff = FeedForward(dim, inner_dim, dtype, device, operations)

    def forward(self, x, rotation_matrix):
        x = x + self.self_attn(self.pre_norm(x), rotation_matrix)
        return x + self.ff(self.ff_norm(x))


class ContinuousTransformer(nn.Module):
    def __init__(self, dtype, device, operations):
        super().__init__()
        self.project_in = operations.Linear(2304, 2048, bias=False, dtype=dtype, device=device)
        self.project_out = operations.Linear(2048, 128, bias=False, dtype=dtype, device=device)
        self.rotary_pos_emb = RotaryEmbedding(32, dtype, device)
        self.layers = nn.ModuleList([
            TransformerBlock(2048, 64, 8192, dtype, device, operations)
            for _ in range(36)
        ])

    def forward(self, x, timestep_embedding):
        x = self.project_in(x)
        x = torch.cat((timestep_embedding.unsqueeze(1), x), dim=1)
        rotation_matrix = self.rotary_pos_emb.forward_from_seq_len(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, rotation_matrix)
        return self.project_out(x[:, 1:])


class DiffusionTransformer(nn.Module):
    def __init__(self, dtype, device, operations):
        super().__init__()
        self.transformer = ContinuousTransformer(dtype, device, operations)
        self.timestep_features = FourierFeatures(1, 256, dtype, device)
        self.to_timestep_embed = nn.Sequential(
            operations.Linear(256, 2048, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(2048, 2048, dtype=dtype, device=device),
        )
        self.preprocess_conv = operations.Conv1d(2304, 2304, 1, bias=False, dtype=dtype, device=device)
        self.postprocess_conv = operations.Conv1d(128, 128, 1, bias=False, dtype=dtype, device=device)

    def forward(self, x, timestep, condition):
        full = torch.cat((x, torch.zeros_like(x), condition), dim=1)
        full = self.preprocess_conv(full) + full
        timestep_features = self.timestep_features(timestep[:, None]).to(dtype=x.dtype)
        timestep_embedding = self.to_timestep_embed(timestep_features)
        out = self.transformer(full.transpose(1, 2), timestep_embedding).transpose(1, 2)
        return self.postprocess_conv(out) + out


class MiniMaxMusic3DiT(nn.Module):
    def __init__(self, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.latent_conditioners = nn.Sequential(
            operations.Conv1d(4096, 2048, kernel_size=3, padding=1, dtype=dtype, device=device)
        )
        self.diffusion_transformer = DiffusionTransformer(dtype, device, operations)
        self.cond_layer_logits = nn.Parameter(torch.empty(8, dtype=dtype, device=device))
        self.cond_layer_scale = nn.Parameter(torch.empty(1, dtype=dtype, device=device))

    def aligned_condition(self, hidden):
        frames = hidden.shape[1]
        hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], 8, 4096, frames)
        weights = torch.softmax(comfy.ops.cast_to_input(self.cond_layer_logits, hidden), dim=0)
        hidden = torch.einsum("blht,l->bht", hidden, weights)
        hidden = comfy.ops.cast_to_input(self.cond_layer_scale, hidden) * hidden
        condition = self.latent_conditioners(hidden)
        return torch.nn.functional.interpolate(condition, size=latent_length(frames), mode="nearest")

    def forward(self, x, timestep, context, conditioning_scale, **kwargs):
        condition = self.aligned_condition(context)
        condition = condition * conditioning_scale[:, :1, :1]
        if condition.shape[-1] < x.shape[-1]:
            condition = torch.nn.functional.pad(condition, (0, x.shape[-1] - condition.shape[-1]))
        else:
            condition = condition[..., :x.shape[-1]]
        window = latent_length(MAX_CONDITION_FRAMES)
        if x.shape[-1] <= window:
            return -self.diffusion_transformer(x, timestep, condition)

        output = torch.zeros_like(x)
        count = torch.zeros((1, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
        hop = latent_length(CONDITION_HOP_FRAMES)
        start = 0
        while start < x.shape[-1]:
            end = min(start + window, x.shape[-1])
            output[..., start:end] -= self.diffusion_transformer(x[..., start:end], timestep, condition[..., start:end])
            count[..., start:end] += 1
            if end == x.shape[-1]:
                break
            start += hop
        return output / count
