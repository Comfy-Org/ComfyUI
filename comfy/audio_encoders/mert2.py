"""MERT-v2 ConvNeXt/Conformer audio encoder."""

import torch
from torch import nn
from torch.nn import functional as F

import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention_for_device


class MelFrontend(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.register_buffer("mel_mean", torch.empty(128, device=device, dtype=torch.float32))
        self.register_buffer("mel_std", torch.empty(128, device=device, dtype=torch.float32))
        self.spectrogram = nn.Module()
        self.spectrogram.register_buffer("window", torch.empty(2048, device=device, dtype=torch.float32))
        self.mel_scale = nn.Module()
        self.mel_scale.register_buffer("fb", torch.empty(1025, 128, device=device, dtype=torch.float32))

    def forward(self, waveform):
        window = comfy.ops.cast_to_input(self.spectrogram.window, waveform)
        spectrum = torch.stft(waveform, n_fft=2048, hop_length=240, win_length=2048,
                              window=window, return_complex=True).abs().square()
        mel = spectrum.transpose(-1, -2) @ comfy.ops.cast_to_input(self.mel_scale.fb, waveform)
        mel = 10.0 * mel.clamp_min(1e-10).log10()
        mean = comfy.ops.cast_to_input(self.mel_mean, waveform)
        std = comfy.ops.cast_to_input(self.mel_std, waveform)
        return (mel[:, :-1] - mean) / std.clamp_min(1e-5)


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))

    def forward(self, x):
        magnitude = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        normalized = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        weight = comfy.ops.cast_to_input(self.weight, x)
        bias = comfy.ops.cast_to_input(self.bias, x)
        return weight * (x * normalized) + bias + x


class ConvNextLayer(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.depthwise_block = nn.Sequential(
            Transpose(), operations.Conv1d(dim, dim, 7, padding=3, groups=dim, device=device, dtype=dtype), Transpose(),
        )
        self.pointwise_block = nn.Sequential(
            operations.LayerNorm(dim, eps=1e-6, device=device, dtype=dtype),
            operations.Linear(dim, 4 * dim, device=device, dtype=dtype), nn.GELU(),
            GlobalResponseNorm(4 * dim, device=device, dtype=dtype),
            operations.Linear(4 * dim, dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        return x + self.pointwise_block(self.depthwise_block(x))


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, device=None, dtype=None, operations=None):
        super().__init__()
        self.resampling_layer = nn.Identity()
        if in_channels != out_channels or stride > 1:
            self.resampling_layer = nn.Sequential(
                operations.LayerNorm(in_channels, eps=1e-6, device=device, dtype=dtype), Transpose(),
                operations.Conv1d(in_channels, out_channels, 2, stride=stride, device=device, dtype=dtype), Transpose(),
            )
        self.convnext_layers = nn.Sequential(*[
            ConvNextLayer(out_channels, device=device, dtype=dtype, operations=operations) for _ in range(depth)
        ])

    def forward(self, x):
        return self.convnext_layers(self.resampling_layer(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.query_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.key_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.value_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.out_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, positions, attention):
        shape = (x.shape[0], x.shape[1], self.heads, -1)
        q = self.query_proj(x).reshape(shape).transpose(1, 2)
        k = self.key_proj(x).reshape(shape).transpose(1, 2)
        v = self.value_proj(x).reshape(shape).transpose(1, 2)
        q, k = comfy.quant_ops.ck.apply_rope_split_half(q, k, positions)
        return self.out_proj(attention(q, k, v, self.heads, skip_reshape=True))


class FeedForward(nn.Module):
    def __init__(self, dim, intermediate, device=None, dtype=None, operations=None):
        super().__init__()
        self.w_1 = operations.Linear(dim, intermediate, device=device, dtype=dtype)
        self.w_2 = operations.Linear(intermediate, dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.w_2(F.gelu(self.w_1(x)))


class ConvolutionModule(nn.Module):
    def __init__(self, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.conv_block = nn.Sequential(
            Transpose(), operations.Conv1d(dim, dim * 2, 1, bias=False, device=device, dtype=dtype), nn.GLU(dim=1),
            operations.Conv1d(dim, dim, 31, padding=15, groups=dim, bias=False, device=device, dtype=dtype),
            nn.Sequential(Transpose(), operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype), Transpose()),
            nn.GELU(), operations.Conv1d(dim, dim, 1, bias=False, device=device, dtype=dtype), Transpose(),
        )

    def forward(self, x):
        return self.conv_block(self.layer_norm(x))


class ConformerBlock(nn.Module):
    def __init__(self, dim, intermediate, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.ffn1_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.ffn1 = FeedForward(dim, intermediate, device=device, dtype=dtype, operations=operations)
        self.attn_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.attn = Attention(dim, heads, device=device, dtype=dtype, operations=operations)
        self.conv_module = ConvolutionModule(dim, device=device, dtype=dtype, operations=operations)
        self.ffn2_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)
        self.ffn2 = FeedForward(dim, intermediate, device=device, dtype=dtype, operations=operations)
        self.final_layer_norm = operations.LayerNorm(dim, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x, positions, attention):
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        x = x + self.attn(self.attn_layer_norm(x), positions, attention)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        return self.final_layer_norm(x)


class MERT2(nn.Module):
    def __init__(self, dim=1024, intermediate=4096, heads=16, layers=24, channels=(128, 512, 1024),
                 depths=(3, 4, 5), device=None, dtype=None, operations=None):
        super().__init__()
        self.head_dim = dim // heads
        self.feature_extractor = MelFrontend(device=device)
        channels = (128, *channels)
        self.subsampling_module = nn.Sequential(*[
            ConvNextBlock(channels[i], channels[i + 1], (1, 2, 2)[i], depths[i], device=device, dtype=dtype, operations=operations)
            for i in range(3)
        ])
        self.layers = nn.ModuleList([
            ConformerBlock(dim, intermediate, heads, device=device, dtype=dtype, operations=operations) for _ in range(layers)
        ])

    def position_embeddings(self, x):
        inverse = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
        angles = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None] * inverse
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, x.shape[1], -1, 2, 2).float()

    def forward(self, mel, layer_weight, output_hidden_states=False):
        x = self.subsampling_module(mel)
        weights = comfy.ops.cast_to_input(layer_weight, x).softmax(dim=0)
        mixed = x * weights[0]
        states = [x] if output_hidden_states else None
        positions = self.position_embeddings(x)
        attention = optimized_attention_for_device(x.device)
        for weight, layer in zip(weights[1:], self.layers):
            x = layer(x, positions, attention)
            mixed = mixed + x * weight
            if output_hidden_states:
                states.append(x)
        return mixed, states
