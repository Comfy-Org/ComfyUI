# MiniMax H3 audio VAE: DAC-lineage waveform encoder + BigVGAN decoder.
# Weight-norm parametrizations are folded into plain conv weights, so this
# module uses ordinary ops.Conv1d / ops.ConvTranspose1d and loads the converted
# checkpoint (plain "*.weight" tensors) with strict=True.
#
# Lineage / licenses of the reference implementation:
#   DAC encoder:      descript-audio-codec (MIT)
#   BigVGAN decoder:  NVIDIA BigVGAN (MIT), adapted from hifi-gan (MIT)
#   Alias-free ops:   junjun3518/alias-free-torch (Apache-2.0), julius (MIT)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init


# Snake activations

def snake(x, alpha, beta):
    # x + 1/beta * sin^2(alpha * x)
    t = torch.sin(alpha * x)
    return t.mul_(t).mul_((beta + 1e-9).reciprocal()).add_(x)


class Snake1d(nn.Module):
    """Snake activation with per-channel alpha (encoder side)."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        alpha = comfy.ops.cast_to_input(self.alpha, x)
        return snake(x, alpha, alpha)


class SnakeBeta(nn.Module):
    """SnakeBeta := x + 1/beta * sin^2(alpha * x); alpha/beta stored in log scale."""

    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        alpha = torch.exp(comfy.ops.cast_to_input(self.alpha, x)).view(1, -1, 1)
        beta = torch.exp(comfy.ops.cast_to_input(self.beta, x)).view(1, -1, 1)
        return snake(x, alpha, beta)


# Alias-free (anti-aliased) activation: kaiser-windowed sinc resampling

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    # returns filter [1, 1, kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # kaiser window design
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize filter to have sum = 1, otherwise there is a small leakage of
    # the constant component in the input signal.
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size),
        )

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = F.conv_transpose1d(x, comfy.ops.cast_to_input(self.filter.expand(C, -1, -1), x), stride=self.stride, groups=C).mul_(self.ratio)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, comfy.ops.cast_to_input(self.filter.expand(C, -1, -1), x), stride=self.stride, groups=C)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    """upsample x2 -> pointwise activation -> downsample x2 (anti-aliased)."""

    def __init__(self, activation, up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# DAC encoder

class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            ops.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            ops.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y.add_(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            ops.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [ops.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block += [EncoderBlock(d_model, stride=stride)]
        block += [
            Snake1d(d_model),
            ops.Conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


# Attention projection (encoder posterior head)

class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = ops.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = ops.Linear(in_features, hidden_features)
        self.w1 = ops.Linear(in_features, hidden_features)
        self.w2 = ops.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul_(self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = ops.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        self.register_buffer("zero_k_bias", torch.empty(in_dim))
        self.proj = ops.Linear(out_dim, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        weight, _, offload_stream = comfy.ops.cast_bias_weight(self.qkv, x, offloadable=True)
        qkv = F.linear(x, weight=weight, bias=comfy.ops.cast_to_input(torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)), x))
        comfy.ops.uncast_bias_weight(self.qkv, weight, None, offload_stream)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        # mean over heads then pool down to the latent width (in_dim >> out_dim)
        x = comfy.ops.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.adaptive_avg_pool1d(torch.mean(x, dim=1), self.out_dim)
        return self.proj(x)


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = ops.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = ops.Linear(in_dim, out_dim)
        self.norm3 = ops.LayerNorm(in_dim)

        self.norm2 = ops.LayerNorm(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden_dim)

    def forward(self, x):
        # x: [B, T, in_dim]
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return x.add_(self.mlp(self.norm2(x)))


# BigVGAN decoder

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                ops.Conv1d(channels, channels, kernel_size, stride=1, dilation=d, padding=get_padding(kernel_size, d))
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                ops.Conv1d(channels, channels, kernel_size, stride=1, dilation=1, padding=get_padding(kernel_size, 1))
                for _ in range(len(dilation))
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [Activation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt.add_(x)
        return x


class BigVGAN(nn.Module):
    """BigVGAN vocoder (MiniMax H3 32 kHz configuration).

    use_bias_at_final=False, use_tanh_at_final=False (output clamped to [-1, 1]).
    """

    def __init__(
        self,
        num_mels=2048,
        upsample_initial_channel=1024,
        upsample_rates=(5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = ops.Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        ops.ConvTranspose1d(
                            upsample_initial_channel // (2 ** i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d))

        self.activation_post = Activation1d(activation=SnakeBeta(ch))
        self.conv_post = ops.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs.div_(self.num_kernels)

        x = self.activation_post(x)
        return self.conv_post(x).clamp_(-1.0, 1.0)


# Top-level VAE

class MiniMaxH3AudioVAE(nn.Module):
    """MiniMax H3 stereo audio VAE at 32 kHz.

    Latents are [B, 32, 2, T]: 32 channels, 2 stereo channels, T frames at
    40 latent frames per second (800 audio samples per latent frame). The
    stereo channels are processed independently by the mono encoder/decoder.
    Latents are normalized with the stored per-channel latents_mean/std.
    """

    def __init__(
        self,
        encoder_dim=64,
        encoder_rates=(2, 4, 4, 5, 5),
        latent_dim=2048,
        decoder_dim=1024,
        vae_latent_channels=32,
    ):
        super().__init__()
        self.sample_rate = 32000

        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r
        self.samples_per_latent = self.hop_length  # 800
        self.latents_per_second = self.sample_rate // self.hop_length  # 40
        self.output_sample_rate = self.sample_rate  # read by LTXVAudioVAEDecode

        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)

        self.mean_proj = ops.Conv1d(vae_latent_channels, vae_latent_channels, 1)
        # logs_proj exists in the checkpoint but is unused at inference
        # (encode returns the posterior mean, no sampling).
        self.logs_proj = ops.Conv1d(vae_latent_channels, vae_latent_channels, 1)

        self.dec_in_proj = ops.Conv1d(vae_latent_channels, latent_dim, 1)
        self.decoder = BigVGAN(num_mels=latent_dim, upsample_initial_channel=decoder_dim)

        self.register_buffer("latents_mean", torch.empty(vae_latent_channels))
        self.register_buffer("latents_std", torch.empty(vae_latent_channels))

    def decode(self, z):
        """Decode normalized latents [B, 32, 2, T] to stereo waveforms [B, 2, L] at 32 kHz."""
        b, c, s, t = z.shape
        z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
        mean = self.latents_mean.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        z = z * std + mean
        x = self.dec_in_proj(z)
        x = self.decoder(x)  # [b * s, 1, L], already clamped to [-1, 1]
        return x.reshape(b, s, -1)

    def encode(self, waveform):
        """Encode stereo waveforms [B, 2, L] at 32 kHz (in [-1, 1]) to normalized latents [B, 32, 2, T].

        L is right-padded with zeros to a multiple of 800 samples; the returned
        posterior mean is used directly (no sampling).
        """
        b, s, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        waveform = F.pad(waveform, (0, right_pad))
        x = waveform.reshape(b * s, 1, -1)
        x = self.encoder(x)  # [b * s, latent_dim, T]
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)  # [b * s, 32, T]
        z = self.mean_proj(x)
        mean = self.latents_mean.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        std = self.latents_std.view(1, -1, 1).to(device=z.device, dtype=z.dtype)
        z = (z - mean) / std
        return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)
