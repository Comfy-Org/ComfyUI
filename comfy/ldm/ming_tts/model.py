# Adapted from https://github.com/inclusionAI/Ming-3.0-tts (MIT, Copyright 2025 inclusionAI).
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_prefetch
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
from comfy.text_encoders import llama
from .tiny import TinyBackbone


class Qwen(llama.Llama2_):
    def __init__(self, config, device, dtype, operations):
        config = SimpleNamespace(**(vars(llama.Qwen25_3BConfig()) | {"use_sliding_window": False, "sliding_window": None, "rope_frequency_dtype": "float32"} | config))
        config.transformer_type = "llama"
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.qkv_bias = True
        config.rms_norm_add = False
        config.mlp_activation = "silu"
        config.q_norm = config.k_norm = config.rope_scale = config.rope_dims = None
        config.lm_head = False
        config.fixed_kv = config.graph_dynamic_vbar_blocks = config.prefetch_dynamic_vbars = True
        super().__init__(config, device=device, dtype=dtype, ops=operations)
        self.rope_dtype = dtype
        self.sliding_window = config.sliding_window if config.use_sliding_window else None

    def compute_freqs_cis(self, position_ids, device):
        # Preserve the released runtime's frequency rounding, including dense BF16 RoPE.
        with torch.autocast(device_type=device.type, enabled=False):
            exponent = torch.arange(0, self.config.head_dim, 2, device=device).float() / self.config.head_dim
            inv_freq = (1.0 / self.config.rope_theta ** exponent).to(getattr(torch, self.config.rope_frequency_dtype)).float()
            phase = position_ids.float()[..., None] * inv_freq
            cos, sin = phase.cos().to(self.rope_dtype), phase.sin().to(self.rope_dtype)
        return torch.cat((cos, cos), dim=-1)[:, None], sin[:, None], -sin[:, None]

    def forward(self, x, embeds=None, **kwargs):
        if self.sliding_window is not None:
            positions = torch.arange(embeds.shape[1], device=embeds.device)
            # Every released codec layer attends only its preceding 64 frames.
            kwargs["attention_mask"] = (positions[None] > positions[:, None] - self.sliding_window)[None]
        return super().forward(x, embeds=embeds, **kwargs)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.register_buffer("inv_freq", torch.empty(dim // 2, device=device, dtype=torch.float32))

    def forward(self, x):
        freq = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None]
        freq = freq * comfy.ops.cast_to_input(self.inv_freq, freq)
        cos, sin = freq.cos(), freq.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, x.shape[1], -1, 2, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.to_q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.to_out = nn.Sequential(operations.Linear(dim, dim, device=device, dtype=dtype), nn.Identity())
        self.comfy_attention = ComfyAttention()

    def forward(self, x, rope):
        shape = (x.shape[0], x.shape[1], self.heads, -1)
        q = self.to_q(x).reshape(shape).transpose(1, 2)
        k = self.to_k(x).reshape(shape).transpose(1, 2)
        v = self.to_v(x).reshape(shape).transpose(1, 2)
        q, k = apply_rope(q, k, rope)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        x = optimized_attention(
            q, k, v,
            self.heads, skip_reshape=True, preferred_attention=self.comfy_attention,
        )
        return self.to_out(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult, device, dtype, operations):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Sequential(operations.Linear(dim, int(dim * mult), device=device, dtype=dtype), nn.GELU(approximate="tanh")),
            nn.Identity(),
            operations.Linear(int(dim * mult), dim, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, dim, heads, mult, device, dtype, operations):
        super().__init__()
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.attn = Attention(dim, heads, device, dtype, operations)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.mlp = FeedForward(dim, mult, device, dtype, operations)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim, device, dtype, operations):
        super().__init__()
        self.norm_final = operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.linear = operations.Linear(dim, out_dim, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, device, dtype, operations):
        super().__init__()
        self.time_mlp = nn.Sequential(
            operations.Linear(256, dim, device=device, dtype=dtype), nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )

    def forward(self, t):
        freq = torch.exp(torch.arange(128, device=t.device, dtype=torch.float32) * (-math.log(10000) / 127))
        phase = 1000 * t[:, None] * freq[None]
        return self.time_mlp(torch.cat((phase.sin(), phase.cos()), dim=-1).to(t.dtype))


class PatchTransformer(nn.Module):
    def __init__(self, config, latent_dim, llm_dim, diffusion, device, dtype, operations):
        super().__init__()
        dim = config["hidden_size"]
        self.diffusion = diffusion
        self.hidden_size = dim
        self.head_dim = dim // config["num_heads"]
        self.x_embedder = operations.Linear(latent_dim, dim, device=device, dtype=dtype)
        self.rotary_embed = RotaryEmbedding(dim // config["num_heads"], device)
        self.blocks = nn.ModuleList([
            Block(dim, config["num_heads"], config["mlp_ratio"], device, dtype, operations)
            for _ in range(config["depth"])
        ])
        if diffusion:
            self.t_embedder = TimestepEmbedder(dim, device, dtype, operations)
            self.c_embedder = nn.Module()
            self.c_embedder.cond_embedder = operations.Linear(llm_dim, dim, device=device, dtype=dtype)
        else:
            self.word_embedder = operations.Embedding(1, dim, device=device, dtype=dtype)
        self.final_layer = FinalLayer(dim, latent_dim if diffusion else llm_dim, device, dtype, operations)

    def init_decode_buffers(self, batch, length, device, dtype):
        return (
            torch.empty((batch, length, self.hidden_size), device=device, dtype=dtype),
            torch.empty((1, 1, length, self.head_dim // 2, 2, 2), device=device, dtype=torch.float32),
        )

    def forward(self, x, t=None, c=None, history=None, decode_buffers=None):
        if self.diffusion:
            conditioning = self.t_embedder(t)[:, None] + self.c_embedder.cond_embedder(c)
            x = torch.cat((conditioning, self.x_embedder(history), self.x_embedder(x)), dim=1)
        else:
            x = self.x_embedder(x)
            cls = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device), out_dtype=x.dtype)
            x = torch.cat((cls, x), dim=1)
        rope = self.rotary_embed(x)
        if decode_buffers is not None:
            decode_buffers[0].copy_(x)
            decode_buffers[1].copy_(rope)
            x, rope = decode_buffers
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, {"prefetch_dynamic_vbars": True})
        scope = "ming_flow" if self.diffusion else "ming_aggregator"
        for block in self.blocks:
            def core():
                x.copy_(block(x, rope))
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, core=core, enable_graph=decode_buffers is not None, malloc_scope=scope)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope=scope)
        x = self.final_layer(x)
        return x if self.diffusion else x[:, :1]

    def get_dynamic_vram__units(self):
        return list(self.blocks), []


def sample_flow(model, cond, history, noise, sde_noise, cfg=2.0, sigma=0.25, temperature=0.0, tiny=False, decode_buffers=None):
    # The released sampler uses ten EPSS steps, including SDE draws at temperature zero.
    t = torch.tensor([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], device=noise.device, dtype=noise.dtype) / 32
    t = t - (torch.cos(torch.pi / 2 * t) - 1 + t)
    c = torch.cat((cond, torch.zeros_like(cond)))
    history = torch.cat((history, history))
    options = {"decode_buffers": decode_buffers} if decode_buffers is not None else {}
    x = noise
    for step in range(10):
        pred, null = model(torch.cat((x, x)), t[step].expand(2 * x.shape[0]), c, history, **options)[:, -x.shape[1]:].chunk(2)
        dt = t[step + 1] - t[step]
        flow = null + (pred - null) * cfg if tiny else pred + (pred - null) * cfg
        if tiny and cfg < 1e-5:
            flow = null
        x = x + flow * dt
        # Tiny's reference returns the last Euler endpoint before its final SDE shift.
        if not tiny or step < 9:
            x = x + sigma * temperature ** 0.5 * dt.abs().sqrt() * sde_noise[step]
    return x


class ISTFT(nn.Module):
    def __init__(self, hop, device, dtype):
        super().__init__()
        self.hop = hop
        self.register_buffer("window", torch.empty(4 * hop, device=device, dtype=dtype))

    def forward(self, real, imag):
        size = 4 * self.hop
        # Real inverse DFT avoids complex dtypes and device-specific FFT restrictions.
        phase = torch.arange(size, device=real.device, dtype=torch.float32)[:, None] * torch.arange(size // 2 + 1, device=real.device, dtype=torch.float32)
        phase = ((phase + size // 2).remainder(size) - size // 2) * (2 * math.pi / size)
        cosine, sine = phase.cos(), phase.sin()
        cosine[:, (0, -1)] *= 0.5
        sine[:, (0, -1)] = 0
        basis = torch.cat((cosine, -sine), dim=1) * (2 / size)
        with torch.autocast(device_type=real.device.type, enabled=False):
            frames = basis @ torch.cat((real, imag), dim=1)
        window = comfy.ops.cast_to_input(self.window, frames)
        frames = frames * window[None, :, None]
        length = (real.shape[-1] + 3) * self.hop
        pad = 3 * self.hop // 2
        audio = F.fold(frames, (1, length), (1, 4 * self.hop), stride=(1, self.hop))[:, 0, 0, pad:-pad]
        envelope = F.fold(
            window.square()[None, :, None].expand(1, -1, real.shape[-1]),
            (1, length), (1, 4 * self.hop), stride=(1, self.hop),
        )[0, 0, 0, pad:-pad]
        return (audio / envelope)[:, None]


class ISTFTHead(nn.Module):
    def __init__(self, dim, hop, device, dtype, operations):
        super().__init__()
        self.out = operations.Linear(dim, 4 * hop + 2, device=device, dtype=dtype)
        self.istft = ISTFT(hop, device, dtype)

    def forward(self, x):
        mag, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        # Promote only the spectral products for float32 inverse-DFT accumulation.
        mag = mag.exp().clamp(max=100).float()
        return self.istft(mag * phase.cos(), mag * phase.sin())


class AudioEncoder(nn.Module):
    def __init__(self, config, patch, device, dtype, operations):
        super().__init__()
        backbone = config["backbone"]
        dim = backbone["hidden_size"]
        self.input_dim = config["input_dim"]
        self.hop_size = config["hop_size"]
        self.patch_size = patch
        self.encoder = Qwen(backbone, device, dtype, operations)
        self.fc1 = operations.Linear(self.input_dim, dim, bias=False, device=device, dtype=dtype)
        self.fc2 = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.fc3 = operations.Linear(dim, 2 * config["latent_dim"], device=device, dtype=dtype)
        self.norm = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.aggregator = Qwen(backbone | {"num_hidden_layers": 4}, device, dtype, operations)
        self.cls_embed = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))

    def forward(self, waveform, generator):
        frames = math.ceil(waveform.shape[-1] / self.hop_size)
        waveform = F.pad(waveform, (0, (frames - 1) * self.hop_size + self.input_dim - waveform.shape[-1]))
        x = self.fc2(self.fc1(waveform.unfold(-1, self.input_dim, self.hop_size)))
        x = self.encoder(None, embeds=x)[0]
        x = F.pad(x, (0, 0, 0, -x.shape[1] % self.patch_size))
        b, _, dim = x.shape
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, comfy.ops.cast_to_input(self.cls_embed, x).expand(x.shape[0], -1, -1)), dim=1).reshape(b, -1, dim)
        x = self.aggregator(None, embeds=x)[0].reshape(b, -1, self.patch_size + 1, dim)[:, :, -1]
        mean, scale = self.fc3(x).transpose(1, 2).chunk(2, dim=1)
        std = F.softplus(scale) + 1e-4
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return (mean + std * noise).transpose(1, 2)


class AudioDecoder(nn.Module):
    def __init__(self, config, patch, device, dtype, operations):
        super().__init__()
        dim = config["backbone"]["hidden_size"]
        self.patch_size = patch
        self.fc1 = operations.Linear(config["latent_dim"], dim, device=device, dtype=dtype)
        self.decoder = Qwen(config["backbone"], device, dtype, operations)
        self.head = ISTFTHead(dim, config["output_dim"], device, dtype, operations)

    def forward(self, latent):
        x = F.interpolate(self.fc1(latent).transpose(1, 2), scale_factor=self.patch_size, mode="linear", align_corners=False).transpose(1, 2)
        return self.head(self.decoder(None, embeds=x)[0])


class MingTextModel(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.config = config
        self.tiny = config["model_type"] == "lingv3_tts"
        llm = config["llm_config"]
        audio = config["audio_config"] if self.tiny else config["audio_tokenizer_config"]
        flow = config["flow_config"] if self.tiny else config["ditar_config"]
        latent = audio["enc_kwargs"]["latent_dim"]
        self.latent_dim = latent
        self.patch_size = config["tts_config"]["patch_size"] if self.tiny else flow["patch_size"]
        self.history_size = config["tts_config"]["history_size"] if self.tiny else flow["history_patch_size"]
        self.sample_rate = audio["sample_rate"]
        self.model = nn.Module()
        self.model.model = (TinyBackbone if self.tiny else Qwen)(llm, device, dtype, operations)
        self.linear_proj_audio = PatchTransformer(config["aggregator_config"], latent, llm["hidden_size"], False, device, dtype, operations)
        self.stop_head = operations.Linear(llm["hidden_size"], 2, device=device, dtype=dtype)
        self.spk_head = operations.Linear(192, llm["hidden_size"], device=device, dtype=dtype)

    def get_dynamic_vram__units(self):
        return list(self.model.model.layers) + list(self.linear_proj_audio.blocks), []


class MingAudioVAE(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        self.sample_rate = config["sample_rate"]
        self.encoder = AudioEncoder(config["enc_kwargs"], config["patch_size"], device, dtype, operations)
        self.decoder = AudioDecoder(config["dec_kwargs"], config["patch_size"], device, dtype, operations)

    def encode(self, waveform):
        align = self.encoder.hop_size * self.encoder.patch_size
        waveform = F.pad(waveform.mean(1), (0, -waveform.shape[-1] % align))
        return self.encoder(waveform, None).transpose(1, 2)

    def decode(self, latents):
        return self.decoder(latents.transpose(1, 2))

    def get_dynamic_vram__units(self):
        units = []
        for model in (self.encoder.encoder, self.encoder.aggregator, self.decoder.decoder):
            units.extend(model.layers)
        return units, []


class MingFlowModel(PatchTransformer):
    def __init__(self, ming_config, device, dtype, operations, **kwargs):
        tiny = ming_config["model_type"] == "lingv3_tts"
        audio = ming_config["audio_config" if tiny else "audio_tokenizer_config"]
        flow = ming_config["flow_config" if tiny else "ditar_config"]
        super().__init__(flow, audio["enc_kwargs"]["latent_dim"], ming_config["llm_config"]["hidden_size"], True, device, dtype, operations)
        self.dtype = dtype
