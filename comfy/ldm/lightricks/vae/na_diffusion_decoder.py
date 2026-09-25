"""LTX 2.4 diffusion video VAE decoder (NADiffusionDecoder).

Port of the reference ``DiffusionVideoDecoder`` without the NATTEN dependency:
``natten.na3d`` is replaced by ``comfy_kitchen.na3d``, which reproduces
NATTEN's semantics (window of exactly ``kernel_size`` per query, shifted
inward at grid boundaries, dilation 1) and dispatches cuda/triton/eager per
device and dtype (the eager backend covers CPU and fp32).

Stages 1-4 deterministically upsample the latent into a context volume via
NA transformer blocks + linear pixel-shuffle upsamples. Stage 5 runs
``DiffusionNABlock``s that denoise patchified noised pixels ``x_t`` guided by
that context through AdaLN-Zero scale/shift. The 2.4 checkpoint is single-step
``x0``: one forward pass yields the pixels directly, no Euler loop.

State dict keys match the shipped checkpoints directly (fused ``attn.qkv``,
``t_embedder.mlp.{0,2}``, ``shared_adaln.proj``); no rename pass is needed.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import comfy.model_management

from comfy.ldm.lightricks.model import get_timestep_embedding
from .causal_video_autoencoder import Encoder, processor

import comfy_kitchen

# Token chunk for the SwiGLU MLP (bounds the [chunk, hidden] workspace).
MLP_TOKEN_CHUNK = 65536


def rms_norm(x, weight, eps=1e-6):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=eps)
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def patchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b c (f p) (h q) (w r) -> b (c p r q) f h w", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


# --- Absolute per-axis RoPE (matches ltx-core rope.py numerics) ---

def default_rope_dim_split(head_dim):
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim, base=10000.0, device=None):
    out_device = device
    if not comfy.model_management.supports_fp64(device):
        device = torch.device("cpu")

    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    return (1.0 / torch.pow(torch.tensor(float(base), dtype=torch.float64, device=device), exponents)).to(dtype=torch.float32, device=out_device)


def _rope_tables(lengths, inv_freqs, device):
    """Precompute per-axis fp32 cos/sin tables for global 0-based positions."""
    tables = []
    for length, inv in zip(lengths, inv_freqs):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        ang = pos[:, None] * inv[None, :]
        tables.append((ang.cos(), ang.sin()))
    return tables


def _rope_matrices_slice(tables, t0, t1, h, w):
    """Per-token rotation matrices ``(1, ts*h*w, 1, hd/2, 2, 2)`` fp32 for
    ``comfy_kitchen.rms_rope_`` (interleaved-pair convention), covering global
    frames ``[t0, t1)`` of the axis-factorized tables."""
    parts = []
    for (c, s), sl in zip(tables, (slice(t0, t1), slice(None), slice(None))):
        c, s = c[sl], s[sl]
        parts.append(torch.stack([c, -s, s, c], dim=-1).reshape(c.shape[0], 1, 1, c.shape[1], 2, 2))
    ts = t1 - t0
    freqs = torch.cat([
        parts[0].expand(ts, h, w, -1, 2, 2),
        parts[1].transpose(0, 1).expand(ts, h, w, -1, 2, 2),
        parts[2].movedim(0, 2).expand(ts, h, w, -1, 2, 2),
    ], dim=3)
    return freqs.reshape(1, ts * h * w, 1, -1, 2, 2)


class NeighborhoodAttention3D(nn.Module):
    """QKV (fused, matching checkpoint keys) + q/k RMSNorm + abs RoPE + NA."""

    def __init__(self, dim, kernel_size, head_dim=64, rope_base=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim ** -0.5
        self.rope_split = default_rope_dim_split(head_dim)
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, pre=None, add_to=None):
        """``pre`` (per-token norm/modulate) is applied slice-wise so the full
        pre-attention tensor is never materialized; ``add_to`` streams the
        output projection into it in place (residual add) and returns it.
        Both bound peak memory without changing results."""
        batch, t, h, w, _ = x.shape
        inv_freqs = tuple(rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
        tables = _rope_tables((t, h, w), inv_freqs, x.device)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q = torch.empty(shape, dtype=x.dtype, device=x.device)
        k = torch.empty(shape, dtype=x.dtype, device=x.device)
        v = torch.empty(shape, dtype=x.dtype, device=x.device)
        q_weight = (self.q_norm.weight.detach() * self.scale).to(x.dtype)  # scale commutes with the rotation
        k_weight = self.k_norm.weight.detach().to(x.dtype)
        chunk = max(1, (2 ** 25) // max(h * w * self.dim, 1))
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
            qc, kc, vc = self.qkv(sl).chunk(3, dim=-1)
            cshape = (batch, t1 - t0, h, w, self.num_heads, self.head_dim)
            q[:, t0:t1] = qc.reshape(cshape)
            k[:, t0:t1] = kc.reshape(cshape)
            v[:, t0:t1] = vc.reshape(cshape)
            freqs = _rope_matrices_slice(tables, t0, t1, h, w)
            nt = (t1 - t0) * h * w
            for b in range(batch):
                comfy_kitchen.rms_rope_(
                    q[b, t0:t1].view(1, nt, self.num_heads, self.head_dim),
                    k[b, t0:t1].view(1, nt, self.num_heads, self.head_dim),
                    freqs, q_weight, k_weight)
        out = comfy_kitchen.na3d(q, k, v, list(self.kernel_size), None, 1.0)
        del q, k, v
        out = out.reshape(batch, t, h, w, self.dim)
        res = add_to if add_to is not None else torch.empty_like(out)
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            if add_to is not None:
                res[:, t0:t1] += self.proj(out[:, t0:t1])
            else:
                res[:, t0:t1] = self.proj(out[:, t0:t1])
        return res


class SwiGLU(nn.Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, chunked over tokens to bound the
    ``[chunk, hidden]`` workspace."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pre=None, add_to=None):
        """``pre``/``add_to`` as in ``NeighborhoodAttention3D.forward``."""
        _, t, h, w, _ = x.shape
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        out = add_to if add_to is not None else torch.empty_like(x)
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
            y = self.w_down(F.silu(self.w_gate(sl)) * self.w_up(sl))
            if add_to is not None:
                out[:, t0:t1] += y
            else:
                out[:, t0:t1] = y
        return out


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds."""

    def __init__(self, dim, kernel_size, head_dim=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x):
        x = self.attn(x, pre=self.norm1, add_to=x)
        return self.mlp(x, pre=self.norm2, add_to=x)


def modulate(x, scale, shift):
    return x * (1.0 + scale) + shift


class AdaLNZero(nn.Module):
    """``t_emb`` -> 7 (scale/shift/gate) chunks; gate slots unused (folded at export)."""

    NUM_CHUNKS = 7

    def __init__(self, dim, t_emb_dim):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb):
        h = self.proj(F.silu(t_emb))
        return tuple(c[:, None, None, None, :] for c in h.chunk(self.NUM_CHUNKS, dim=-1))


class DiffusionNABlock(nn.Module):
    """NA + SwiGLU with shared AdaLN-Zero scale/shift (ungated residuals)."""

    def __init__(self, dim, kernel_size, context_channels, head_dim=64, mlp_ratio=4.0):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x, latent_context, modulation):
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        chunk = max(1, MLP_TOKEN_CHUNK // max(x.shape[2] * x.shape[3], 1))
        for t0 in range(0, x.shape[1], chunk):
            x[:, t0:t0 + chunk] += self.context_proj(latent_context[:, t0:t0 + chunk])
        x = self.attn(x, pre=lambda s: modulate(self.norm1(s), scale_msa, shift_msa), add_to=x)
        return self.mlp(x, pre=lambda s: modulate(self.norm2(s), scale_mlp, shift_mlp), add_to=x)


class LinearPixelShuffleUpsample(nn.Module):
    """Linear channel-expand, then channels-last pixel shuffle."""

    def __init__(self, in_channels, stride, out_channels_reduction_factor=1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, x, drop_leading_frame=True):
        batch, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        out = torch.empty((batch, t * p1, h * p2, w * p3, self.out_channels), dtype=x.dtype, device=x.device)
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            out[:, t0 * p1:t1 * p1] = rearrange(
                self.proj(x[:, t0:t1]), "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
                p1=p1, p2=p2, p3=p3,
            )
        if p1 == 2 and drop_leading_frame:
            # The causal temporal pixel-shuffle duplicates the leading frame.
            out = out[:, 1:]
        return out


class TimestepEmbedder(nn.Module):
    """Sinusoidal(256) -> MLP. ``mlp.{0,2}`` naming matches the checkpoint."""

    def __init__(self, t_emb_dim=384, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, t_emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim, bias=True),
        )

    def forward(self, timestep, dtype):
        emb = get_timestep_embedding(timestep.flatten(), self.freq_dim, flip_sin_to_cos=True,
                                     downscale_freq_shift=0, scale=1)
        return self.mlp(emb.to(dtype))


class NADiffusionDecoder(nn.Module):
    """Stages 1-4 (deterministic NA upsample) + stage-5 diffusion blocks.

    Input latent must already be un-normalized (the wrapper applies
    ``per_channel_statistics.un_normalize``, same as the conv VAE path).
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        head_dim=64,
        stage_channels=(2048, 1024, 512, 512, 256),
        stage_depths=(4, 6, 4, 2, 8),
        stage_kernels=((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)),
        upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
        stage5_kernel=(11, 11, 11),
        t_emb_dim=384,
        default_num_inference_steps=1,
        timestep_scale_multiplier=1000.0,
        model_output_type="x0",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps),
            persistent=False,
        )
        self.temporal_upscale = math.prod(s[0] for s, _ in upsamples)
        self.spatial_upscale = math.prod(s[1] for s, _ in upsamples) * patch_size
        # NATTEN-style last-frame border mitigation: replicate the last latent
        # frame through stages 1-4, crop the appendix off the context after.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(len(stage_channels) - 1):
            c = stage_channels[stage_i]
            self.det_stages.append(nn.ModuleList(
                [NABlock(c, stage_kernels[stage_i], head_dim=head_dim) for _ in range(stage_depths[stage_i])]
            ))
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(LinearPixelShuffleUpsample(c, stride, out_channels_reduction_factor=reduction))

        self.t_embedder = TimestepEmbedder(t_emb_dim=t_emb_dim)

        c5 = stage_channels[-1]
        self.context_channels = c5
        noised_pixel_channels = out_channels * (patch_size ** 2)
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim)
        self.diff_blocks = nn.ModuleList([
            DiffusionNABlock(c5, stage5_kernel, context_channels=c5, head_dim=head_dim)
            for _ in range(stage_depths[-1])
        ])
        self.norm_out = RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)

    def forward_pre_diffusion(self, z, drop_leading_frame=True, pad_trailing=True):
        """Stages 1-4: latent -> stage-5 context, channels-last.

        ``drop_leading_frame`` must be True only when ``z`` contains the
        latent's true temporal origin (t=0); tiled callers decoding a later
        temporal chunk pass False (the duplicate leading frame belongs solely
        to the origin chunk). ``pad_trailing`` only for chunks containing the
        latent's last frame."""
        n = self.trailing_pad_latent_frames if pad_trailing else 0
        if n > 0:
            z = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
        x = z.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
            x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        if n > 0:
            x = x[:, :-(n * self.temporal_upscale)]
        return x

    def forward_diff_step(self, context, x_t, t):
        x = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(x.permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, context, modulation)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward(self, z, generator=None, drop_leading_frame=True, pad_trailing=True):
        context = self.forward_pre_diffusion(z, drop_leading_frame=drop_leading_frame, pad_trailing=pad_trailing)
        batch, t5, h5, w5, _ = context.shape
        pixel_shape = (batch, self.out_channels, t5, h5 * self.patch_size, w5 * self.patch_size)
        x_t = torch.randn(pixel_shape, dtype=z.dtype, device=z.device, generator=generator)

        timesteps = self.default_inference_timesteps.to(z.device)
        num_steps = timesteps.shape[0]
        for i in range(num_steps):
            t_now = timesteps[i].expand(batch)
            model_out = self.forward_diff_step(context, x_t, t_now)
            if self.model_output_type == "x0":
                x0 = model_out
                if i == num_steps - 1:
                    return x0
                velocity = (x_t.float() - x0.float()) / timesteps[i]
            else:  # "v"
                velocity = model_out.float()
                if i == num_steps - 1:
                    return (x_t.float() - timesteps[i] * velocity).to(z.dtype)
            t_next = timesteps[i + 1] if i + 1 < num_steps else torch.zeros_like(timesteps[i])
            x_t = (x_t.float() - (timesteps[i] - t_next) * velocity).to(z.dtype)
        return x_t


LTX_24_VAE_CONFIG = {
    "_class_name": "CausalDiffusionVAE",
    "dims": 3,
    "model_output_type": "x0",
    "encoder": {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 128,
        "blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
            ["compress_all_res", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
        ],
        "patch_size": 4,
        "latent_log_var": "constant",
        "norm_layer": "pixel_norm",
        "base_channels": 128,
        "spatial_padding_mode": "zeros",
    },
    "decoder": {
        "in_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "head_dim": 64,
        "stage_channels": [2048, 1024, 512, 512, 256],
        "stage_depths": [4, 6, 4, 2, 8],
        "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
        "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
        "stage5_kernel": [11, 11, 11],
        "timestep_scale_multiplier": 1000.0,
        "default_num_inference_steps": 1,
    },
}


class CausalDiffusionVAE(nn.Module):
    """LTX 2.4 video VAE: conv encoder (shared with the 2.0 arch) + NA
    diffusion decoder. Interface mirrors ``causal_video_autoencoder.VideoVAE``.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = LTX_24_VAE_CONFIG
        self.config = config
        enc = config.get("encoder", LTX_24_VAE_CONFIG["encoder"])
        dec = config.get("decoder", LTX_24_VAE_CONFIG["decoder"])
        dec_defaults = LTX_24_VAE_CONFIG["decoder"]

        self.encoder = Encoder(
            dims=enc.get("dims", 3),
            in_channels=enc.get("in_channels", 3),
            out_channels=enc.get("out_channels", 128),
            blocks=enc.get("blocks", LTX_24_VAE_CONFIG["encoder"]["blocks"]),
            patch_size=enc.get("patch_size", 4),
            latent_log_var=enc.get("latent_log_var", "constant"),
            norm_layer=enc.get("norm_layer", "pixel_norm"),
            spatial_padding_mode=enc.get("spatial_padding_mode", "zeros"),
            base_channels=enc.get("base_channels", 128),
        )

        self.decoder = NADiffusionDecoder(
            in_channels=dec.get("in_channels", 128),
            out_channels=dec.get("out_channels", 3),
            patch_size=dec.get("patch_size", 4),
            head_dim=dec.get("head_dim", 64),
            stage_channels=tuple(dec.get("stage_channels", dec_defaults["stage_channels"])),
            stage_depths=tuple(dec.get("stage_depths", dec_defaults["stage_depths"])),
            stage_kernels=tuple(tuple(k) for k in dec.get("stage_kernels", dec_defaults["stage_kernels"])),
            upsamples=tuple((tuple(s), r) for s, r in dec.get("upsamples", dec_defaults["upsamples"])),
            stage5_kernel=tuple(dec.get("stage5_kernel", dec_defaults["stage5_kernel"])),
            t_emb_dim=dec.get("t_emb_dim", 384),
            default_num_inference_steps=dec.get("default_num_inference_steps", 1),
            timestep_scale_multiplier=dec.get("timestep_scale_multiplier", 1000.0),
            model_output_type=config.get("model_output_type", "x0"),
        )

        self.per_channel_statistics = processor()

    def encode(self, x, device=None):
        x = x[:, :, :max(1, 1 + ((x.shape[2] - 1) // 8) * 8), :, :]
        means, logvar = torch.chunk(self.encoder(x, device=device), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode(self, x):
        # Fixed-seed noise so decodes are reproducible TODO: expose?
        generator = torch.Generator(device=x.device)
        generator.manual_seed(0)
        return self.decoder(self.per_channel_statistics.un_normalize(x), generator=generator)
