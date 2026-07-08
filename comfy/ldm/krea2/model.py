"""Krea 2 (K2) — single-stream MMDiT.

Text tokens produced by a Qwen3-VL-4B 12-layer ``txtfusion`` adapter and patchified image tokens are
concatenated into one sequence and run through ``layers`` shared transformer blocks with
AdaLN-single modulation, GQA + per-head QK-norm + sigmoid-gated attention, SwiGLU MLP, and 3-axis RoPE.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.model_management
import comfy.patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import EmbedND, timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (scale stored zero-centered)."""

    def __init__(self, features: int, eps: float = 1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = comfy.model_management.cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.qnorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)
        self.knorm = RMSNorm(dim, device=device, dtype=dtype, operations=operations)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128,
                 device=None, dtype=None, operations=None):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = operations.Linear(features, mlpdim, bias=bias, device=device, dtype=dtype)
        self.up = operations.Linear(features, mlpdim, bias=bias, device=device, dtype=dtype)
        self.down = operations.Linear(mlpdim, features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False,
                 device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = operations.Linear(dim, self.headdim * self.heads, bias=bias, device=device, dtype=dtype)
        self.wk = operations.Linear(dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype)
        self.wv = operations.Linear(dim, self.headdim * self.kvheads, bias=bias, device=device, dtype=dtype)
        self.gate = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)
        self.qknorm = QKNorm(self.headdim, device=device, dtype=dtype, operations=operations)
        self.wo = operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                         transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + comfy.model_management.cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.lin = nn.Parameter(torch.empty(6 * dim, device=device, dtype=dtype))

    def forward(self, vec):
        out = vec + comfy.model_management.cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.prenorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.postnorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, device=device, dtype=dtype, operations=operations)
        self.mlp = SwiGLU(features, multiplier, bias, device=device, dtype=dtype, operations=operations)

    def forward(self, x, mask=None, transformer_options={}):
        x = x + self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(2)
        ])
        self.projector = operations.Linear(num_txt_layers, 1, bias=False, device=device, dtype=dtype)
        self.refiner_blocks = nn.ModuleList([
            TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(2)
        ])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.mod = DoubleSharedModulation(features, device=device, dtype=dtype, operations=operations)
        self.prenorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.postnorm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias, device=device, dtype=dtype, operations=operations)
        self.mlp = SwiGLU(features, multiplier, bias, device=device, dtype=dtype, operations=operations)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask, transformer_options=transformer_options)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.linear = operations.Linear(features, patch * patch * channels, bias=True, device=device, dtype=dtype)
        self.modulation = SimpleModulation(features, device=device, dtype=dtype, operations=operations)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(self, features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
                 layers=28, patch=2, channels=16, bias=False, theta=1e3, txtlayers=12,
                 txtheads=20, txtkvheads=20, image_model=None,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = operations.Linear(channels * patch ** 2, features, bias=True, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            SingleStreamBlock(features, heads, multiplier, bias, kvheads, device=device, dtype=dtype, operations=operations)
            for _ in range(layers)
        ])
        self.tmlp = nn.Sequential(
            operations.Linear(tdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads,
                                               device=device, dtype=dtype, operations=operations)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim, device=device, dtype=dtype, operations=operations),
            operations.Linear(txtdim, features, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features, device=device, dtype=dtype),
        )
        self.last = LastLayer(features, patch, channels, device=device, dtype=dtype, operations=operations)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            operations.Linear(features, features * 6, device=device, dtype=dtype),
        )

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    def _forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch
        # Pad the latent up to a multiple of patch (as Flux/Lumina/QwenImage do); crop back at the end.
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        # context arrives as (B, seq, txtlayers*txtdim); reshape to (B, txtlayers, seq, txtdim).
        context = self._unpack_context(context)

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Position ids: text at 0, image at (0, h_idx, w_idx).
        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
        pos = torch.cat((txtpos, imgpos), dim=1)

        freqs = self.pe_embedder(pos)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

        final = self.last(combined, t)
        out = final[:, txtlen:txtlen + imglen, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]  # crop padding back off
        if temporal:
            out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
        return out

    def _unpack_context(self, context):
        # context: (B, seq, txtlayers*txtdim) -> (B, seq, txtlayers, txtdim).
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}={self.txtlayers * self.txtdim} "
                f"features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}. "
                f"Load the text encoder with CLIPLoader type 'krea2'."
            )
        return context.reshape(b, seq, self.txtlayers, self.txtdim)
