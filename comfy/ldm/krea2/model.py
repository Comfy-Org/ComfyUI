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
import comfy.utils
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
        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)

        if "block_index" in transformer_options and "attn1_patch" in transformer_patches:
            for p in transformer_patches["attn1_patch"]:
                out = p(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
                q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
                freqs, mask = out.get("pe", freqs), out.get("attn_mask", mask)

        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                         transformer_options=transformer_options)

        if "block_index" in transformer_options and "attn1_output_patch" in transformer_patches:
            for p in transformer_patches["attn1_output_patch"]:
                out = p(out, extra_options)

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

    def forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        if timestep_zero_index is not None:
            bs = x.shape[0]
            ref_prescale = prescale[bs:]
            ref_preshift = preshift[bs:]
            ref_pregate = pregate[bs:]
            ref_postscale = postscale[bs:]
            ref_postshift = postshift[bs:]
            ref_postgate = postgate[bs:]
            prescale = prescale[:bs]
            preshift = preshift[:bs]
            pregate = pregate[:bs]
            postscale = postscale[:bs]
            postshift = postshift[:bs]
            postgate = postgate[:bs]

            pre = self.prenorm(x)
            pre[:, :timestep_zero_index].mul_(1 + prescale).add_(preshift)
            pre[:, timestep_zero_index:].mul_(1 + ref_prescale).add_(ref_preshift)
            attn = self.attn(pre, freqs, mask, transformer_options=transformer_options)
            del pre
            attn[:, :timestep_zero_index].mul_(pregate)
            attn[:, timestep_zero_index:].mul_(ref_pregate)
            x = x + attn
            del attn

            post = self.postnorm(x)
            post[:, :timestep_zero_index].mul_(1 + postscale).add_(postshift)
            post[:, timestep_zero_index:].mul_(1 + ref_postscale).add_(ref_postshift)
            mlp = self.mlp(post)
            del post
            mlp[:, :timestep_zero_index].mul_(postgate)
            mlp[:, timestep_zero_index:].mul_(ref_postgate)
            x = x + mlp
            del mlp
            return x

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
                 txtheads=20, txtkvheads=20, default_ref_method=None, image_model=None,
                 device=None, dtype=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers
        self.default_ref_method = default_ref_method

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

    def forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

    def process_img(self, x, index=0):
        patch = self.patch
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
        h, w = x.shape[-2] // patch, x.shape[-1] // patch
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        img_ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        img_ids[..., 0] = index
        img_ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        img_ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return img, img_ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w

    def _forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        bs, _, h_orig, w_orig = x.shape
        patch = self.patch

        # context arrives as (B, seq, txtlayers*txtdim); reshape to (B, txtlayers, seq, txtdim).
        context = self._unpack_context(context)

        img, imgpos, h_, w_ = self.process_img(x)
        img_tokens = img.shape[1]
        timestep_zero_index = None
        ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
        if ref_method is not None and ref_latents is not None and len(ref_latents) > 0:
            ref_tokens = []
            ref_pos = []
            ref_num_tokens = []
            for index, ref in enumerate(ref_latents, 1):
                if ref.ndim == 5:
                    rb, rc, rt, rh5, rw5 = ref.shape
                    ref = ref.reshape(rb * rt, rc, rh5, rw5)
                ref = comfy.utils.repeat_to_batch_size(ref, bs)
                kontext, kontext_ids, _, _ = self.process_img(ref, index=index)
                ref_tokens.append(kontext)
                ref_pos.append(kontext_ids)
                ref_num_tokens.append(kontext.shape[1])
            img = torch.cat([img] + ref_tokens, dim=1)
            imgpos = torch.cat([imgpos] + ref_pos, dim=1)
            del ref_tokens, ref_pos
            if ref_method == "index_timestep_zero":
                timestep_zero_index = img_tokens
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)
        if timestep_zero_index is not None:
            t0 = self.tmlp(timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype))
            tvec = torch.cat((tvec, self.tproj(t0)), dim=0)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen = context.shape[1]
        device = context.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

        patches = transformer_options.get("patches", {})
        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": context, "img_ids": imgpos, "txt_ids": txtpos, "transformer_options": transformer_options})
                img, context = out["img"], out["txt"]
                imgpos, txtpos = out["img_ids"], out["txt_ids"]

        combined = torch.cat((context, img), dim=1)
        del context, img
        if timestep_zero_index is not None:
            timestep_zero_index += txtlen

        # Position ids: text at 0, image at (0, h_idx, w_idx).
        pos = torch.cat((txtpos, imgpos), dim=1)
        del txtpos, imgpos

        freqs = self.pe_embedder(pos)
        del pos

        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txtlen, combined.shape[1]]
        for i, block in enumerate(self.blocks):
            transformer_options["block_index"] = i
            combined = block(combined, tvec, freqs, None, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)

        final = self.last(combined, t)
        del combined
        out = final[:, txtlen:txtlen + img_tokens, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :h_orig, :w_orig]  # crop padding back off
        if temporal:
            out = out.reshape(b5, t5, self.channels, h_orig, w_orig).movedim(1, 2)
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
