"""Krea 2 (K2) — single-stream MMDiT.

Text tokens produced by a Qwen3-VL-4B 12-layer ``txtfusion`` adapter and patchified image tokens are
concatenated into one sequence and run through ``layers`` shared transformer blocks with
AdaLN-single modulation, GQA + per-head QK-norm + sigmoid-gated attention, SwiGLU MLP, and 3-axis RoPE.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.ldm.common_dit
import comfy.utils
from comfy.ldm.flux.layers import EmbedND, timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_masked


class RMSNorm(nn.Module):
    """RMSNorm with the reference ``(1 + scale)`` weight convention (scale stored zero-centered)."""

    def __init__(self, features: int, eps: float = 1e-5, device=None, dtype=None, operations=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features, device=device, dtype=dtype))

    def fp32_weight(self, x):
        return comfy.model_management.cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.fp32_weight(x).to(x.dtype), eps=self.eps)


def modulated_norm(x, norm, scale, shift):
    if comfy.model_management.in_training:
        return (1 + scale) * norm(x) + shift
    # the norm weight is folded in fp32 into the scale of the fused rmsnorm(x) * (1 + scale) + shift
    scale = (norm.fp32_weight(x) * (1 + scale.float()) - 1).to(x.dtype)
    return comfy.quant_ops.ck.rms_adaln(x, scale, shift, norm.eps)


def modulate(x, norm, mod, i, ref):
    # mod[i] is the scale and mod[i + 1] the shift; tokens from ref[0] on take the t=0 modulation ref[1]
    if ref is None:
        return modulated_norm(x, norm, mod[i], mod[i + 1])
    s, ref_mod = ref
    return torch.cat((modulated_norm(x[:, :s], norm, mod[i], mod[i + 1]), modulated_norm(x[:, s:], norm, ref_mod[i], ref_mod[i + 1])), dim=1)


def gated_residual(x, y, gate):
    # in place inside the memory compiler region; out of place keeps x for backward when training
    if comfy.model_management.in_training:
        return torch.addcmul(x, y, gate)
    return x.addcmul_(y, gate)


def residual(x, y, mod, i, ref):
    if ref is None:
        return gated_residual(x, y, mod[i])
    s, ref_mod = ref
    if comfy.model_management.in_training:
        return torch.cat((gated_residual(x[:, :s], y[:, :s], mod[i]), gated_residual(x[:, s:], y[:, s:], ref_mod[i])), dim=1)
    gated_residual(x[:, :s], y[:, :s], mod[i])
    gated_residual(x[:, s:], y[:, s:], ref_mod[i])
    return x


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
        self.comfy_attention = ComfyAttention()
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
        # (B, L, H, D) throughout, the rope table is laid out (B, L, 1, ...) to match
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.heads, self.headdim)
        k = self.wk(x).view(B, L, self.kvheads, self.headdim)
        v = self.wv(x).view(B, L, self.kvheads, self.headdim)
        gate = self.gate(x)
        transformer_patches = transformer_options.get("patches", {})
        patches = transformer_patches.get("attn1_patch", []) if "block_index" in transformer_options else []

        if freqs is not None and not patches and not comfy.model_management.in_training:
            qnorm, knorm = self.qknorm.qnorm, self.qknorm.knorm
            q, k = comfy.quant_ops.ck.rms_rope_(q, k, freqs, qnorm.fp32_weight(q), knorm.fp32_weight(k), qnorm.eps)
        else:
            q, k = self.qknorm(q, k)
            if patches:
                # patches see (B, H, L, D) before rope and the (B, 1, L, ...) rope table
                extra_options = transformer_options.copy()
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                freqs = None if freqs is None else freqs.transpose(1, 2)
                for p in patches:
                    out = p(q, k, v, pe=freqs, attn_mask=mask, extra_options=extra_options)
                    q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
                    freqs, mask = out.get("pe", freqs), out.get("attn_mask", mask)
                q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
                freqs = None if freqs is None else freqs.transpose(1, 2)
            if freqs is not None:
                q, k = apply_rope(q, k, freqs)

        # expanded here: block-sparse, sol and custom attention overrides reshape k/v by the query heads
        k, v = comfy.ops.repeat_kv_for_gqa(k, v, self.heads, 2)
        q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v.flatten(2))
        out = optimized_attention_masked(q, k, v, self.heads, mask=mask, preferred_attention=self.comfy_attention, transformer_options=transformer_options)

        if "block_index" in transformer_options and "attn1_output_patch" in transformer_patches:
            extra_options = transformer_options.copy()
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
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.layerwise_blocks) + list(self.refiner_blocks), x.device, transformer_options)
        for block in self.layerwise_blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = self.projector(x.unflatten(0, (b, l)).transpose(2, 3)).squeeze(-1)
        for block in self.refiner_blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x, mask=mask, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
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
        # prescale, preshift, pregate, postscale, postshift, postgate
        mod = self.mod(vec)
        ref = None
        if timestep_zero_index is not None:
            # the reference tokens take the t=0 modulation stacked after the batch in vec
            bs = x.shape[0]
            ref = (timestep_zero_index, [m[bs:] for m in mod])
            mod = [m[:bs] for m in mod]
        x = residual(x, self.attn(modulate(x, self.prenorm, mod, 0, ref), freqs, mask, transformer_options=transformer_options), mod, 2, ref)
        return residual(x, self.mlp(modulate(x, self.postnorm, mod, 3, ref)), mod, 5, ref)


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm = RMSNorm(features, device=device, dtype=dtype, operations=operations)
        self.linear = operations.Linear(features, patch * patch * channels, bias=True, device=device, dtype=dtype)
        self.modulation = SimpleModulation(features, device=device, dtype=dtype, operations=operations)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear(modulated_norm(x, self.norm, scale, shift))


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
        b, c = x.shape[:2]
        h, w = x.shape[-2] // patch, x.shape[-1] // patch
        img = x.reshape(b, c, h, patch, w, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, h * w, c * patch * patch)

        img_ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        img_ids[..., 0] = index
        img_ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        img_ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return img, img_ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w

    def preprocess_text(self, context, transformer_options={}):
        # the txtfusion adapter and txtmlp depend on the conditioning alone, extra_conds runs them once per sampling
        context = self.txtfusion(self._unpack_context(context), mask=None, transformer_options=transformer_options)
        return self.txtmlp(context)

    def _forward(self, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options={}, txt_states=None, **kwargs):
        transformer_options = transformer_options.copy()
        temporal = x.ndim == 5
        if temporal:
            b5, c5, t5, h5, w5 = x.shape
            x = x.reshape(b5 * t5, c5, h5, w5)
        bs, _, h_orig, w_orig = x.shape
        patch = self.patch

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

        # every step when extra_conds could not precompute it (training, hooks)
        context = self.preprocess_text(context, transformer_options) if txt_states is None else txt_states

        txtlen = context.shape[1]
        device = context.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

        patches = transformer_options.get("patches", {})
        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": context, "img_ids": imgpos, "txt_ids": txtpos, "transformer_options": transformer_options})
                img, context = out["img"], out["txt"]
                imgpos, txtpos = out["img_ids"], out["txt_ids"]

        # allocated outside the memory compiler region, the blocks update it in place
        combined = torch.cat((context, img), dim=1)
        del context, img
        if timestep_zero_index is not None:
            timestep_zero_index += txtlen

        # Position ids: text at 0, image at (0, h_idx, w_idx).
        pos = torch.cat((txtpos, imgpos), dim=1)
        del txtpos, imgpos

        freqs = self.pe_embedder(pos).transpose(1, 2)
        del pos

        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txtlen, combined.shape[1]]
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(device)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, combined.dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            combined = block(combined, tvec, freqs, None, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()

        out = self.last(combined[:, txtlen:txtlen + img_tokens], t)
        del combined
        out = out.reshape(bs, h_, w_, self.channels, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(bs, self.channels, h_ * patch, w_ * patch)
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
