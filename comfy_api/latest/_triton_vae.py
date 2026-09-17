"""Triton VAE decode/encode kernels: fused norm+SiLU and optional int8 convs.

Supported (auto-detected): Wan (RMSNorm), KL/Flux2/SDXL/SD1.5 (GroupNorm)
and LTXV/LTX2 (PixelNorm). int8 covers Wan, flux-family KL and LTX decoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import triton
    import triton.language as tl
except ImportError as e:
    raise ImportError("PatchTritonVAE requires triton (pip install triton, or triton-windows)") from e

from comfy.ldm.wan.vae import RMS_norm, CausalConv3d
from comfy.ldm.modules.diffusionmodules.model import ResnetBlock
from comfy.ldm.lightricks.vae.pixel_norm import PixelNorm
from comfy.ldm.lightricks.vae.causal_conv3d import CausalConv3d as LTXCausalConv3d
from comfy.ldm.lightricks.vae.causal_video_autoencoder import ResnetBlock3D as LTXResnetBlock3D, Encoder as LTXEncoder, Decoder as LTXDecoder

import os
import copy
import logging

import comfy.ops

PROFILE = os.environ.get("KJNODES_TRITON_PROFILE", "") == "1"
CL3D = torch.channels_last_3d

import comfy.model_patcher
from comfy.patcher_extension import CallbacksMP

@triton.jit
def _rms_silu_kernel(x_ptr, g_ptr, out_ptr, C, S, stride_b, eps, scale, rcount, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                     HAS_GAMMA: tl.constexpr = True, SILU: tl.constexpr = True, EPS_INSIDE: tl.constexpr = False):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    # int64 offsets: C*S per batch can exceed 2^31 at high resolutions
    base = x_ptr + pid_b.to(tl.int64) * stride_b
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        ptrs = base + offs_c[:, None].to(tl.int64) * S + offs_s[None, :]
        v = tl.load(ptrs, mask=mask_c[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(v * v, axis=0)
    if EPS_INSIDE:  # pixel_norm: 1/sqrt(mean(x^2) + eps)
        inv = scale / tl.sqrt(acc * rcount + eps)
    else:
        inv = scale / tl.maximum(tl.sqrt(acc), eps)
    out_base = out_ptr + pid_b.to(tl.int64) * stride_b
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        m = mask_c[:, None] & mask_s[None, :]
        ptrs = base + offs_c[:, None].to(tl.int64) * S + offs_s[None, :]
        v = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
        y = v * inv[None, :]
        if HAS_GAMMA:
            g = tl.load(g_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            y = y * g[:, None]
        if SILU:
            y = y * tl.sigmoid(y)
        tl.store(out_base + offs_c[:, None].to(tl.int64) * S + offs_s[None, :], y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rms_silu_cl_kernel(x_ptr, g_ptr, out_ptr, amax_ptr, C, ROWS, eps, scale, rcount, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                        HAS_GAMMA: tl.constexpr = True, SILU: tl.constexpr = True, EPS_INSIDE: tl.constexpr = False,
                        AMAX: tl.constexpr = False):
    pid = tl.program_id(0)
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    mask_s = offs_s < ROWS
    mask_c = offs_c < C
    m = mask_s[:, None] & mask_c[None, :]
    # int64 offsets: ROWS*C (= numel) can exceed 2^31 at high resolutions
    ptrs = x_ptr + offs_s[:, None].to(tl.int64) * C + offs_c[None, :]
    v = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
    acc = tl.sum(v * v, axis=1)
    if EPS_INSIDE:
        inv = scale / tl.sqrt(acc * rcount + eps)
    else:
        inv = scale / tl.maximum(tl.sqrt(acc), eps)
    y = v * inv[:, None]
    if HAS_GAMMA:
        g = tl.load(g_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        y = y * g[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    if AMAX:
        # free global amax for downstream int8 quantization
        tl.atomic_max(amax_ptr, tl.max(tl.max(tl.abs(y), axis=1), axis=0))
    tl.store(out_ptr + offs_s[:, None].to(tl.int64) * C + offs_c[None, :], y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gn_stats_cl(x_ptr, sum_ptr, sumsq_ptr, S,
                 G: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    acc1 = tl.zeros([G], dtype=tl.float32)
    acc2 = tl.zeros([G], dtype=tl.float32)
    s0 = pid * BLOCK_S
    step = tl.num_programs(0) * BLOCK_S
    while s0 < S:
        offs_s = s0 + tl.arange(0, BLOCK_S)
        m = (offs_s < S)[:, None]
        # int64 offsets: (B*S)*C can exceed 2^31 at high resolutions
        ptr = x_ptr + (b.to(tl.int64) * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :]
        v = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
        v3 = tl.reshape(v, (BLOCK_S, G, BLOCK_C // G))
        acc1 += tl.sum(tl.sum(v3, axis=2), axis=0)
        acc2 += tl.sum(tl.sum(v3 * v3, axis=2), axis=0)
        s0 += step
    tl.atomic_add(sum_ptr + b * G + tl.arange(0, G), acc1)
    tl.atomic_add(sumsq_ptr + b * G + tl.arange(0, G), acc2)


@triton.jit
def _gn_silu_apply_cl(x_ptr, sum_ptr, sumsq_ptr, w_ptr, bias_ptr, out_ptr, amax_ptr, S, count, eps,
                      G: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr, SILU: tl.constexpr,
                      AMAX: tl.constexpr = False):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    CS: tl.constexpr = BLOCK_C // G
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    m = (offs_s < S)[:, None]
    ptr = x_ptr + (b.to(tl.int64) * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :]
    v = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
    mean_g = tl.load(sum_ptr + b * G + tl.arange(0, G)) / count
    var_g = tl.load(sumsq_ptr + b * G + tl.arange(0, G)) / count - mean_g * mean_g
    var_g = tl.maximum(var_g, 0.0)  # E[x²]-mean² can go slightly negative from fp32 cancellation
    rstd_g = 1.0 / tl.sqrt(var_g + eps)
    mean_c = tl.reshape(tl.broadcast_to(mean_g[:, None], (G, CS)), (BLOCK_C,))
    rstd_c = tl.reshape(tl.broadcast_to(rstd_g[:, None], (G, CS)), (BLOCK_C,))
    w = tl.load(w_ptr + offs_c).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c).to(tl.float32)
    y = (v - mean_c[None, :]) * rstd_c[None, :] * w[None, :] + bias[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    if AMAX:
        # free global amax for downstream int8 quantization
        masked = tl.where(m, tl.abs(y), 0.0)
        tl.atomic_max(amax_ptr, tl.max(tl.max(masked, axis=1), axis=0))
    tl.store(out_ptr + (b.to(tl.int64) * S + offs_s[:, None]) * BLOCK_C + offs_c[None, :],
             y.to(out_ptr.dtype.element_ty), mask=m)


# autotuned variants: benchmark configs once per shape key, cache the fastest
_rms_silu_kernel_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs, "BLOCK_C": bc}, num_warps=w)
             for bs in (128, 256, 512) for bc in (32, 64) for w in (4, 8)],
    key=["C", "S"])(_rms_silu_kernel)

_rms_silu_cl_kernel_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (4, 8, 16, 32, 64) for w in (4, 8)],
    key=["C", "ROWS"])(_rms_silu_cl_kernel)

_gn_stats_cl_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (8, 16, 32, 64) for w in (4, 8)],
    key=["S"], reset_to_zero=["sum_ptr", "sumsq_ptr"])(_gn_stats_cl)

_gn_silu_apply_cl_tuned = triton.autotune(
    configs=[triton.Config({"BLOCK_S": bs}, num_warps=w) for bs in (8, 16, 32, 64) for w in (4, 8)],
    key=["S", "count"])(_gn_silu_apply_cl)


def fused_rms_silu(x, gamma, scale, eps=1e-12, autotune=False, silu=True, eps_inside=False):
    B, C = x.shape[0], x.shape[1]
    S = x.numel() // (B * C)
    x = x.contiguous()
    out = torch.empty_like(x)
    flags = {"HAS_GAMMA": gamma is not None, "SILU": silu, "EPS_INSIDE": eps_inside}
    g = gamma if gamma is not None else x
    if autotune:
        grid = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), B)
        _rms_silu_kernel_tuned[grid](x, g, out, C, S, C * S, eps, scale, 1.0 / C, **flags)
    else:
        _rms_silu_kernel[(triton.cdiv(S, 256), B)](x, g, out, C, S, C * S, eps, scale, 1.0 / C,
                                                   BLOCK_S=256, BLOCK_C=32, num_warps=8, **flags)
    return out


def fused_rms_silu_cl(x, gamma, scale, eps=1e-12, autotune=False, silu=True, eps_inside=False, return_amax=False):
    C = x.shape[1]
    rows = x.numel() // C
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    flags = {"HAS_GAMMA": gamma is not None, "SILU": silu, "EPS_INSIDE": eps_inside, "AMAX": return_amax}
    g = gamma if gamma is not None else x
    amax = torch.zeros(1, device=x.device, dtype=torch.float32) if return_amax else out
    if autotune:
        grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_S"]),)
        _rms_silu_cl_kernel_tuned[grid](x, g, out, amax, C, rows, eps, scale, 1.0 / C, BLOCK_C=BLOCK_C, **flags)
    else:
        BLOCK_S = max(1, 4096 // BLOCK_C)
        _rms_silu_cl_kernel[(triton.cdiv(rows, BLOCK_S),)](x, g, out, amax, C, rows, eps, scale, 1.0 / C,
                                                           BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, num_warps=4, **flags)
    if return_amax:
        return out, amax[0]
    return out


def fused_gn_silu_cl(x, weight, bias, groups, eps, silu=True, autotune=False, return_amax=False):
    B, C, H, W = x.shape
    S = H * W
    sums = torch.zeros(2, B * groups, device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)
    count = S * (C // groups)
    amax = torch.zeros(1, device=x.device, dtype=torch.float32) if return_amax else out
    if autotune:
        # program count beyond ~1024 only adds same-address atomic contention
        grid_stats = lambda meta: (min(1024, triton.cdiv(S, meta["BLOCK_S"])), B)
        _gn_stats_cl_tuned[grid_stats](x, sums[0], sums[1], S, G=groups, BLOCK_C=C)
        grid_apply = lambda meta: (triton.cdiv(S, meta["BLOCK_S"]), B)
        _gn_silu_apply_cl_tuned[grid_apply](x, sums[0], sums[1], weight, bias, out, amax, S, count, eps,
                                            G=groups, BLOCK_C=C, SILU=silu, AMAX=return_amax)
    else:
        BLOCK_S = max(1, 8192 // C)
        nprog = min(1024, triton.cdiv(S, BLOCK_S))
        _gn_stats_cl[(nprog, B)](x, sums[0], sums[1], S, G=groups, BLOCK_S=BLOCK_S, BLOCK_C=C, num_warps=8)
        _gn_silu_apply_cl[(triton.cdiv(S, BLOCK_S), B)](x, sums[0], sums[1], weight, bias, out, amax, S, count, eps,
                                                        G=groups, BLOCK_S=BLOCK_S, BLOCK_C=C, SILU=silu, AMAX=return_amax, num_warps=8)
    if return_amax:
        return out, amax[0]
    return out


class FusedRMSSiLU(nn.Module):
    emit_amax = False  # lets the downstream int8 conv skip its amax pass

    def __init__(self, rms, autotune=False):
        super().__init__()
        self.gamma = rms.gamma
        self.scale = rms.scale
        self.autotune = autotune

    def forward(self, x):
        if not x.is_cuda:
            return F.silu(F.normalize(x, dim=1) * self.scale * self.gamma.to(x))
        gamma = self.gamma
        if gamma.device != x.device:  # lowvram partial load keeps params offloaded on cpu
            gamma = gamma.to(x.device)
        gamma = gamma.reshape(-1)
        if not gamma.is_contiguous():
            gamma = gamma.contiguous()
        if x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous():
            if self.emit_amax:
                out, amax = fused_rms_silu_cl(x, gamma, self.scale, autotune=self.autotune, return_amax=True)
                out._kj_amax = amax
                return out
            return fused_rms_silu_cl(x, gamma, self.scale, autotune=self.autotune)
        return fused_rms_silu(x, gamma, self.scale, autotune=self.autotune)


class FusedPixelNorm(nn.Module):
    def __init__(self, pn, silu=True, autotune=False):
        super().__init__()
        self.eps = pn.eps
        self.silu = silu
        self.autotune = autotune

    def forward(self, x):
        if x.is_cuda:
            if x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous():
                return fused_rms_silu_cl(x, None, 1.0, self.eps, autotune=self.autotune, silu=self.silu, eps_inside=True)
            return fused_rms_silu(x, None, 1.0, self.eps, autotune=self.autotune, silu=self.silu, eps_inside=True)
        out = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        if self.silu:
            out = F.silu(out)
        return out


class FusedGNSiLU(nn.Module):
    emit_amax = False  # lets the downstream int8 conv skip its amax pass

    def __init__(self, gn, silu=True, autotune=False):
        super().__init__()
        self.weight = gn.weight
        self.bias = gn.bias
        self.num_groups = gn.num_groups
        self.eps = gn.eps
        self.silu = silu
        self.autotune = autotune

    def forward(self, x):
        C = x.shape[1]
        G = self.num_groups
        weight, bias = self.weight, self.bias
        if weight.device != x.device:  # lowvram partial load keeps params offloaded on cpu
            weight = weight.to(x.device)
            bias = bias.to(x.device)
        if x.is_cuda and x.ndim == 4 and (C & (C - 1)) == 0 and (G & (G - 1)) == 0 and C % G == 0 \
                and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
            if self.emit_amax and self.silu:
                out, amax = fused_gn_silu_cl(x, weight, bias, self.num_groups, self.eps, self.silu,
                                             autotune=self.autotune, return_amax=True)
                out._kj_amax = amax
                return out
            return fused_gn_silu_cl(x, weight, bias, self.num_groups, self.eps, self.silu,
                                    autotune=self.autotune)
        out = F.group_norm(x, self.num_groups, weight, bias, self.eps)
        if self.silu:
            out = F.silu(out)
        return out


def convert_conv_layout(model, channels_last=True):
    for mod in model.modules():
        if isinstance(mod, nn.Conv3d):
            mod.to(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)
        elif isinstance(mod, nn.Conv2d):
            mod.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def build_object_patches(model, autotune=False):
    """Norm fusion object patches; matches already-fused modules for rebuilds."""
    patches = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Sequential):  # Wan-style RMS_norm+SiLU pairs
            for i in range(len(mod) - 1):
                if isinstance(mod[i], RMS_norm) and isinstance(mod[i + 1], nn.SiLU) \
                        and mod[i].gamma.ndim == 4 and mod[i].bias is None:
                    patches[f"{name}.{i}"] = FusedRMSSiLU(mod[i], autotune=autotune)
                    patches[f"{name}.{i + 1}"] = nn.Identity()
                elif isinstance(mod[i], FusedRMSSiLU) and isinstance(mod[i + 1], nn.Identity):
                    mod[i].autotune = autotune
                    patches[f"{name}.{i}"] = mod[i]
                    patches[f"{name}.{i + 1}"] = mod[i + 1]
        elif isinstance(mod, ResnetBlock) and not hasattr(mod, "temb_proj"):  # KL-style
            for norm_name in ("norm1", "norm2"):
                norm = getattr(mod, norm_name)
                if isinstance(norm, nn.GroupNorm):
                    patches[f"{name}.{norm_name}"] = FusedGNSiLU(norm, autotune=autotune)
                elif isinstance(norm, FusedGNSiLU):
                    norm.autotune = autotune
                    patches[f"{name}.{norm_name}"] = norm
            patches[f"{name}.swish"] = nn.Identity()
        elif name.endswith("norm_out"):  # KL encoder/decoder head, SiLU applied separately
            if isinstance(mod, nn.GroupNorm):
                patches[name] = FusedGNSiLU(mod, silu=False, autotune=autotune)
            elif isinstance(mod, FusedGNSiLU):
                mod.autotune = autotune
                patches[name] = mod
        elif isinstance(mod, LTXResnetBlock3D):
            # conditioned decoder blocks apply a scale-shift between norm and SiLU
            fuse_silu = not mod.timestep_conditioning
            for norm_name in ("norm1", "norm2"):
                norm = getattr(mod, norm_name)
                if isinstance(norm, PixelNorm) and norm.dim == 1:
                    patches[f"{name}.{norm_name}"] = FusedPixelNorm(norm, silu=fuse_silu, autotune=autotune)
                elif isinstance(norm, FusedPixelNorm):
                    norm.autotune = autotune
                    patches[f"{name}.{norm_name}"] = norm
            if fuse_silu and f"{name}.norm1" in patches:
                patches[f"{name}.non_linearity"] = nn.Identity()
        elif isinstance(mod, (LTXEncoder, LTXDecoder)):  # LTX head norm + conv_act
            fuse_silu = isinstance(mod, LTXEncoder) or not mod.timestep_conditioning
            if isinstance(mod.conv_norm_out, PixelNorm) and mod.conv_norm_out.dim == 1:
                patches[f"{name}.conv_norm_out"] = FusedPixelNorm(mod.conv_norm_out, silu=fuse_silu, autotune=autotune)
            elif isinstance(mod.conv_norm_out, FusedPixelNorm):
                mod.conv_norm_out.autotune = autotune
                patches[f"{name}.conv_norm_out"] = mod.conv_norm_out
            if fuse_silu and f"{name}.conv_norm_out" in patches:
                patches[f"{name}.conv_act"] = nn.Identity()
    return patches


# int8 (W8A8) convolutions: implicit GEMM on int8 tensor cores

@triton.jit
def _quant_pad_s8(x_ptr, c_ptr, out_ptr, amax_ptr, C, C_PAD, S_PAD, S_CACHE, NTOT, BLOCK: tl.constexpr):
    # one-pass padded int8 buffer: [zero frames | cache | payload], channels zero-extended to C_PAD
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NTOT
    inv = 127.0 / tl.maximum(tl.load(amax_ptr).to(tl.float32), 1e-8)
    s_idx = offs // C_PAD
    c = offs % C_PAD
    valid = m & (c < C)
    in_c = valid & (s_idx >= S_PAD) & (s_idx < S_PAD + S_CACHE)
    in_x = valid & (s_idx >= S_PAD + S_CACHE)
    v = tl.load(x_ptr + (s_idx - S_PAD - S_CACHE) * C + c, mask=in_x, other=0.0).to(tl.float32)
    v += tl.load(c_ptr + (s_idx - S_PAD) * C + c, mask=in_c, other=0.0).to(tl.float32)
    v = v * inv
    v = tl.minimum(tl.maximum(v + tl.where(v >= 0, 0.5, -0.5), -127.0), 127.0)
    tl.store(out_ptr + offs, v.to(tl.int8), mask=m)


@triton.jit
def _conv3d_s8(x_ptr, w_ptr, out_ptr, scale_ptr, bias_ptr,
               H, W, C, C_out, C3, M,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               HAS_BIAS: tl.constexpr):
    # the three dw taps of each (dt, dh) group are one contiguous 3C span in NDHWC
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < C_out

    w_o = offs_m % W
    h_o = (offs_m // W) % H
    t_o = offs_m // (W * H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for g in tl.static_range(9):
        dt = g // 3
        dh = g % 3
        t_i = t_o + dt
        h_i = h_o + dh - 1
        h_ok = mask_m & (h_i >= 0) & (h_i < H)
        base = ((t_i * H + h_i) * W + (w_o - 1)) * C
        for k0 in range(0, C3, BLOCK_K):
            dw = k0 // C  # padded C is a multiple of BLOCK_K, one dw tap per chunk
            m_ok = h_ok & (w_o + dw - 1 >= 0) & (w_o + dw - 1 < W)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + base[:, None] + offs_k[None, :], mask=m_ok[:, None], other=0)
            b = tl.load(w_ptr + (g * C3 + offs_k)[:, None] * C_out + offs_n[None, :],
                        mask=mask_n[None, :], other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    y = acc.to(tl.float32) * tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:
        y += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    tl.store(out_ptr + offs_m[:, None] * C_out + offs_n[None, :],
             y.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _conv2d_s8(x_ptr, w_ptr, out_ptr, scale_ptr, bias_ptr,
               H, W, C, C_out, C3, M,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               HAS_BIAS: tl.constexpr):
    # 2D (3x3, pad 1) variant of _conv3d_s8: 3 dh groups, batch-aware rows
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < C_out

    w_o = offs_m % W
    h_o = (offs_m // W) % H
    b = offs_m // (W * H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for g in tl.static_range(3):
        h_i = h_o + g - 1
        h_ok = mask_m & (h_i >= 0) & (h_i < H)
        base = ((b * H + h_i) * W + (w_o - 1)) * C
        for k0 in range(0, C3, BLOCK_K):
            dw = k0 // C
            m_ok = h_ok & (w_o + dw - 1 >= 0) & (w_o + dw - 1 < W)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + base[:, None] + offs_k[None, :], mask=m_ok[:, None], other=0)
            wb = tl.load(w_ptr + (g * C3 + offs_k)[:, None] * C_out + offs_n[None, :],
                         mask=mask_n[None, :], other=0)
            acc = tl.dot(a, wb, acc, out_dtype=tl.int32)

    y = acc.to(tl.float32) * tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    if HAS_BIAS:
        y += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    tl.store(out_ptr + offs_m[:, None] * C_out + offs_n[None, :],
             y.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _pack_int8_weight(weight, spatial_dims):
    sw = (weight.detach().abs().amax(dim=tuple(range(1, weight.ndim)), keepdim=False).clamp(min=1e-8).float() / 127.0)
    shape = [-1] + [1] * (weight.ndim - 1)
    qw = (weight.detach().float() / sw.reshape(shape)).round_().clamp_(-127, 127).to(torch.int8)
    c_in = weight.shape[1]
    # zero-pad channels to keep every K chunk dw-aligned
    c_pad = ((c_in + 63) // 64) * 64
    if c_pad != c_in:
        pad = [0, 0] * spatial_dims + [0, c_pad - c_in]
        qw = torch.nn.functional.pad(qw, pad)
    perm = tuple(range(2, weight.ndim)) + (1, 0)
    return qw.permute(*perm).reshape(-1, weight.shape[0]).contiguous(), sw, c_pad


class Int8CausalConv3d(CausalConv3d):
    """int8 Wan CausalConv3d; subclass keeps isinstance cache routing and shares the weight Parameter."""

    _logged = set()
    _prof = []

    def __init__(self, orig):
        super().__init__(orig.in_channels, orig.out_channels, orig.kernel_size,
                         stride=orig.stride, padding=1)
        self.weight = orig.weight
        self.bias = orig.bias
        qw, sw, self.int8_cpad = _pack_int8_weight(orig.weight, spatial_dims=3)
        self.register_buffer("int8_qw", qw, persistent=False)
        self.register_buffer("int8_sw", sw * (1.0 / 127.0), persistent=False)
        self._last_amax = None

    def forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
        if not x.is_cuda or x.shape[0] != 1:
            reason = "non-cuda input" if not x.is_cuda else f"batch size {x.shape[0]}"
            if reason not in Int8CausalConv3d._logged:
                Int8CausalConv3d._logged.add(reason)
                logging.info(f"PatchTritonVAE int8: falling back to bf16 conv ({reason})")
            return super().forward(x, cache_x=cache_x, cache_list=cache_list, cache_idx=cache_idx)
        if "active" not in Int8CausalConv3d._logged:
            Int8CausalConv3d._logged.add("active")
            logging.info("PatchTritonVAE int8: int8 conv kernel path active")
        if PROFILE:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = self._int8_forward(x, cache_x, cache_list, cache_idx)
            ev1.record()
            Int8CausalConv3d._prof.append((ev0, ev1))
            if len(Int8CausalConv3d._prof) >= 250:
                torch.cuda.synchronize()
                total = sum(a.elapsed_time(b) for a, b in Int8CausalConv3d._prof)
                logging.info(f"PatchTritonVAE int8 profile: {len(Int8CausalConv3d._prof)} conv calls, {total:.0f} ms total")
                Int8CausalConv3d._prof.clear()
            return out
        return self._int8_forward(x, cache_x, cache_list, cache_idx)

    def _int8_forward(self, x, cache_x, cache_list, cache_idx):
        if cache_list is not None:
            cache_x = cache_list[cache_idx]
            cache_list[cache_idx] = None

        amax = getattr(x, "_kj_amax", None)
        if not x.is_contiguous(memory_format=CL3D) or x.is_contiguous():
            x = x.contiguous(memory_format=CL3D)
        B, C, T, H, W = x.shape
        ct = 0 if cache_x is None else cache_x.shape[2]
        pad = max(0, 2 - ct)
        T_in = T + ct + pad

        if amax is None:
            amax = x.abs().amax()
        if ct and self._last_amax is not None:
            # cache frames are bounded by the previous payload amax
            amax = torch.maximum(amax, self._last_amax)
        self._last_amax = amax.detach()

        xr = x.permute(0, 2, 3, 4, 1).reshape(-1)
        if ct:
            if not cache_x.is_contiguous(memory_format=CL3D) or cache_x.is_contiguous():
                cache_x = cache_x.contiguous(memory_format=CL3D)
            cr = cache_x.permute(0, 2, 3, 4, 1).reshape(-1)
        else:
            cr = xr
        C_pad = self.int8_cpad
        hw = H * W
        n_tot = T_in * hw * C_pad
        qx = torch.empty(n_tot, device=x.device, dtype=torch.int8)
        _quant_pad_s8[(triton.cdiv(n_tot, 4096),)](xr, cr, qx, amax, C, C_pad, pad * hw, ct * hw, n_tot,
                                                   BLOCK=4096, num_warps=8)

        C_out = self.out_channels
        scale_vec = self.int8_sw * amax.float()
        bias = self.bias
        T_out = T_in - 2
        M = T_out * hw
        out = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 128), triton.cdiv(C_out, 64))
        _conv3d_s8[grid](qx, self.int8_qw, out, scale_vec,
                         bias.float() if bias is not None else scale_vec,
                         H, W, C_pad, C_out, 3 * C_pad, M,
                         BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                         HAS_BIAS=bias is not None, num_warps=4, num_stages=4)
        return out.view(1, T_out, H, W, C_out).permute(0, 4, 1, 2, 3)


class Int8Conv2d(comfy.ops.disable_weight_init.Conv2d):
    """int8 3x3 Conv2d for KL VAE ResnetBlocks; shares the weight Parameter."""

    def __init__(self, orig):
        super().__init__(orig.in_channels, orig.out_channels, orig.kernel_size,
                         stride=orig.stride, padding=orig.padding)
        self.weight = orig.weight
        self.bias = orig.bias
        qw, sw, self.int8_cpad = _pack_int8_weight(orig.weight, spatial_dims=2)
        self.register_buffer("int8_qw", qw, persistent=False)
        self.register_buffer("int8_sw", sw * (1.0 / 127.0), persistent=False)

    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        amax = getattr(x, "_kj_amax", None)
        if not x.is_contiguous(memory_format=torch.channels_last) or x.is_contiguous():
            x = x.contiguous(memory_format=torch.channels_last)
        B, C, H, W = x.shape
        if amax is None:
            amax = x.abs().amax()
        xr = x.permute(0, 2, 3, 1).reshape(-1)

        C_pad = self.int8_cpad
        n_tot = B * H * W * C_pad
        qx = torch.empty(n_tot, device=x.device, dtype=torch.int8)
        _quant_pad_s8[(triton.cdiv(n_tot, 4096),)](xr, xr, qx, amax, C, C_pad, 0, 0, n_tot,
                                                   BLOCK=4096, num_warps=8)

        C_out = self.out_channels
        scale_vec = self.int8_sw * amax.float()
        bias = self.bias
        M = B * H * W
        out = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 128), triton.cdiv(C_out, 64))
        _conv2d_s8[grid](qx, self.int8_qw, out, scale_vec,
                         bias.float() if bias is not None else scale_vec,
                         H, W, C_pad, C_out, 3 * C_pad, M,
                         BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                         HAS_BIAS=bias is not None, num_warps=4, num_stages=4)
        return out.view(B, H, W, C_out).permute(0, 3, 1, 2)


class Int8InnerConv3d(comfy.ops.disable_weight_init.Conv3d):
    """int8 replacement for the plain Conv3d inside the LTX CausalConv3d wrapper."""

    def __init__(self, orig):
        super().__init__(orig.in_channels, orig.out_channels, orig.kernel_size,
                         stride=orig.stride, padding=orig.padding)
        self.weight = orig.weight
        self.bias = orig.bias
        qw, sw, self.int8_cpad = _pack_int8_weight(orig.weight, spatial_dims=3)
        self.register_buffer("int8_qw", qw, persistent=False)
        self.register_buffer("int8_sw", sw * (1.0 / 127.0), persistent=False)

    def forward(self, x):
        if not x.is_cuda or x.shape[0] != 1 or x.shape[2] < 3:
            return super().forward(x)
        if not x.is_contiguous(memory_format=CL3D) or x.is_contiguous():
            x = x.contiguous(memory_format=CL3D)
        B, C, T_in, H, W = x.shape
        amax = x.abs().amax()
        xr = x.permute(0, 2, 3, 4, 1).reshape(-1)

        C_pad = self.int8_cpad
        n_tot = T_in * H * W * C_pad
        qx = torch.empty(n_tot, device=x.device, dtype=torch.int8)
        _quant_pad_s8[(triton.cdiv(n_tot, 4096),)](xr, xr, qx, amax, C, C_pad, 0, 0, n_tot,
                                                   BLOCK=4096, num_warps=8)

        C_out = self.out_channels
        scale_vec = self.int8_sw * amax.float()
        bias = self.bias
        T_out = T_in - 2
        M = T_out * H * W
        out = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, 128), triton.cdiv(C_out, 64))
        _conv3d_s8[grid](qx, self.int8_qw, out, scale_vec,
                         bias.float() if bias is not None else scale_vec,
                         H, W, C_pad, C_out, 3 * C_pad, M,
                         BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                         HAS_BIAS=bias is not None, num_warps=4, num_stages=4)
        return out.view(1, T_out, H, W, C_out).permute(0, 4, 1, 2, 3)


def _ltx_inner_eligible(conv):
    return isinstance(conv, torch.nn.Conv3d) and tuple(conv.kernel_size) == (3, 3, 3) \
        and tuple(conv.stride) == (1, 1, 1) and tuple(conv.dilation) == (1, 1, 1) \
        and tuple(conv.padding) == (0, 1, 1) and conv.padding_mode == "zeros" \
        and conv.in_channels >= 64 and conv.in_channels % 32 == 0 and conv.out_channels >= 64


def build_int8_patches(model):
    # SD1.5/SDXL activation ranges break per-tensor int8; flux-family KL is fine
    kl_ok = getattr(getattr(getattr(model, "decoder", None), "conv_in", None), "in_channels", 0) >= 16
    kl_skipped = False
    patches = {}
    for name, mod in model.named_modules():
        if not name.startswith("decoder."):
            continue
        if isinstance(mod, Int8CausalConv3d):
            patches[name] = mod
        elif isinstance(mod, CausalConv3d) and tuple(mod.kernel_size) == (3, 3, 3) \
                and tuple(mod.stride) == (1, 1, 1) and mod.in_channels >= 64 \
                and mod.in_channels % 32 == 0 and mod.out_channels >= 64:
            patches[name] = Int8CausalConv3d(mod)
        elif isinstance(mod, ResnetBlock):  # KL image VAEs
            if not kl_ok:
                kl_skipped = True
                continue
            for conv_name in ("conv1", "conv2"):
                conv = getattr(mod, conv_name)
                if isinstance(conv, Int8Conv2d):
                    patches[f"{name}.{conv_name}"] = conv
                elif isinstance(conv, torch.nn.Conv2d) and tuple(conv.kernel_size) == (3, 3) \
                        and tuple(conv.stride) == (1, 1) and conv.in_channels >= 64 \
                        and conv.out_channels >= 64:
                    patches[f"{name}.{conv_name}"] = Int8Conv2d(conv)
        elif isinstance(mod, LTXResnetBlock3D):  # LTXV/LTX2
            for conv_name in ("conv1", "conv2"):
                wrapper = getattr(mod, conv_name)
                if not isinstance(wrapper, LTXCausalConv3d):
                    continue
                inner = wrapper.conv
                if isinstance(inner, Int8InnerConv3d):
                    patches[f"{name}.{conv_name}.conv"] = inner
                elif _ltx_inner_eligible(inner):
                    patches[f"{name}.{conv_name}.conv"] = Int8InnerConv3d(inner)
    if kl_skipped:
        logging.info("int8_conv: skipping SD1.5/SDXL-class VAE (4ch latent), int8 quality is insufficient there")
    return patches


def patch_vae(
    vae, fuse_norm_silu=True, channels_last=True, int8_conv=False,
    autotune=False,
):
    vae = copy.copy(vae)
    model = vae.first_stage_model
    if vae.patcher.is_dynamic():
        model.to(vae.vae_dtype)
        new_patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=vae.patcher.load_device,
            offload_device=vae.patcher.offload_device)
        new_patcher.parent = vae.patcher
        vae.patcher = new_patcher

        def clear_dynamic_cast_flags(
            patcher, device_to, lowvram_model_memory, force_patch_weights,
            full_load,
        ):
            for mod in patcher.model.modules():
                if (hasattr(mod, "comfy_cast_weights")
                        and not hasattr(mod, "prev_comfy_cast_weights")):
                    mod.comfy_cast_weights = False
                    mod._v_signature = None

        vae.patcher.add_callback_with_key(
            CallbacksMP.ON_LOAD, "wan_vae_fused_clear_cast",
            clear_dynamic_cast_flags)
    else:
        vae.patcher = vae.patcher.clone()

    if fuse_norm_silu:
        patches = build_object_patches(model, autotune=autotune)
        if not patches:
            raise RuntimeError(
                "No fusable norm layers found, this node supports Wan video "
                "VAEs, KL image VAEs (Flux2/SDXL/SD1.5) and LTXV/LTX2 video "
                "VAEs")
        for name, obj in patches.items():
            vae.patcher.add_object_patch(name, obj)
        logging.info(
            f"PatchTritonVAE: registered {len(patches)} fused norm object "
            "patches")

    int8_applied = False
    if int8_conv:
        int8_patches = build_int8_patches(model)
        if int8_patches:
            for name, obj in int8_patches.items():
                vae.patcher.add_object_patch(name, obj)
            int8_applied = True
            logging.info(
                f"PatchTritonVAE: registered {len(int8_patches)} int8 conv "
                "object patches")
            if fuse_norm_silu:
                for obj in vae.patcher.object_patches.values():
                    if isinstance(obj, (FusedRMSSiLU, FusedGNSiLU)):
                        obj.emit_amax = True
        else:
            logging.warning(
                "PatchTritonVAE: int8_conv enabled but no eligible conv layers "
                "found (Wan video VAEs and flux-family KL VAEs only), skipping")

    if channels_last or int8_applied:
        if int8_applied and not channels_last:
            logging.info(
                "PatchTritonVAE: channels_last enforced, the int8 conv path "
                "requires it")
        vae.disable_offload = True

        def reapply_channels_last(
            patcher, device_to, lowvram_model_memory, force_patch_weights,
            full_load,
        ):
            convert_conv_layout(patcher.model, channels_last=True)

        vae.patcher.add_callback_with_key(
            CallbacksMP.ON_LOAD, "wan_vae_fused_channels_last",
            reapply_channels_last)
    else:
        convert_conv_layout(model, channels_last=False)
    return vae

