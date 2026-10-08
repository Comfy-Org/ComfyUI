"""NAF (Neighborhood Attention Filtering) feature upsampler.

Vendored from valeoai/NAF (Apache-2.0):
  https://github.com/valeoai/NAF — src/model/naf.py + src/layers/{convolutions,attentions,rope}.py
Used by Pixal3D's shape/texture conditioning to produce
the 2x-upsampled half of the 2048-channel proj feature map.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops


# Pure-torch neighborhood attention (replaces natten.na2d / na2d_qk + na2d_av).

def upsample_lr_slice(src_lr: torch.Tensor, lr_dh: int, lr_dw: int,
                       hr_h_range: Tuple[int, int], hr_w_range: Tuple[int, int]) -> torch.Tensor:
    """Slice a LR-layout tensor [B, h_lr, w_lr, n, C], permute to BCHW, and
    nearest-exact upsample only the region covering [hr_h_range, hr_w_range].
    Returns BCHW at hr_h_end-hr_h_start x hr_w_end-hr_w_start (no padding for
    out-of-bounds regions)."""
    B = src_lr.shape[0]
    n = src_lr.shape[-2]
    C = src_lr.shape[-1]
    h_hr_start, h_hr_end = hr_h_range
    w_hr_start, w_hr_end = hr_w_range
    # LR positions covering [h_hr_start, h_hr_end). Nearest-exact maps HR p → p // D.
    lr_h_start = h_hr_start // lr_dh
    lr_h_end   = (h_hr_end - 1) // lr_dh + 1
    lr_w_start = w_hr_start // lr_dw
    lr_w_end   = (w_hr_end - 1) // lr_dw + 1
    lr_slice = src_lr[:, lr_h_start:lr_h_end, lr_w_start:lr_w_end]
    lh, lw = lr_slice.shape[1], lr_slice.shape[2]
    lr_bcd = lr_slice.permute(0, 3, 4, 1, 2).reshape(B * n, C, lh, lw).contiguous()
    up = F.interpolate(lr_bcd, scale_factor=(lr_dh, lr_dw), mode="nearest-exact")
    offset_h = h_hr_start - lr_h_start * lr_dh
    offset_w = w_hr_start - lr_w_start * lr_dw
    return up[:, :, offset_h:offset_h + (h_hr_end - h_hr_start),
                    offset_w:offset_w + (w_hr_end - w_hr_start)]


def na2d_pure(
        q: torch.Tensor,                # [B, H, W, n_heads, d_qk] at HR.
        k_lr: torch.Tensor,             # [B, h_lr, w_lr, n_heads, d_qk] at LR
        v_lr: torch.Tensor,             # [B, h_lr, w_lr, n_heads, d_v] at LR
        kernel_size: Tuple[int, int],   # (Kh, Kw) attention window.
        dilation: Tuple[int, int],      # (Dh, Dw) stride within the unrolled K/V grid; also the LR→HR upsample factor.
        scale: float,                   # 1 / sqrt(d_qk) scaling for the Q·K scores.
        tile: int = 128,                # Spatial tile size (output positions per tile)
        v_chunk: int = 64,              # Sub-divide d_v into chunks of this size when computing attn·V. None disables chunking.
        output: torch.Tensor = None,    # Pre-allocated [B, n_heads, d_v, H, W] buffer (may be on CPU).
    ) -> torch.Tensor:                  # [B, n_heads, d_v, H, W] (caller views as BCHW).
    """Neighborhood attention in pure torch via F.unfold + per-tile slicing.

    K and V are passed at LR resolution and upsampled (nearest-exact) per-tile only
    for the slice the unfold needs. Avoids the [B, n*d, H, W] HR allocations for K
    (512 MB) and V (2 GB) at tex_1024 fp16. Spatial tiling bounds the per-tile
    F.unfold blob; `v_chunk` further slices d_v so attn·V is computed in C-sized
    chunks (attn is reused, computed once from Q/K).

    """
    B, H, W, n, d_qk = q.shape
    d_v = v_lr.shape[-1]
    Kh, Kw = kernel_size
    Dh, Dw = dilation
    pad_h, pad_w = (Kh // 2) * Dh, (Kw // 2) * Dw

    out = output if output is not None else torch.empty((B, n, d_v, H, W), device=q.device, dtype=q.dtype)

    th = min(tile, H) if tile else H
    tw = min(tile, W) if tile else W
    chunk = v_chunk if (v_chunk and v_chunk < d_v) else d_v

    for h0 in range(0, H, th):
        for w0 in range(0, W, tw):
            h1, w1 = min(h0 + th, H), min(w0 + tw, W)
            t_h, t_w = h1 - h0, w1 - w0

            # Padded HR region the unfold needs (kernel span = (K-1)*D + 1).
            h_src_start = max(0, h0 - pad_h)
            h_src_end   = min(H, h1 + pad_h)
            w_src_start = max(0, w0 - pad_w)
            w_src_end   = min(W, w1 + pad_w)
            pad_top = max(0, pad_h - h0)
            pad_bot = max(0, (h1 + pad_h) - H)
            pad_lft = max(0, pad_w - w0)
            pad_rgt = max(0, (w1 + pad_w) - W)

            # Upsample only the tile region from k_lr / v_lr.
            k_tile = upsample_lr_slice(k_lr, Dh, Dw,
                                        (h_src_start, h_src_end),
                                        (w_src_start, w_src_end))
            v_tile = upsample_lr_slice(v_lr, Dh, Dw,
                                        (h_src_start, h_src_end),
                                        (w_src_start, w_src_end))
            if pad_top or pad_bot or pad_lft or pad_rgt:
                k_tile = F.pad(k_tile, [pad_lft, pad_rgt, pad_top, pad_bot])
                v_tile = F.pad(v_tile, [pad_lft, pad_rgt, pad_top, pad_bot])

            # Q·K → attention weights (small: KK=81 per output position).
            KK = Kh * Kw
            k_w = F.unfold(k_tile, kernel_size=(Kh, Kw), dilation=(Dh, Dw), padding=0)
            k_w = k_w.view(B, n, d_qk, KK, t_h * t_w).permute(0, 1, 4, 3, 2)  # [B, n, t, KK, d_qk]
            # q is [B, H, W, n, d_qk]; per-tile slice + permute -> [B, n, t_h*t_w, 1, d_qk].
            q_tile = q[:, h0:h1, w0:w1].permute(0, 3, 1, 2, 4).reshape(B, n, t_h * t_w, 1, d_qk)
            scores = torch.matmul(q_tile, k_w.transpose(-1, -2)) * scale
            attn = scores.softmax(dim=-1)
            del k_w, scores, q_tile, k_tile

            # attn · V, chunked over d_v.
            for c0 in range(0, d_v, chunk):
                c1 = min(c0 + chunk, d_v)
                v_w = F.unfold(v_tile[:, c0:c1], kernel_size=(Kh, Kw),dilation=(Dh, Dw), padding=0) # [B*n, (c1-c0)*KK, t]
                v_w = v_w.view(B, n, c1 - c0, KK, t_h * t_w).permute(0, 1, 4, 3, 2)
                out_chunk = torch.matmul(attn, v_w).squeeze(-2) # [B, n, t, c1-c0]
                out_chunk = out_chunk.view(B, n, t_h, t_w, c1 - c0).permute(0, 1, 4, 2, 3)
                out[:, :, c0:c1, h0:h1, w0:w1] = out_chunk
                del v_w, out_chunk
            del attn, v_tile

    return out  # [B, n, d_v, H, W] — sole caller (CrossAttention) views it as BCHW directly.


class CrossAttention(nn.Module):
    """Window-restricted cross-attention. No learnable parameters; the model's
    capacity lives entirely in the ImageEncoder convs."""

    def __init__(self, dim: int, num_heads: int, kernel_size: Tuple[int, int] = (9, 9)):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = (dim // num_heads) ** -0.5

    @staticmethod
    def _split_heads_lr(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """[B, n*d, h, w] -> [B, h, w, n, d] at the input resolution (no upsample)."""
        B, C, H, W = x.shape
        return x.view(B, num_heads, C // num_heads, H, W).permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                output=None) -> torch.Tensor:
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (hq // hk, wq // wk)
        B, C, _, _ = q.shape
        q = q.view(B, self.num_heads, C // self.num_heads, hq, wq).permute(0, 3, 4, 1, 2).contiguous()
        k_lr = self._split_heads_lr(k, self.num_heads).to(q.dtype)
        v_lr = self._split_heads_lr(v, self.num_heads).to(q.dtype)
        out_buf = output.view(B, self.num_heads, v.shape[1] // self.num_heads, hq, wq) if output is not None else None
        out = na2d_pure(q, k_lr, v_lr, self.kernel_size, dilation, self.scale, output=out_buf)
        return out.view(B, -1, hq, wq)


# RoPE positional embedding

def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.num_heads = num_heads
        self.D_head = embed_dim // num_heads
        self.base = base
        self.register_buffer("periods", torch.empty(self.D_head // 4), persistent=True) # loaded from the checkpoint

    def _cos_sin(self, H: int, W: int, x: torch.Tensor):
        """cos/sin depend only on (H, W, dtype) and the checkpoint-fixed periods; recomputed per forward."""
        periods = comfy.ops.cast_to_input(self.periods, x)
        coords_h = torch.arange(0.5, H, device=x.device, dtype=torch.float32) / H
        coords_w = torch.arange(0.5, W, device=x.device, dtype=torch.float32) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1) * 2.0 - 1.0  # [HW, 2]
        angles = 2 * math.pi * coords[:, :, None] / periods.to(coords.dtype)[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D]
        cos = torch.cos(angles).to(x.dtype)
        sin = torch.sin(angles).to(x.dtype)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n*D_head, H, W]
        B, C, H, W = x.shape
        n = self.num_heads
        D = C // n
        x = x.view(B, n, D, H, W).permute(0, 1, 3, 4, 2).reshape(B, n, H * W, D)
        cos, sin = self._cos_sin(H, W, x)
        x = (x * cos) + (rope_rotate_half(x) * sin)
        x = x.view(B, n, H, W, D).permute(0, 1, 4, 2, 3).reshape(B, n * D, H, W)
        return x


# Image encoder

class EncBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, operations, num_groups: int = 8):
        super().__init__()
        self.norm1 = operations.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv1 = operations.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size // 2, padding_mode="reflect", bias=True)
        self.norm2 = operations.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = operations.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding=kernel_size // 2, padding_mode="reflect", bias=True)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        return x  # no skip connection


def _encoder(in_dim: int, hidden_dim: int, operations, kernel_size: int = 1, ks_res: int = 1, num_layers: int = 2) -> nn.Sequential:
    return nn.Sequential(
        operations.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="reflect", bias=True),
        *[EncBlock(hidden_dim, kernel_size=ks_res, operations=operations) for _ in range(num_layers)],
    )


class ImageEncoder(nn.Module):
    """Two parallel conv stacks (1x1 + 3x3) producing dim/2 channels each, then concat,
    spatial average-pool to target size, RoPE-embed positions."""

    def __init__(self, operations, in_channels: int = 3, out_channels: int = 256,
                 heads_rope: int = 4, rope_base: float = 100.0, img_layers: int = 2):
        super().__init__()
        half = out_channels // 2
        self.encoder = _encoder(in_channels, half, operations=operations, kernel_size=1, ks_res=1, num_layers=img_layers)
        self.sem_encoder = _encoder(in_channels, half, operations=operations, kernel_size=3, ks_res=3, num_layers=img_layers)
        self.rope = RoPE(embed_dim=out_channels, num_heads=heads_rope, base=rope_base)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        # Avoid running the conv stacks on >4× the target resolution.
        out_h, out_w = output_size
        if x.shape[-2] > 4 * out_h or x.shape[-1] > 4 * out_w:
            x = F.interpolate(x, size=(min(x.shape[-2], 4 * out_h),
                                       min(x.shape[-1], 4 * out_w)),
                              mode="bilinear", align_corners=False)
        x = torch.cat([self.encoder(x), self.sem_encoder(x)], dim=1)
        x = F.adaptive_avg_pool2d(x, output_size=output_size)
        x = self.rope(x)
        return x


class NAF(nn.Module):
    """NAF feature upsampler."""

    def __init__(
            self, operations,
            dim: int = 256,             # internal channel dimension of the ImageEncoder
            heads_attn: int = 4,        # attention heads in the windowed cross-attn
            heads_rope: int = 4,        # heads for RoPE position encoding (must divide dim)
            kernel_size: int = 9,       # square kernel for the neighborhood attention window
            rope_base: float = 100.0,   # base for RoPE frequency periods
            img_layers: int = 2,        # number of EncBlocks in each conv stack
        ):
        super().__init__()
        self.image_encoder = ImageEncoder(operations=operations, in_channels=3, out_channels=dim, heads_rope=heads_rope, rope_base=rope_base, img_layers=img_layers)
        self.upsampler = CrossAttention(dim=dim, num_heads=heads_attn, kernel_size=(kernel_size, kernel_size))

    def forward(
            self,
            image: torch.Tensor,            # [B, 3, H_img, W_img] in [0, 1].
            features: torch.Tensor,         # [B, C, H_feat, W_feat] low-resolution features (any C).
            output_size: Tuple[int, int],   # (H_out, W_out) target spatial resolution for the upsampled features.
            output=None,
        ) -> torch.Tensor:                  # [B, C, H_out, W_out] upsampled features.
        """Upsample low-res feature map to output_size, guided by the image."""
        q = self.image_encoder(image, output_size=output_size)
        k = F.adaptive_avg_pool2d(q, output_size=features.shape[-2:])
        return self.upsampler(q, k, features, output=output)
