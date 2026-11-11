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

_CHUNK_BYTES = 2 ** 29

def na2d_pure(
        q: torch.Tensor,                # [B, n_heads, d_qk, H, W] at HR.
        k: torch.Tensor,                # [B, n_heads * d_qk, h, w] at LR; H and W are multiples of h and w.
        v: torch.Tensor,                # [B, n_heads * d_v, h, w] at LR.
        kernel_size: Tuple[int, int],   # (Kh, Kw) attention window over the LR grid.
        scale: float,                   # 1 / sqrt(d_qk) scaling for the Q·K scores.
        output: torch.Tensor = None,    # Pre-allocated [B, n_heads, d_v, H, W] buffer (may be on CPU).
    ) -> torch.Tensor:                  # [B, n_heads, d_v, H, W] (caller views as BCHW).
    """natten.na2d on K/V nearest-upsampled to the HR grid, with the upsample factor as dilation.

    All HR queries of one LR cell see the same Kh x Kw window of the zero-padded LR grid, so each cell is one small
    attention of its queries against that window instead of one unfolded window per HR position.
    """
    B, n, d_qk, H, W = q.shape
    h, w = k.shape[-2:]
    d_v = v.shape[1] // n
    Kh, Kw = kernel_size
    Dh, Dw = H // h, W // w
    KK, DD = Kh * Kw, Dh * Dw
    out = output if output is not None else torch.empty((B, n, d_v, H, W), device=q.device, dtype=q.dtype)
    pad = (Kw // 2, Kw // 2, Kh // 2, Kh // 2)
    k, v = F.pad(k, pad), F.pad(v, pad)
    # LR rows per chunk, from one row's live temporaries: K/V windows, Q, fp32 copies and scores, outputs
    es = q.element_size()
    row_bytes = B * n * w * (KK * (d_qk * (es + 4) + d_v * es) + DD * (d_qk * (es + 4) + KK * (8 + es) + 2 * d_v * es))
    rows = max(1, _CHUNK_BYTES // row_bytes)
    for r0 in range(0, h, rows):
        r1 = min(r0 + rows, h)
        cells = (r1 - r0) * w
        k_w = F.unfold(k[:, :, r0:r1 + Kh - 1], (Kh, Kw)).view(B, n, d_qk, KK, cells).transpose(2, 4)  # [B, n, cells, KK, d_qk]
        v_w = F.unfold(v[:, :, r0:r1 + Kh - 1], (Kh, Kw)).view(B, n, d_v, KK, cells).transpose(2, 4)   # [B, n, cells, KK, d_v]
        q_c = q[..., r0 * Dh:r1 * Dh, :].reshape(B, n, d_qk, r1 - r0, Dh, w, Dw).permute(0, 1, 3, 5, 4, 6, 2).reshape(B, n, cells, DD, d_qk)
        attn = (torch.matmul(q_c.float(), k_w.float().transpose(-1, -2)) * scale).softmax(dim=-1).to(v_w.dtype)
        o = torch.matmul(attn, v_w).view(B, n, r1 - r0, w, Dh, Dw, d_v)
        out[..., r0 * Dh:r1 * Dh, :] = o.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, n, d_v, (r1 - r0) * Dh, W)
    return out


class CrossAttention(nn.Module):
    """Window-restricted cross-attention. No learnable parameters; the model's
    capacity lives entirely in the ImageEncoder convs."""

    def __init__(self, dim: int, num_heads: int, kernel_size: Tuple[int, int] = (9, 9)):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                output=None) -> torch.Tensor:
        B, C, hq, wq = q.shape
        n = self.num_heads
        out_buf = output.view(B, n, v.shape[1] // n, hq, wq) if output is not None else None
        out = na2d_pure(q.view(B, n, C // n, hq, wq), k.to(q.dtype), v.to(q.dtype), self.kernel_size, self.scale, output=out_buf)
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
