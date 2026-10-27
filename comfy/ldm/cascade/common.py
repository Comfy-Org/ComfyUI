"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention
import comfy.ops

class OptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.heads = nhead

        self.to_q = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(c, c, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, q, k, v, transformer_options={}):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        out = optimized_attention(q, k, v, self.heads, preferred_attention=self.comfy_attention, transformer_options=transformer_options)

        return self.out_proj(out)

class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(c, nhead, dtype=dtype, device=device, operations=operations)
        # self.attn = nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True, dtype=dtype, device=device)

    def forward(self, x, kv, self_attn=False, transformer_options={}):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # Bx4xHxW -> Bx(HxW)x4
        if self_attn:
            kv = torch.cat([x, kv], dim=1)
        # x = self.attn(x, kv, kv, need_weights=False)[0]
        x = self.attn(x, kv, kv, transformer_options=transformer_options)
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x


def LayerNorm2d_op(operations):
    class LayerNorm2d(operations.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, x):
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    return LayerNorm2d

class GlobalResponseNorm(nn.Module):
    "from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"
    def __init__(self, dim, dtype=None, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(1, 1, 1, dim, dtype=dtype, device=device))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return comfy.ops.cast_to_input(self.gamma, x) * (x * Nx) + comfy.ops.cast_to_input(self.beta, x) + x


class ResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0, dtype=None, device=None, operations=None):  # , num_heads=4, expansion=2):
        super().__init__()
        self.depthwise = operations.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c, dtype=dtype, device=device)
        #         self.depthwise = SAMBlock(c, num_heads, expansion)
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.channelwise = nn.Sequential(
            operations.Linear(c + c_skip, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + x_res


class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attention = Attention2D(c, nhead, dropout, dtype=dtype, device=device, operations=operations)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            operations.Linear(c_cond, c, dtype=dtype, device=device)
        )

    def forward(self, x, kv, transformer_options={}):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn, transformer_options=transformer_options)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = LayerNorm2d_op(operations)(c, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.channelwise = nn.Sequential(
            operations.Linear(c, c * 4, dtype=dtype, device=device),
            nn.GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype, device=device),
            nn.Dropout(dropout),
            operations.Linear(c * 4, c, dtype=dtype, device=device)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=['sca'], dtype=None, device=None, operations=None):
        super().__init__()
        self.mapper = operations.Linear(c_timestep, c * 2, dtype=dtype, device=device)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", operations.Linear(c_timestep, c * 2, dtype=dtype, device=device))

    def forward(self, x, t):
        t = t.chunk(len(self.conds) + 1, dim=1)
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        for i, c in enumerate(self.conds):
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            a, b = a + ac, b + bc
        return x * (1 + a) + b
