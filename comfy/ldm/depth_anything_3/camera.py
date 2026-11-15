"""Camera-token encoder and decoder for Depth Anything 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention_for_device
from .transform import affine_inverse, extri_intri_to_pose_encoding


# -----------------------------------------------------------------------
# Building blocks (mirror depth_anything_3.model.utils.{attention,block})
# -----------------------------------------------------------------------


class _Mlp(nn.Module):
    """Standard 2-layer MLP with GELU. Matches upstream ``utils.attention.Mlp``."""

    def __init__(self, in_features, hidden_features=None, out_features=None, *, device=None, dtype=None, operations=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, bias=True, device=device, dtype=dtype)
        self.fc2 = operations.Linear(hidden_features, out_features, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LayerScale(nn.Module):
    """Per-channel learnable scaling. Matches upstream LayerScale."""

    def __init__(self, dim, *, device=None, dtype=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x * self.gamma.to(dtype=x.dtype, device=x.device)


class _Attention(nn.Module):
    """ Self-attention with fused QKV projection. Mirrors upstream utils.attention.Attention;
    Layout matches the HF safetensors (attn.qkv.{weight,bias} and attn.proj.{weight,bias})."""

    def __init__(self, dim, num_heads, *, device=None, dtype=None, operations=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)                      # each (B, N, C)
        attn_fn = optimized_attention_for_device(x.device, small_input=True)
        out = attn_fn(q, k, v, heads=self.num_heads)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block with LayerScale. Used by :class:CameraEnc. Layout follows upstream utils.block.Block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, init_values=0.01, *, device=None, dtype=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.attn = _Attention(dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.ls1 = _LayerScale(dim, device=device, dtype=dtype) if init_values else nn.Identity()
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), device=device, dtype=dtype, operations=operations)
        self.ls2 = _LayerScale(dim, device=device, dtype=dtype) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class CameraEnc(nn.Module):
    """Encode per-view (extrinsics, intrinsics) into a camera token.

    Maps a 9-D pose-encoding vector through a small MLP up to the backbone's
    ``embed_dim``, then runs ``trunk_depth`` transformer blocks. The output
    has shape ``(B, S, embed_dim)`` and is injected at block ``alt_start``
    of the DINOv2 backbone in place of the cls token.

    Parameters mirror the upstream ``cam_enc.py`` so HF weights load directly.
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        *,
        device=None, dtype=None, operations=None,
        **_kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(*[
            _Block(dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   init_values=init_values,
                   device=device, dtype=dtype, operations=operations)
            for _ in range(trunk_depth)
        ])
        self.token_norm = operations.LayerNorm(dim_out, device=device, dtype=dtype)
        self.trunk_norm = operations.LayerNorm(dim_out, device=device, dtype=dtype)
        self.pose_branch = _Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            device=device, dtype=dtype, operations=operations,
        )

    def forward(self, extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                image_size_hw) -> torch.Tensor:
        """Encode camera parameters into ``(B, S, dim_out)`` tokens."""
        c2ws = affine_inverse(extrinsics)
        pose_encoding = extri_intri_to_pose_encoding(c2ws, intrinsics, image_size_hw)
        tokens = self.pose_branch(pose_encoding.to(self.pose_branch.fc1.weight.dtype))
        tokens = self.token_norm(tokens)
        tokens = self.trunk(tokens)
        tokens = self.trunk_norm(tokens)
        return tokens


class CameraDec(nn.Module):
    """Decode the final cam token into a 9-D pose encoding.

    Output layout: ``[T(3), quat_xyzw(4), fov_h, fov_w]``. The translation is
    always predicted by the network; the quaternion and FoV can either be
    predicted or supplied via ``camera_encoding`` (used at training time
    when GT cameras are available -- not exercised at inference here).

    Parameters mirror the upstream ``cam_dec.py`` so HF weights load directly.
    """

    def __init__(self, dim_in: int = 1536,
                 *, device=None, dtype=None, operations=None, **_kwargs):
        super().__init__()
        d = dim_in
        self.backbone = nn.Sequential(
            operations.Linear(d, d, device=device, dtype=dtype),
            nn.ReLU(),
            operations.Linear(d, d, device=device, dtype=dtype),
            nn.ReLU(),
        )
        self.fc_t = operations.Linear(d, 3, device=device, dtype=dtype)
        self.fc_qvec = operations.Linear(d, 4, device=device, dtype=dtype)
        self.fc_fov = nn.Sequential(
            operations.Linear(d, 2, device=device, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor,
                camera_encoding: "torch.Tensor | None" = None) -> torch.Tensor:
        """Decode ``(B, N, dim_in)`` cam tokens into ``(B, N, 9)`` pose enc."""
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        if camera_encoding is None:
            out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]
        return torch.cat([out_t, out_qvec, out_fov], dim=-1)
