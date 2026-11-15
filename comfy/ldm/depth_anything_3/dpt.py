"""DPT / DualDPT heads for Depth Anything 3."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def _custom_interpolate(
    x: torch.Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks]
        return torch.cat(outs, dim=0).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _create_uv_grid(width: int, height: int, aspect_ratio: float, dtype, device) -> torch.Tensor:
    """Normalised UV grid spanning (-x_span, -y_span)..(x_span, y_span)."""
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)  # (H, W, 2)


def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / omega_0 ** (omega / (embed_dim / 2.0))
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([out.sin(), out.cos()], dim=1).float()


def _position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100.0) -> torch.Tensor:
    H, W, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def _add_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    """Stateless UV positional embedding added to a feature map (B, C, h, w)."""
    pw, ph = x.shape[-1], x.shape[-2]
    pe = _create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pe = _position_grid_to_embed(pe, x.shape[1]) * ratio
    pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1).to(dtype=x.dtype)
    return x + pe


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    act = (activation or "linear").lower()
    if act == "exp":
        return torch.exp(x)
    if act == "expp1":
        return torch.exp(x) + 1
    if act == "expm1":
        return torch.expm1(x)
    if act == "relu":
        return torch.relu(x)
    if act == "sigmoid":
        return torch.sigmoid(x)
    if act == "softplus":
        return F.softplus(x)
    if act == "tanh":
        return torch.tanh(x)
    return x


# -----------------------------------------------------------------------------
# Fusion building blocks
# -----------------------------------------------------------------------------


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = operations.Conv2d(features, features, 3, 1, 1, bias=True, device=device, dtype=dtype)
        self.conv2 = operations.Conv2d(features, features, 3, 1, 1, bias=True, device=device, dtype=dtype)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True, align_corners: bool = True, device=None, dtype=None, operations=None):
        super().__init__()
        self.align_corners = align_corners
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, device=device, dtype=dtype, operations=operations)
        else:
            self.resConfUnit1 = None
        self.resConfUnit2 = ResidualConvUnit(features, device=device, dtype=dtype, operations=operations)
        self.out_conv = operations.Conv2d(features, features, 1, 1, 0, bias=True, device=device, dtype=dtype)

    def forward(self, *xs: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)
        if size is None:
            up_kwargs = {"scale_factor": 2.0}
        else:
            up_kwargs = {"size": size}
        y = _custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y


class _Scratch(nn.Module):
    """Container that mirrors upstream ``scratch`` attribute layout."""


def _make_scratch(in_shape: List[int], out_shape: int, device=None, dtype=None, operations=None) -> _Scratch:
    scratch = _Scratch()
    scratch.layer1_rn = operations.Conv2d(in_shape[0], out_shape, 3, 1, 1, bias=False, device=device, dtype=dtype)
    scratch.layer2_rn = operations.Conv2d(in_shape[1], out_shape, 3, 1, 1, bias=False, device=device, dtype=dtype)
    scratch.layer3_rn = operations.Conv2d(in_shape[2], out_shape, 3, 1, 1, bias=False, device=device, dtype=dtype)
    scratch.layer4_rn = operations.Conv2d(in_shape[3], out_shape, 3, 1, 1, bias=False, device=device, dtype=dtype)
    return scratch


def _make_fusion_block(features: int, has_residual: bool = True, device=None, dtype=None, operations=None) -> FeatureFusionBlock:
    return FeatureFusionBlock(features, has_residual=has_residual, align_corners=True, device=device, dtype=dtype, operations=operations)


# -----------------------------------------------------------------------------
# DPT (single head + optional sky head) -- used by DA3Mono/Metric
# -----------------------------------------------------------------------------


class DPT(nn.Module):
    """Single-head DPT used by DA3Mono-Large and DA3Metric-Large."""

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",
        norm_type: str = "idt",
        device=None, dtype=None, operations=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main = head_name
        self.sky_name = sky_name
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        if norm_type == "layer":
            self.norm = operations.LayerNorm(dim_in, device=device, dtype=dtype)
        else:
            self.norm = nn.Identity()

        out_channels = list(out_channels)
        self.projects = nn.ModuleList([
            operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0, device=device, dtype=dtype),
            operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0, device=device, dtype=dtype),
            nn.Identity(),
            operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
        ])

        self.scratch = _make_scratch(out_channels, features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet1 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet2 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet3 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, device=device, dtype=dtype, operations=operations)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = operations.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1,
            device=device, dtype=dtype,
        )
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
            nn.ReLU(inplace=False),
            operations.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype),
        )

        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
                nn.ReLU(inplace=False),
                operations.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype),
            )

    def forward(self, feats: List[torch.Tensor], H: int, W: int, patch_start_idx: int = 0, **_kwargs) -> dict:
        # feats[i][0] is the patch-token tensor with shape (B, S, N_patch, C)
        B, S, N, C = feats[0][0].shape
        feats_flat = [feat[0].reshape(B * S, N, C) for feat in feats]

        ph, pw = H // self.patch_size, W // self.patch_size
        resized = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats_flat[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B * S, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = _add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized.append(x)

        l1_rn = self.scratch.layer1_rn(resized[0])
        l2_rn = self.scratch.layer2_rn(resized[1])
        l3_rn = self.scratch.layer3_rn(resized[2])
        l4_rn = self.scratch.layer4_rn(resized[3])

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(out)
        fused = _custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = _add_pos_embed(fused, W, H)
        feat = fused

        main_logits = self.scratch.output_conv2(feat)
        outs = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            pred = _apply_activation(fmap[..., :-1], self.activation)
            conf = _apply_activation(fmap[..., -1], self.conf_activation)
            outs[self.head_main] = pred.squeeze(-1).view(B, S, *pred.shape[1:-1])
            outs[f"{self.head_main}_conf"] = conf.view(B, S, *conf.shape[1:])
        else:
            pred = _apply_activation(main_logits, self.activation)
            outs[self.head_main] = pred.squeeze(1).view(B, S, *pred.shape[2:])

        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(feat)
            if self.sky_activation.lower() == "sigmoid":
                sky = torch.sigmoid(sky_logits)
            elif self.sky_activation.lower() == "relu":
                sky = F.relu(sky_logits)
            else:
                sky = sky_logits
            outs[self.sky_name] = sky.squeeze(1).view(B, S, *sky.shape[2:])

        return outs


# -----------------------------------------------------------------------------
# DualDPT (depth + auxiliary "ray" head) -- used by DA3-Small / DA3-Base
# -----------------------------------------------------------------------------


class DualDPT(nn.Module):
    """Two-head DPT used by DA3-Small / DA3-Base."""

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
        device=None, dtype=None, operations=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.head_main, self.head_aux = head_names
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)
        # Toggle the auxiliary ray branch at runtime. Default off (mono path).
        # DepthAnything3Net flips this on when running multi-view + ray-pose.
        self.enable_aux: bool = False

        self.norm = operations.LayerNorm(dim_in, device=device, dtype=dtype)
        out_channels = list(out_channels)
        self.projects = nn.ModuleList([
            operations.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            operations.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0, device=device, dtype=dtype),
            operations.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0, device=device, dtype=dtype),
            nn.Identity(),
            operations.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
        ])

        self.scratch = _make_scratch(out_channels, features, device=device, dtype=dtype, operations=operations)
        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet2 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet3 = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, device=device, dtype=dtype, operations=operations)
        # Auxiliary fusion chain (separate copies)
        self.scratch.refinenet1_aux = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet2_aux = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet3_aux = _make_fusion_block(features, device=device, dtype=dtype, operations=operations)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False, device=device, dtype=dtype, operations=operations)

        head_features_1 = features
        head_features_2 = 32

        # Main head neck + final projection
        self.scratch.output_conv1 = operations.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1,
            device=device, dtype=dtype,
        )
        self.scratch.output_conv2 = nn.Sequential(
            operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
            nn.ReLU(inplace=False),
            operations.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype),
        )

        # Aux pre-head per level (multi-level pyramid)
        self.scratch.output_conv1_aux = nn.ModuleList([
            self._make_aux_out1_block(head_features_1, device=device, dtype=dtype, operations=operations)
            for _ in range(self.aux_levels)
        ])

        # Aux final projection per level (includes LayerNorm permute path).
        ln_seq = [Permute((0, 2, 3, 1)),
                  operations.LayerNorm(head_features_2, device=device, dtype=dtype),
                  Permute((0, 3, 1, 2))]
        self.scratch.output_conv2_aux = nn.ModuleList([
            nn.Sequential(
                operations.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
                *ln_seq,
                nn.ReLU(inplace=False),
                operations.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0, device=device, dtype=dtype),
            )
            for _ in range(self.aux_levels)
        ])

    @staticmethod
    def _make_aux_out1_block(in_ch: int, *, device=None, dtype=None, operations=None) -> nn.Sequential:
        # aux_out1_conv_num=5 in all Apache-2.0 variants.
        return nn.Sequential(
            operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, device=device, dtype=dtype),
            operations.Conv2d(in_ch // 2, in_ch, 3, 1, 1, device=device, dtype=dtype),
            operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, device=device, dtype=dtype),
            operations.Conv2d(in_ch // 2, in_ch, 3, 1, 1, device=device, dtype=dtype),
            operations.Conv2d(in_ch, in_ch // 2, 3, 1, 1, device=device, dtype=dtype),
        )

    def forward(self, feats: List[torch.Tensor], H: int, W: int, patch_start_idx: int = 0, **_kwargs) -> dict:
        B, S, N, C = feats[0][0].shape
        feats_flat = [feat[0].reshape(B * S, N, C) for feat in feats]

        ph, pw = H // self.patch_size, W // self.patch_size
        resized = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats_flat[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B * S, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = _add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized.append(x)

        l1_rn = self.scratch.layer1_rn(resized[0])
        l2_rn = self.scratch.layer2_rn(resized[1])
        l3_rn = self.scratch.layer3_rn(resized[2])
        l4_rn = self.scratch.layer4_rn(resized[3])

        # Main pyramid (output_conv1 is applied inside the upstream `_fuse`,
        # before interpolation -- replicate that order here).
        m = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        if self.enable_aux:
            a4 = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
            aux_pyr = [a4]
        m = self.scratch.refinenet3(m, l3_rn, size=l2_rn.shape[2:])
        if self.enable_aux:
            aux_pyr.append(self.scratch.refinenet3_aux(aux_pyr[-1], l3_rn, size=l2_rn.shape[2:]))
        m = self.scratch.refinenet2(m, l2_rn, size=l1_rn.shape[2:])
        if self.enable_aux:
            aux_pyr.append(self.scratch.refinenet2_aux(aux_pyr[-1], l2_rn, size=l1_rn.shape[2:]))
        m = self.scratch.refinenet1(m, l1_rn)
        if self.enable_aux:
            aux_pyr.append(self.scratch.refinenet1_aux(aux_pyr[-1], l1_rn))
        m = self.scratch.output_conv1(m)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        m = _custom_interpolate(m, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            m = _add_pos_embed(m, W, H)
        main_logits = self.scratch.output_conv2(m)
        fmap = main_logits.permute(0, 2, 3, 1)
        depth_pred = _apply_activation(fmap[..., :-1], self.activation)
        depth_conf = _apply_activation(fmap[..., -1], self.conf_activation)

        outs = {
            self.head_main: depth_pred.squeeze(-1).view(B, S, *depth_pred.shape[1:-1]),
            f"{self.head_main}_conf": depth_conf.view(B, S, *depth_conf.shape[1:]),
        }

        if self.enable_aux:
            # Auxiliary "ray" head (multi-level inside) -- only the last level
            # is returned. Mirrors upstream ``DualDPT._fuse`` + ``_forward_impl``:
            # each aux pyramid level goes through ``output_conv1_aux[i]``
            # (5-layer conv stack that ends at ``features // 2`` channels),
            # then the last level optionally gets a pos-embed and finally
            # ``output_conv2_aux[-1]``.
            aux_processed = [
                self.scratch.output_conv1_aux[i](a) for i, a in enumerate(aux_pyr)
            ]
            last_aux = aux_processed[-1]
            if self.pos_embed:
                last_aux = _add_pos_embed(last_aux, W, H)
            last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
            fmap_last = last_aux_logits.permute(0, 2, 3, 1)
            # Channels: [ray(6), ray_conf(1)]; ray uses 'linear' activation.
            aux_pred = fmap_last[..., :-1]
            aux_conf = _apply_activation(fmap_last[..., -1], self.conf_activation)
            outs[self.head_aux] = aux_pred.view(B, S, *aux_pred.shape[1:])
            outs[f"{self.head_aux}_conf"] = aux_conf.view(B, S, *aux_conf.shape[1:])

        return outs
