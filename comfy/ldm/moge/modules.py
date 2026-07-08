"""Building blocks for MoGe: residual conv stack, resamplers, MLP, DINOv2 encoder, v1 head."""


from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
from comfy.image_encoders.dino2 import Dinov2Model

from .geometry import normalized_view_plane_uv


def _conv2d(operations, c_in: int, c_out: int, k: int = 3, *, dtype=None, device=None):
    return operations.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2, padding_mode="replicate", dtype=dtype, device=device)


def _view_plane_uv_grid(batch: int, height: int, width: int, aspect_ratio: float, dtype, device) -> torch.Tensor:
    """Batched normalized view-plane UV grid as a (B, 2, H, W) tensor."""
    uv = normalized_view_plane_uv(width, height, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
    return uv.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)


def _concat_view_plane_uv(x: torch.Tensor, aspect_ratio: float) -> torch.Tensor:
    """Append a 2-channel normalized view-plane UV grid to x along the channel dim."""
    uv = _view_plane_uv_grid(x.shape[0], x.shape[-2], x.shape[-1], aspect_ratio, x.dtype, x.device)
    return torch.cat([x, uv], dim=1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: Optional[int] = None, in_norm: str = "layer_norm", hidden_norm: str = "group_norm",
                 dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels is not None else channels

        in_norm_layer = operations.GroupNorm(1, channels, dtype=dtype, device=device) if in_norm == "layer_norm" else nn.Identity()
        hidden_norm_layer = (operations.GroupNorm(max(hidden_channels // 32, 1), hidden_channels, dtype=dtype, device=device)
                             if hidden_norm == "group_norm" else nn.Identity())

        self.layers = nn.Sequential(
            in_norm_layer, nn.ReLU(), _conv2d(operations, channels, hidden_channels, dtype=dtype, device=device),
            hidden_norm_layer, nn.ReLU(), _conv2d(operations, hidden_channels, channels, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.layers(x) + x


class Resampler(nn.Sequential):
    """2x upsampler: ConvTranspose2d(2x2) or bilinear upsample, followed by a 3x3 conv."""

    def __init__(self, in_channels: int, out_channels: int, type_: str, dtype=None, device=None, operations=comfy.ops.manual_cast):
        if type_ == "conv_transpose":
            up = operations.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=dtype, device=device)
            conv_in = out_channels
        else:  # "bilinear"
            up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = in_channels
        super().__init__(up, _conv2d(operations, conv_in, out_channels, dtype=dtype, device=device))


class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int], dtype=None, device=None, operations=comfy.ops.manual_cast):
        layers = []
        for d_in, d_out in zip(dims[:-2], dims[1:-1]):
            layers.append(operations.Linear(d_in, d_out, dtype=dtype, device=device))
            layers.append(nn.ReLU(inplace=True))
        layers.append(operations.Linear(dims[-2], dims[-1], dtype=dtype, device=device))
        super().__init__(*layers)


class ConvStack(nn.Module):
    def __init__(self, dim_in: List[Optional[int]], dim_res_blocks: List[int], dim_out: List[Optional[int]], resamplers: List[str],
                 num_res_blocks: List[int], dim_times_res_block_hidden: int = 1, res_block_in_norm: str = "layer_norm", res_block_hidden_norm: str = "group_norm",
                 dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()

        self.input_blocks = nn.ModuleList([
            (_conv2d(operations, d_in, d_res, k=1, dtype=dtype, device=device)
             if d_in is not None else nn.Identity())
            for d_in, d_res in zip(dim_in, dim_res_blocks)
        ])

        self.resamplers = nn.ModuleList([
            Resampler(prev, succ, type_=r, dtype=dtype, device=device, operations=operations)
            for prev, succ, r in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resamplers)
        ])

        self.res_blocks = nn.ModuleList([
            nn.Sequential(*[
                ResidualConvBlock(d_res, dim_times_res_block_hidden * d_res, in_norm=res_block_in_norm, hidden_norm=res_block_hidden_norm, dtype=dtype, device=device, operations=operations)
                for _ in range(num_res_blocks[i])
            ])
            for i, d_res in enumerate(dim_res_blocks)
        ])

        self.output_blocks = nn.ModuleList([
            (_conv2d(operations, d_res, d_out, k=1, dtype=dtype, device=device)
             if d_out is not None else nn.Identity())
            for d_out, d_res in zip(dim_out, dim_res_blocks)
        ])

    def forward(self, in_features: List[Optional[torch.Tensor]]):
        out_features = []
        x = None
        for i in range(len(self.res_blocks)):
            feat = self.input_blocks[i](in_features[i]) if in_features[i] is not None else None
            if i == 0:
                x = feat
            elif feat is not None:
                x = x + feat
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


class DINOv2Encoder(nn.Module):
    """Comfy DINOv2 backbone with per-layer 1x1 projection heads."""

    def __init__(self, backbone: dict, intermediate_layers: List[int], dim_out: int, dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.intermediate_layers = list(intermediate_layers)
        dim_features = backbone["hidden_size"]
        self.backbone = Dinov2Model(backbone, dtype, device, operations)
        self.output_projections = nn.ModuleList([
            _conv2d(operations, dim_features, dim_out, k=1, dtype=dtype, device=device)
            for _ in range(len(self.intermediate_layers))
        ])
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, token_rows: int, token_cols: int,
                return_class_token: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False, antialias=True)
        image_14 = (image_14 - self.image_mean) / self.image_std
        feats = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, apply_norm=True)
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
            for proj, (feat, _cls) in zip(self.output_projections, feats)
        ], dim=1).sum(dim=1)
        if return_class_token:
            return x, feats[-1][1]
        return x


class HeadV1(nn.Module):
    """v1 head: 4 backbone-feature projections -> shared upsample stack -> per-target output convs (points, mask)."""

    NUM_FEATURES = 4
    DIM_PROJ = 512
    DIM_OUT = (3, 1) # 3 channels for points, 1 for mask
    LAST_CONV_CHANNELS = 32

    def __init__(self, dim_in: int, dim_upsample: List[int] = (256, 128, 128), num_res_blocks: int = 1, dim_times_res_block_hidden: int = 1,
                 dtype=None, device=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.projects = nn.ModuleList([
            _conv2d(operations, dim_in, self.DIM_PROJ, k=1, dtype=dtype, device=device)
            for _ in range(self.NUM_FEATURES)
        ])
        def upsampler(in_ch, out_ch):
            return nn.Sequential(
                operations.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, dtype=dtype, device=device),
                _conv2d(operations, out_ch, out_ch, dtype=dtype, device=device),
            )

        in_chs = [self.DIM_PROJ] + list(dim_upsample[:-1])
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, dim_times_res_block_hidden * out_ch, dtype=dtype, device=device, operations=operations)
                  for _ in range(num_res_blocks))
            )
            for in_ch, out_ch in zip(in_chs, dim_upsample)
        ])
        self.output_block = nn.ModuleList([
            nn.Sequential(
                _conv2d(operations, dim_upsample[-1] + 2, self.LAST_CONV_CHANNELS, dtype=dtype, device=device),
                nn.ReLU(inplace=True),
                _conv2d(operations, self.LAST_CONV_CHANNELS, d_out, k=1, dtype=dtype, device=device),
            )
            for d_out in self.DIM_OUT
        ])

    def forward(self, hidden_states, image: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14
        aspect = img_w / img_h
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
            for proj, (feat, _cls) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        for block in self.upsample_blocks:
            x = block(_concat_view_plane_uv(x, aspect))

        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        x = _concat_view_plane_uv(x, aspect)
        return [block(x) for block in self.output_block]
