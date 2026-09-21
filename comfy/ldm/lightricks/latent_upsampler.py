from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _rational_for_scale(scale: float) -> Tuple[int, int]:
    mapping = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}
    if float(scale) not in mapping:
        raise ValueError(
            f"Unsupported spatial_scale {scale}. Choose from {list(mapping.keys())}"
        )
    return mapping[float(scale)]


class PixelShuffleND(nn.Module):
    def __init__(self, dims, upscale_factors=(2, 2, 2)):
        super().__init__()
        assert dims in [1, 2, 3], "dims must be 1, 2, or 3"
        self.dims = dims
        self.upscale_factors = upscale_factors

    def forward(self, x):
        if self.dims == 3:
            return rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
                p3=self.upscale_factors[2],
            )
        elif self.dims == 2:
            return rearrange(
                x,
                "b (c p1 p2) h w -> b c (h p1) (w p2)",
                p1=self.upscale_factors[0],
                p2=self.upscale_factors[1],
            )
        elif self.dims == 1:
            return rearrange(
                x,
                "b (c p1) f h w -> b c (f p1) h w",
                p1=self.upscale_factors[0],
            )


class BlurDownsample(nn.Module):
    """
    Anti-aliased spatial downsampling by integer stride using a fixed separable binomial kernel.
    Applies only on H,W. Works for dims=2 or dims=3 (per-frame).
    """

    def __init__(self, dims: int, stride: int):
        super().__init__()
        assert dims in (2, 3)
        assert stride >= 1 and isinstance(stride, int)
        self.dims = dims
        self.stride = stride

        # 5x5 separable binomial kernel [1,4,6,4,1] (outer product), normalized
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()  # shape (5,5)
        self.register_buffer("kernel", k2d[None, None, :, :])  # (1,1,5,5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        def _apply_2d(x2d: torch.Tensor) -> torch.Tensor:
            # x2d: (B, C, H, W)
            B, C, H, W = x2d.shape
            weight = self.kernel.expand(C, 1, 5, 5)  # depthwise
            x2d = F.conv2d(
                x2d, weight=weight, bias=None, stride=self.stride, padding=2, groups=C
            )
            return x2d

        if self.dims == 2:
            return _apply_2d(x)
        else:
            # dims == 3: apply per-frame on H,W
            b, c, f, h, w = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = _apply_2d(x)
            h2, w2 = x.shape[-2:]
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f, h=h2, w=w2)
            return x


class SpatialRationalResampler(nn.Module):
    """
    Fully-learned rational spatial scaling: up by 'num' via PixelShuffle, then anti-aliased
    downsample by 'den' using fixed blur + stride. Operates on H,W only.

    For dims==3, work per-frame for spatial scaling (temporal axis untouched).
    """

    def __init__(self, mid_channels: int, scale: float, operations):
        super().__init__()
        self.scale = float(scale)
        self.num, self.den = _rational_for_scale(self.scale)
        self.conv = operations.Conv2d(
            mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1
        )
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, channels: int, operations, mid_channels: Optional[int] = None, dims: int = 3
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        Conv = operations.Conv2d if dims == 2 else operations.Conv3d

        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = operations.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = operations.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class LatentUpsampler(nn.Module):
    """
    Model to spatially upsample VAE latents.

    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
    """

    def __init__(
        self,
        operations,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        Conv = operations.Conv2d if dims == 2 else operations.Conv3d

        self.initial_conv = Conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = operations.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims, operations=operations) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                operations.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            if rational_resampler:
                self.upsampler = SpatialRationalResampler(
                    mid_channels=mid_channels, scale=self.spatial_scale, operations=operations
                )
            else:
                self.upsampler = nn.Sequential(
                    operations.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                operations.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError(
                "Either spatial_upsample or temporal_upsample must be True"
            )

        self.post_upsample_res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims, operations=operations) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = Conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def get_dtype(self):
        return getattr(self.initial_conv, "weight_comfy_model_dtype", self.initial_conv.weight.dtype)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                x = x[:, :, 1:, :, :]
            else:
                if isinstance(self.upsampler, SpatialRationalResampler):
                    x = self.upsampler(x)
                else:
                    x = rearrange(x, "b c f h w -> (b f) c h w")
                    x = self.upsampler(x)
                    x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x

    @classmethod
    def from_config(cls, config, operations):
        return cls(
            in_channels=config.get("in_channels", 4),
            mid_channels=config.get("mid_channels", 128),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            dims=config.get("dims", 2),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
            spatial_scale=config.get("spatial_scale", 2.0),
            rational_resampler=config.get("rational_resampler", False),
            operations=operations,
        )

    def config(self):
        return {
            "_class_name": "LatentUpsampler",
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "num_blocks_per_stage": self.num_blocks_per_stage,
            "dims": self.dims,
            "spatial_upsample": self.spatial_upsample,
            "temporal_upsample": self.temporal_upsample,
            "spatial_scale": self.spatial_scale,
            "rational_resampler": self.rational_resampler,
        }
