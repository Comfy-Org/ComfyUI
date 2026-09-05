"""Big-LaMa image inpainting generator.

The model architecture is the fixed ``big-lama`` FFC generator from
https://github.com/advimman/lama (Apache-2.0).  Training-only options and
unused architecture variants are intentionally omitted.
"""

import torch
from torch import nn


class FourierUnit(nn.Module):
    def __init__(self, channels, operations):
        super().__init__()
        self.conv_layer = operations.Conv2d(
            channels * 2, channels * 2, kernel_size=1, bias=False)
        self.bn = operations.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        transformed = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
        height, frequency_width = transformed.shape[-2:]
        transformed = torch.stack(
            (transformed.real, transformed.imag), dim=-1)
        transformed = transformed.permute(0, 1, 4, 2, 3).reshape(
            batch, -1, height, frequency_width)
        transformed = self.relu(self.bn(self.conv_layer(transformed)))
        transformed = transformed.reshape(
            batch, -1, 2, transformed.shape[-2], transformed.shape[-1]
        ).permute(0, 1, 3, 4, 2).contiguous()
        transformed = torch.complex(
            transformed[..., 0], transformed[..., 1])
        return torch.fft.irfftn(
            transformed, s=x.shape[-2:], dim=(-2, -1), norm="ortho")


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, operations):
        super().__init__()
        self.conv1 = nn.Sequential(
            operations.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, bias=False),
            operations.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, operations)
        self.conv2 = operations.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    def __init__(
        self, in_channels, out_channels, ratio_gin, ratio_gout,
        operations, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        in_global = int(in_channels * ratio_gin)
        in_local = in_channels - in_global
        out_global = int(out_channels * ratio_gout)
        out_local = out_channels - out_global

        def conv(enabled, source, target):
            if not enabled:
                return nn.Identity()
            return operations.Conv2d(
                source, target, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False,
                padding_mode="reflect")

        self.ratio_gout = ratio_gout
        self.global_in_num = in_global
        self.convl2l = conv(in_local > 0 and out_local > 0,
                            in_local, out_local)
        self.convl2g = conv(in_local > 0 and out_global > 0,
                            in_local, out_global)
        self.convg2l = conv(in_global > 0 and out_local > 0,
                            in_global, out_local)
        self.convg2g = (
            SpectralTransform(in_global, out_global, operations)
            if in_global > 0 and out_global > 0 else nn.Identity()
        )

    def forward(self, value):
        local, global_ = value if isinstance(value, tuple) else (value, 0)
        out_local = 0
        out_global = 0
        if self.ratio_gout != 1:
            out_local = self.convl2l(local) + self.convg2l(global_)
        if self.ratio_gout != 0:
            out_global = self.convl2g(local) + self.convg2g(global_)
        return out_local, out_global


class FFCBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, ratio_gin, ratio_gout,
        operations, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.ffc = FFC(
            in_channels, out_channels, ratio_gin, ratio_gout,
            operations, kernel_size=kernel_size, stride=stride,
            padding=padding)
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = (
            nn.Identity() if ratio_gout == 1
            else operations.BatchNorm2d(out_channels - global_channels)
        )
        self.bn_g = (
            nn.Identity() if ratio_gout == 0
            else operations.BatchNorm2d(global_channels)
        )
        self.act_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.act_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)

    def forward(self, value):
        local, global_ = self.ffc(value)
        return self.act_l(self.bn_l(local)), self.act_g(self.bn_g(global_))


class FFCResnetBlock(nn.Module):
    def __init__(self, channels, operations):
        super().__init__()
        self.conv1 = FFCBlock(
            channels, channels, 0.75, 0.75, operations)
        self.conv2 = FFCBlock(
            channels, channels, 0.75, 0.75, operations)

    def forward(self, value):
        identity_local, identity_global = value
        local, global_ = self.conv1(value)
        local, global_ = self.conv2((local, global_))
        return identity_local + local, identity_global + global_


class ConcatTupleLayer(nn.Module):
    def forward(self, value):
        return torch.cat(value, dim=1)


class BigLamaGenerator(nn.Module):
    """The single published 18-block Big-LaMa generator configuration."""

    def __init__(self, operations):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            FFCBlock(
                4, 64, 0.0, 0.0, operations,
                kernel_size=7, stride=1, padding=0),
            FFCBlock(64, 128, 0.0, 0.0, operations, stride=2),
            FFCBlock(128, 256, 0.0, 0.0, operations, stride=2),
            FFCBlock(256, 512, 0.0, 0.75, operations, stride=2),
        ]
        layers.extend(FFCResnetBlock(512, operations) for _ in range(18))
        layers.append(ConcatTupleLayer())
        for in_channels, out_channels in ((512, 256), (256, 128), (128, 64)):
            layers.extend((
                operations.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3, stride=2,
                    padding=1, output_padding=1),
                operations.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        layers.extend((
            nn.ReflectionPad2d(3),
            operations.Conv2d(64, 3, kernel_size=7),
            nn.Sigmoid(),
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, image, mask):
        masked = torch.cat((image * (1.0 - mask), mask), dim=1)
        predicted = self.model(masked)
        return mask * predicted + (1.0 - mask) * image
