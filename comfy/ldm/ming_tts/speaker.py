# Adapted from https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/campplus
# (Apache-2.0, Copyright 3D-Speaker). Batch normalization is folded during conversion.
import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.ops


def filterbank(waveform):
    """Kaldi's default 16 kHz, 80-bin log filterbank, without dither."""
    frames = waveform.unfold(-1, 400, 160)
    frames = frames - frames.mean(-1, keepdim=True)
    frames = frames - 0.97 * F.pad(frames, (1, 0), mode="replicate")[..., :-1]
    frames = frames * torch.hann_window(400, periodic=False, device=frames.device, dtype=frames.dtype).pow(0.85)
    phase = torch.arange(257, device=frames.device, dtype=frames.dtype)[:, None] * torch.arange(400, device=frames.device, dtype=frames.dtype)
    phase = ((phase + 256).remainder(512) - 256) * (2 * math.pi / 512)
    real, imag = F.linear(frames, torch.cat((phase.cos(), phase.sin()))).chunk(2, dim=-1)
    power = real.square() + imag.square()
    low = 1127 * math.log(1 + 20 / 700)
    step = (1127 * math.log(1 + 8000 / 700) - low) / 81
    bins = torch.arange(80, device=frames.device)[:, None]
    left, center, right = low + bins * step, low + (bins + 1.0) * step, low + (bins + 2.0) * step
    frequencies = 1127 * torch.log(1 + (16000 / 512) * torch.arange(256, device=frames.device) / 700)
    bank = torch.minimum((frequencies - left) / (center - left), (right - frequencies) / (right - center)).clamp_min(0)
    features = (power @ F.pad(bank, (0, 1)).T).clamp_min(torch.finfo(torch.float32).eps).log()
    return features - features.mean(-2, keepdim=True)


class Affine(nn.Module):
    def __init__(self, channels, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, device=device))
        self.bias = nn.Parameter(torch.empty(channels, device=device))

    def forward(self, x):
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * comfy.ops.cast_to_input(self.weight, x).view(shape) + comfy.ops.cast_to_input(self.bias, x).view(shape)


class Residual(nn.Module):
    def __init__(self, stride, device, operations):
        super().__init__()
        self.conv1 = operations.Conv2d(32, 32, 3, stride=(stride, 1), padding=1, device=device)
        self.conv2 = operations.Conv2d(32, 32, 3, padding=1, device=device)
        self.shortcut = nn.Sequential(operations.Conv2d(32, 32, 1, stride=(stride, 1), device=device)) if stride != 1 else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + self.shortcut(x))


class CAMPHead(nn.Module):
    def __init__(self, device, operations):
        super().__init__()
        self.conv1 = operations.Conv2d(1, 32, 3, padding=1, device=device)
        self.layer1 = nn.Sequential(Residual(2, device, operations), Residual(1, device, operations))
        self.layer2 = nn.Sequential(Residual(2, device, operations), Residual(1, device, operations))
        self.conv2 = operations.Conv2d(32, 32, 3, stride=(2, 1), padding=1, device=device)

    def forward(self, x):
        x = self.layer2(self.layer1(F.relu(self.conv1(x[:, None]))))
        return F.relu(self.conv2(x)).flatten(1, 2)


class CAMLayer(nn.Module):
    def __init__(self, channels, dilation, device, operations):
        super().__init__()
        self.norm = Affine(channels, device)
        self.linear1 = operations.Conv1d(channels, 128, 1, device=device)
        self.cam_layer = nn.Module()
        self.cam_layer.linear_local = operations.Conv1d(128, 32, 3, padding=dilation, dilation=dilation, bias=False, device=device)
        self.cam_layer.linear1 = operations.Conv1d(128, 64, 1, device=device)
        self.cam_layer.linear2 = operations.Conv1d(64, 32, 1, device=device)

    def forward(self, x):
        x = F.relu(self.linear1(F.relu(self.norm(x))))
        context = F.avg_pool1d(x, 100, 100, ceil_mode=True).repeat_interleave(100, dim=-1)[..., :x.shape[-1]]
        context = context + x.mean(-1, keepdim=True)
        mask = self.cam_layer.linear2(F.relu(self.cam_layer.linear1(context))).sigmoid()
        return self.cam_layer.linear_local(x) * mask


class CAMPPlus(nn.Module):
    def __init__(self, device, operations):
        super().__init__()
        self.head = CAMPHead(device, operations)
        self.xvector = nn.Module()
        self.xvector.tdnn = nn.Module()
        self.xvector.tdnn.linear = operations.Conv1d(320, 128, 5, stride=2, padding=2, device=device)
        channels = 128
        for i, (count, dilation) in enumerate(((12, 1), (24, 2), (16, 2)), 1):
            block = nn.ModuleDict({f"tdnnd{j + 1}": CAMLayer(channels + 32 * j, dilation, device, operations) for j in range(count)})
            setattr(self.xvector, f"block{i}", block)
            channels += 32 * count
            transit = nn.Module()
            transit.norm = Affine(channels, device)
            transit.linear = operations.Conv1d(channels, channels // 2, 1, bias=i == 3, device=device)
            setattr(self.xvector, f"transit{i}", transit)
            channels //= 2
        self.xvector.dense = nn.Module()
        self.xvector.dense.linear = operations.Conv1d(channels * 2, 192, 1, bias=False, device=device)
        self.xvector.dense.norm = Affine(192, device)

    def forward(self, features):
        x = F.relu(self.xvector.tdnn.linear(self.head(features.transpose(1, 2))))
        for i in range(1, 4):
            for layer in getattr(self.xvector, f"block{i}").values():
                x = torch.cat((x, layer(x)), dim=1)
            transit = getattr(self.xvector, f"transit{i}")
            x = transit.linear(F.relu(transit.norm(x)))
        x = F.relu(x)
        x = torch.cat((x.mean(-1), x.std(-1)), dim=-1)
        return self.xvector.dense.norm(self.xvector.dense.linear(x[..., None]).squeeze(-1))
