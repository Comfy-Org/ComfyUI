# RAFT-large optical flow (Teed and Deng 2020) with torchvision's raft_large state dict layout
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_prefetch

RADIUS = 4
LEVELS = 4
ITERATIONS = 12


def conv_norm_relu(in_channels, out_channels, kernel_size, stride=1, norm=None, relu=True, device=None, dtype=None, operations=None):
    # a Sequential keeps torchvision's Conv2dNormActivation keys: conv at .0, norm at .1
    layers = [operations.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, device=device, dtype=dtype)]
    if norm is not None:
        layers.append(norm(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm, device=None, dtype=None, operations=None):
        super().__init__()
        ops = dict(norm=norm, device=device, dtype=dtype, operations=operations)
        self.convnormrelu1 = conv_norm_relu(in_channels, out_channels, 3, stride, **ops)
        self.convnormrelu2 = conv_norm_relu(out_channels, out_channels, 3, **ops)
        self.downsample = nn.Identity() if stride == 1 else conv_norm_relu(in_channels, out_channels, 1, stride, relu=False, **ops)

    def forward(self, x):
        return F.relu(self.downsample(x) + self.convnormrelu2(self.convnormrelu1(x)))


class Encoder(nn.Module):
    def __init__(self, norm, device=None, dtype=None, operations=None):
        super().__init__()
        ops = dict(device=device, dtype=dtype, operations=operations)
        self.convnormrelu = conv_norm_relu(3, 64, 7, 2, norm, **ops)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64, 1, norm, **ops), ResidualBlock(64, 64, 1, norm, **ops))
        self.layer2 = nn.Sequential(ResidualBlock(64, 96, 2, norm, **ops), ResidualBlock(96, 96, 1, norm, **ops))
        self.layer3 = nn.Sequential(ResidualBlock(96, 128, 2, norm, **ops), ResidualBlock(128, 128, 1, norm, **ops))
        self.conv = operations.Conv2d(128, 256, 1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv(self.layer3(self.layer2(self.layer1(self.convnormrelu(x)))))


class MotionEncoder(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = dict(device=device, dtype=dtype, operations=operations)
        self.convcorr1 = conv_norm_relu(LEVELS * (2 * RADIUS + 1) ** 2, 256, 1, **ops)
        self.convcorr2 = conv_norm_relu(256, 192, 3, **ops)
        self.convflow1 = conv_norm_relu(2, 128, 7, **ops)
        self.convflow2 = conv_norm_relu(128, 64, 3, **ops)
        self.conv = conv_norm_relu(256, 126, 3, **ops)

    def forward(self, flow, corr):
        corr = self.convcorr2(self.convcorr1(corr))
        motion = self.convflow2(self.convflow1(flow))
        return torch.cat([self.conv(torch.cat([corr, motion], dim=1)), flow], dim=1)


class ConvGRU(nn.Module):
    def __init__(self, kernel_size, padding, device=None, dtype=None, operations=None):
        super().__init__()
        self.convz = operations.Conv2d(384, 128, kernel_size, padding=padding, device=device, dtype=dtype)
        self.convr = operations.Conv2d(384, 128, kernel_size, padding=padding, device=device, dtype=dtype)
        self.convq = operations.Conv2d(384, 128, kernel_size, padding=padding, device=device, dtype=dtype)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return torch.lerp(h, q, z)


class RecurrentBlock(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.convgru1 = ConvGRU((1, 5), (0, 2), device=device, dtype=dtype, operations=operations)
        self.convgru2 = ConvGRU((5, 1), (2, 0), device=device, dtype=dtype, operations=operations)

    def forward(self, h, x):
        return self.convgru2(self.convgru1(h, x), x)


class FlowHead(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = operations.Conv2d(128, 256, 3, padding=1, device=device, dtype=dtype)
        self.conv2 = operations.Conv2d(256, 2, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.motion_encoder = MotionEncoder(device=device, dtype=dtype, operations=operations)
        self.recurrent_block = RecurrentBlock(device=device, dtype=dtype, operations=operations)
        self.flow_head = FlowHead(device=device, dtype=dtype, operations=operations)

    def forward(self, hidden, context, corr, flow):
        hidden = self.recurrent_block(hidden, torch.cat([context, self.motion_encoder(flow, corr)], dim=1))
        return hidden, self.flow_head(hidden)


class MaskPredictor(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.convrelu = conv_norm_relu(128, 256, 3, device=device, dtype=dtype, operations=operations)
        self.conv = operations.Conv2d(256, 8 * 8 * 9, 1, device=device, dtype=dtype)

    def forward(self, x):
        return self.conv(self.convrelu(x)) * 0.25


def lookup(pyramid, coords, scales, offsets):
    # bilinear correlation samples in a (2 RADIUS + 1)^2 window around coords on each pyramid level
    _, _, h, w = coords.shape
    centroids = coords.permute(0, 2, 3, 1).reshape(h * w, 1, 1, 2)
    out = [F.grid_sample(corr, torch.addcmul(offset, centroids, scale), mode="bilinear", align_corners=True).view(1, h, w, -1)
           for corr, scale, offset in zip(pyramid, scales, offsets)]
    return torch.cat(out, dim=-1).permute(0, 3, 1, 2)


def upsample_flow(flow, mask):
    # each pixel of the 8x finer flow is a convex combination of its 3x3 coarse neighbours
    b, _, h, w = flow.shape
    mask = mask.view(b, 1, 9, 8, 8, h, w).softmax(dim=2)
    up = F.unfold(flow * 8, kernel_size=3, padding=1).view(b, 2, 9, 1, 1, h, w)
    return (mask * up).sum(dim=2).permute(0, 1, 4, 2, 5, 3).reshape(b, 2, h * 8, w * 8)


class RAFT(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.feature_encoder = Encoder(nn.InstanceNorm2d, device=device, dtype=dtype, operations=operations)
        self.context_encoder = Encoder(lambda c: operations.BatchNorm2d(c, device=device, dtype=dtype), device=device, dtype=dtype, operations=operations)
        self.update_block = UpdateBlock(device=device, dtype=dtype, operations=operations)
        self.mask_predictor = MaskPredictor(device=device, dtype=dtype, operations=operations)

    def get_dynamic_vram__units(self):
        return [self.update_block], []

    def init_state(self, h, w, device, dtype):
        d = torch.linspace(-RADIUS, RADIUS, 2 * RADIUS + 1, device=device)
        delta = torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1)
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        state = {"pyramid": [], "scales": [], "offsets": [], "coords0": torch.stack([xs, ys])[None]}
        for i in range(LEVELS):
            size = torch.tensor([(w >> i) - 1, (h >> i) - 1], device=device, dtype=torch.float32)
            state["pyramid"].append(torch.empty((h * w, 1, h >> i, w >> i), device=device))
            # grid_sample coordinates of level i: (coords / 2^i + delta) * 2 / (size - 1) - 1
            state["scales"].append(2 / (size * 2 ** i))
            state["offsets"].append(2 * delta / size - 1)
        state["coords1"] = torch.empty_like(state["coords0"])
        # channels last suits the update block's convolutions, and the correlation lookup already comes out that way
        state["hidden"] = torch.empty((1, 128, h, w), device=device, dtype=dtype, memory_format=torch.channels_last)
        state["context"] = torch.empty((1, 128, h, w), device=device, dtype=dtype, memory_format=torch.channels_last)
        return state

    def update(self, state):
        coords1, hidden = state["coords1"], state["hidden"]
        corr = lookup(state["pyramid"], coords1, state["scales"], state["offsets"]).to(hidden.dtype)
        flow = (coords1 - state["coords0"]).to(hidden.dtype, memory_format=torch.channels_last)
        new_hidden, delta = self.update_block(hidden, state["context"], corr, flow)
        hidden.copy_(new_hidden)
        coords1.add_(delta)

    def forward(self, image1, image2, state):
        # images (1, 3, H, W) in [-1, 1], H and W multiples of 8; returns the float32 flow in pixels mapping image1 onto image2
        # state holds one video's buffers, which the update loop reads and writes in place so its captured graph stays valid
        # correlation and coordinates stay float32: half precision costs 2-3x the flow error
        fmap1, fmap2 = self.feature_encoder(torch.cat([image1, image2])).float().chunk(2)
        _, c, h, w = fmap1.shape
        if not state:
            state.update(self.init_state(h, w, image1.device, image1.dtype))
        # correlating against fmap2 pooled i times equals pooling the correlation volume i times, at a fraction of the cost
        fmap2 = fmap2 / c ** 0.5
        for i, corr in enumerate(state["pyramid"]):
            if i:
                fmap2 = F.avg_pool2d(fmap2, 2)
            torch.mm(fmap1.view(c, h * w).t(), fmap2.view(c, -1), out=corr.view(h * w, -1))
        hidden, context = self.context_encoder(image1).split(128, dim=1)
        torch.tanh(hidden, out=state["hidden"])
        state["context"].copy_(context).relu_()
        state["coords1"].copy_(state["coords0"])

        device = image1.device
        queue = comfy.model_prefetch.make_prefetch_queue([self.update_block] * ITERATIONS, device, {"prefetch_dynamic_vbars": True})
        comfy.model_prefetch.malloc_graph_begin(device)
        for _ in range(ITERATIONS):
            comfy.model_prefetch.prefetch_queue_pop(queue, device, self.update_block, image1.dtype, core=lambda: self.update(state), enable_graph=True, malloc_scope="block")
        comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
        return upsample_flow(state["coords1"] - state["coords0"], self.mask_predictor(state["hidden"]).float())
