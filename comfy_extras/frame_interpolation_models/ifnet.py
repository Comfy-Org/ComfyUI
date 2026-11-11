import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init


def _warp(img, flow, warp_grids):
    B, _, H, W = img.shape
    base_grid, flow_div = warp_grids[(H, W)]
    flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
    grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(img.dtype)


class Head(nn.Module):
    def __init__(self, out_ch=4, device=None, dtype=None, operations=ops):
        super().__init__()
        self.cnn0 = operations.Conv2d(3, 16, 3, 2, 1, device=device, dtype=dtype)
        self.cnn1 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn2 = operations.Conv2d(16, 16, 3, 1, 1, device=device, dtype=dtype)
        self.cnn3 = operations.ConvTranspose2d(16, out_ch, 4, 2, 1, device=device, dtype=dtype)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c, device=None, dtype=None, operations=ops):
        super().__init__()
        self.conv = operations.Conv2d(c, c, 3, 1, 1, device=device, dtype=dtype)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1), device=device, dtype=dtype))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(torch.addcmul(x, self.conv(x), self.beta))


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, device=None, dtype=None, operations=ops):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Sequential(operations.Conv2d(in_planes, c // 2, 3, 2, 1, device=device, dtype=dtype), nn.LeakyReLU(0.2, True)),
            nn.Sequential(operations.Conv2d(c // 2, c, 3, 2, 1, device=device, dtype=dtype), nn.LeakyReLU(0.2, True)))
        self.convblock = nn.Sequential(*(ResConv(c, device=device, dtype=dtype, operations=operations) for _ in range(8)))
        self.lastconv = nn.Sequential(operations.ConvTranspose2d(c, 4 * 13, 4, 2, 1, device=device, dtype=dtype), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear").div_(scale)
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self, head_ch=4, channels=(192, 128, 96, 64, 32), device=None, dtype=None, operations=ops):
        super().__init__()
        self.encode = Head(out_ch=head_ch, device=device, dtype=dtype, operations=operations)
        block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
        self.blocks = nn.ModuleList([IFBlock(block_in[i], channels[i], device=device, dtype=dtype, operations=operations) for i in range(5)])
        self.scale_list = [16, 8, 4, 2, 1]
        self.pad_align = 64
        self._warp_grids = {}

    def get_dtype(self):
        return self.encode.cnn0.weight.dtype

    def memory_used_forward(self, shape, dtype):
        return 300 * shape[1] * shape[2] * dtype.itemsize

    def _build_warp_grids(self, H, W, device):
        if (H, W) in self._warp_grids:
            return
        self._warp_grids = {}  # clear old resolution grids to prevent memory leaks
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32), indexing="ij")
        self._warp_grids[(H, W)] = (
            torch.stack((grid_x, grid_y), dim=0).unsqueeze(0),
            torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], dtype=torch.float32, device=device))

    def warp(self, img, flow):
        return _warp(img, flow, self._warp_grids)

    def extract_features(self, img):
        """Extract head features for a single frame. Can be cached across pairs."""
        return self.encode(img)

    def forward(self, img0, img1, timestep=0.5, cache=None):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, device=img0.device, dtype=img0.dtype)

        self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)

        B = img0.shape[0]
        f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else self.encode(img0)
        f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else self.encode(img1)
        flow = mask = feat = None
        warped_img0, warped_img1 = img0, img1
        for i, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[i])
            else:
                fd, mask, feat = block(
                    torch.cat((warped_img0, warped_img1, self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4]), timestep, mask, feat), 1),
                    flow, scale=self.scale_list[i])
                flow = flow.add_(fd)
            warped_img0 = self.warp(img0, flow[:, :2])
            warped_img1 = self.warp(img1, flow[:, 2:4])
        return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))


def detect_rife_config(state_dict):
    head_ch = state_dict["encode.cnn3.weight"].shape[1]  # ConvTranspose2d: (in_ch, out_ch, kH, kW)
    channels = []
    for i in range(5):
        key = f"blocks.{i}.conv0.1.0.weight"
        if key in state_dict:
            channels.append(state_dict[key].shape[0])
    if len(channels) != 5:
        raise ValueError(f"Unsupported RIFE model: expected 5 blocks, found {len(channels)}")
    return head_ch, channels
