"""FILM: Frame Interpolation for Large Motion (ECCV 2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops

ops = comfy.ops.disable_weight_init


class FilmConv2d(nn.Module):
    """Conv2d with optional LeakyReLU and FILM-style padding."""

    def __init__(self, in_channels, out_channels, size, activation=True, device=None, dtype=None, operations=ops):
        super().__init__()
        self.even_pad = not size % 2
        self.conv = operations.Conv2d(in_channels, out_channels, kernel_size=size, padding=size // 2 if size % 2 else 0, device=device, dtype=dtype)
        self.activation = nn.LeakyReLU(0.2) if activation else None

    def forward(self, x):
        if self.even_pad:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _warp_core(image, flow, grid_x, grid_y):
    dtype = image.dtype
    H, W = flow.shape[2], flow.shape[3]
    dx = flow[:, 0].float() / (W * 0.5)
    dy = flow[:, 1].float() / (H * 0.5)
    grid = torch.stack([grid_x[None, None, :] + dx, grid_y[None, :, None] + dy], dim=3)
    return F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="border", align_corners=False).to(dtype)


def build_image_pyramid(image, pyramid_levels):
    pyramid = [image]
    for _ in range(1, pyramid_levels):
        image = F.avg_pool2d(image, 2, 2)
        pyramid.append(image)
    return pyramid


def flow_pyramid_synthesis(residual_pyramid):
    flow = residual_pyramid[-1]
    flow_pyramid = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        flow = F.interpolate(flow, size=residual_flow.shape[2:4], mode="bilinear", scale_factor=None).mul_(2).add_(residual_flow)
        flow_pyramid.append(flow)
    flow_pyramid.reverse()
    return flow_pyramid


def multiply_pyramid(pyramid, scalar):
    return [image * scalar[:, None, None, None] for image in pyramid]


def pyramid_warp(feature_pyramid, flow_pyramid, warp_fn):
    return [warp_fn(features, flow) for features, flow in zip(feature_pyramid, flow_pyramid)]


def concatenate_pyramids(pyramid1, pyramid2):
    return [torch.cat([f1, f2], dim=1) for f1, f2 in zip(pyramid1, pyramid2)]


class SubTreeExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=64, n_layers=4, device=None, dtype=None, operations=ops):
        super().__init__()
        convs = []
        for i in range(n_layers):
            out_ch = channels << i
            convs.append(nn.Sequential(
                FilmConv2d(in_channels, out_ch, 3, device=device, dtype=dtype, operations=operations),
                FilmConv2d(out_ch, out_ch, 3, device=device, dtype=dtype, operations=operations)))
            in_channels = out_ch
        self.convs = nn.ModuleList(convs)

    def forward(self, image, n):
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, 2, 2)
        return pyramid


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, channels=64, sub_levels=4, device=None, dtype=None, operations=ops):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels, device=device, dtype=dtype, operations=operations)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid):
        sub_pyramids = [self.extract_sublevels(image_pyramid[i], min(len(image_pyramid) - i, self.sub_levels))
                        for i in range(len(image_pyramid))]
        feature_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
            # Free sub-pyramids no longer needed by future levels
            if i >= self.sub_levels - 1:
                sub_pyramids[i - self.sub_levels + 1] = None
        return feature_pyramid


class FlowEstimator(nn.Module):
    def __init__(self, in_channels, num_convs, num_filters, device=None, dtype=None, operations=ops):
        super().__init__()
        self._convs = nn.ModuleList()
        for _ in range(num_convs):
            self._convs.append(FilmConv2d(in_channels, num_filters, 3, device=device, dtype=dtype, operations=operations))
            in_channels = num_filters
        self._convs.append(FilmConv2d(in_channels, num_filters // 2, 1, device=device, dtype=dtype, operations=operations))
        self._convs.append(FilmConv2d(num_filters // 2, 2, 1, activation=False, device=device, dtype=dtype, operations=operations))

    def forward(self, features_a, features_b):
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(nn.Module):
    def __init__(self, filters=64, flow_convs=(3, 3, 3, 3), flow_filters=(32, 64, 128, 256), device=None, dtype=None, operations=ops):
        super().__init__()
        in_channels = filters << 1
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(FlowEstimator(in_channels, flow_convs[i], flow_filters[i], device=device, dtype=dtype, operations=operations))
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a, feature_pyramid_b, warp_fn):
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        # Coarse-to-fine: shared predictor for deep levels, then specialized predictors for fine levels
        steps = [(i, self._predictor) for i in range(levels - 2, len(self._predictors) - 1, -1)]
        steps += [(len(self._predictors) - 1 - k, p) for k, p in enumerate(self._predictors)]
        for i, predictor in steps:
            v = F.interpolate(v, size=feature_pyramid_a[i].shape[2:4], mode="bilinear").mul_(2)
            v_residual = predictor(feature_pyramid_a[i], warp_fn(feature_pyramid_b[i], v))
            residuals.append(v_residual)
            v = v.add_(v_residual)
        residuals.reverse()
        return residuals


def _get_fusion_channels(level, filters):
    # Per direction: multi-scale features + RGB image (3ch) + flow (2ch), doubled for both directions
    return (sum(filters << i for i in range(level)) + 3 + 2) * 2


class Fusion(nn.Module):
    def __init__(self, n_layers=4, specialized_layers=3, filters=64, device=None, dtype=None, operations=ops):
        super().__init__()
        self.output_conv = operations.Conv2d(filters, 3, kernel_size=1, device=device, dtype=dtype)
        self.convs = nn.ModuleList()
        in_channels = _get_fusion_channels(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            self.convs.append(nn.ModuleList([
                FilmConv2d(in_channels, num_filters, 2, activation=False, device=device, dtype=dtype, operations=operations),
                FilmConv2d(in_channels + (increase or num_filters), num_filters, 3, device=device, dtype=dtype, operations=operations),
                FilmConv2d(num_filters, num_filters, 3, device=device, dtype=dtype, operations=operations)]))
            in_channels = num_filters
            increase = _get_fusion_channels(i, filters) - num_filters // 2

    def forward(self, pyramid):
        net = pyramid[-1]
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            net = layers[0](F.interpolate(net, size=pyramid[i].shape[2:4], mode="nearest"))
            net = layers[2](layers[1](torch.cat([pyramid[i], net], dim=1)))
        return self.output_conv(net)


class FILMNet(nn.Module):
    def __init__(self, pyramid_levels=7, fusion_pyramid_levels=5, specialized_levels=3, sub_levels=4,
                 filters=64, flow_convs=(3, 3, 3, 3), flow_filters=(32, 64, 128, 256), device=None, dtype=None, operations=ops):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels
        self.extract = FeatureExtractor(3, filters, sub_levels, device=device, dtype=dtype, operations=operations)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters, device=device, dtype=dtype, operations=operations)
        self.fuse = Fusion(sub_levels, specialized_levels, filters, device=device, dtype=dtype, operations=operations)
        self._warp_grids = {}

    def get_dtype(self):
        return self.extract.extract_sublevels.convs[0][0].conv.weight.dtype

    def memory_used_forward(self, shape, dtype):
        return 1700 * shape[1] * shape[2] * dtype.itemsize

    def _build_warp_grids(self, H, W, device):
        """Pre-compute warp grids for all pyramid levels."""
        if (H, W) in self._warp_grids:
            return
        self._warp_grids = {}  # clear old resolution grids to prevent memory leaks
        for _ in range(self.pyramid_levels):
            self._warp_grids[(H, W)] = (
                torch.linspace(-(1 - 1 / W), 1 - 1 / W, W, dtype=torch.float32, device=device),
                torch.linspace(-(1 - 1 / H), 1 - 1 / H, H, dtype=torch.float32, device=device),
            )
            H, W = H // 2, W // 2

    def warp(self, image, flow):
        grid_x, grid_y = self._warp_grids[(flow.shape[2], flow.shape[3])]
        return _warp_core(image, flow, grid_x, grid_y)

    def extract_features(self, img):
        """Extract image and feature pyramids for a single frame. Can be cached across pairs."""
        image_pyramid = build_image_pyramid(img, self.pyramid_levels)
        feature_pyramid = self.extract(image_pyramid)
        return image_pyramid, feature_pyramid

    def forward(self, img0, img1, timestep=0.5, cache=None):
        # FILM uses a scalar timestep per batch element (spatially-varying timesteps not supported)
        t = timestep.mean(dim=(1, 2, 3)).item() if isinstance(timestep, torch.Tensor) else timestep
        return self.forward_multi_timestep(img0, img1, [t], cache=cache)

    def forward_multi_timestep(self, img0, img1, timesteps, cache=None):
        """Compute flow once, synthesize at multiple timesteps. Expects batch=1 inputs."""
        self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)

        image_pyr0, feat_pyr0 = cache["img0"] if cache and "img0" in cache else self.extract_features(img0)
        image_pyr1, feat_pyr1 = cache["img1"] if cache and "img1" in cache else self.extract_features(img1)

        fwd_flow = flow_pyramid_synthesis(self.predict_flow(feat_pyr0, feat_pyr1, self.warp))[:self.fusion_pyramid_levels]
        bwd_flow = flow_pyramid_synthesis(self.predict_flow(feat_pyr1, feat_pyr0, self.warp))[:self.fusion_pyramid_levels]

        # Build warp targets and free full pyramids (only first fpl levels needed from here)
        fpl = self.fusion_pyramid_levels
        p2w = [concatenate_pyramids(image_pyr0[:fpl], feat_pyr0[:fpl]),
               concatenate_pyramids(image_pyr1[:fpl], feat_pyr1[:fpl])]
        del image_pyr0, image_pyr1, feat_pyr0, feat_pyr1

        results = []
        dt_tensors = torch.tensor(timesteps, device=img0.device, dtype=img0.dtype)
        for idx in range(len(timesteps)):
            batch_dt = dt_tensors[idx:idx + 1]
            bwd_scaled = multiply_pyramid(bwd_flow, batch_dt)
            fwd_scaled = multiply_pyramid(fwd_flow, 1 - batch_dt)
            fwd_warped = pyramid_warp(p2w[0], bwd_scaled, self.warp)
            bwd_warped = pyramid_warp(p2w[1], fwd_scaled, self.warp)
            aligned = [torch.cat([fw, bw, bf, ff], dim=1)
                       for fw, bw, bf, ff in zip(fwd_warped, bwd_warped, bwd_scaled, fwd_scaled)]
            del fwd_warped, bwd_warped, bwd_scaled, fwd_scaled
            results.append(self.fuse(aligned))
            del aligned
        return torch.cat(results, dim=0)
