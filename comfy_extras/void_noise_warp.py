"""Optical-flow-warped noise for VOID pass 2.

Adapted from noise_warp.py and raft.py of RyannDaGreat/CommonSource (MIT License, Ryan Burgert):
https://github.com/RyannDaGreat/CommonSource
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.model_management


def _torch_resize_chw(image, size, interp, copy=True):
    """Resize a CHW tensor by a scalar factor or to an (h, w) tuple; copy=False may return the input itself."""
    if image.ndim != 3:
        raise ValueError(
            f"_torch_resize_chw expects a 3D CHW tensor, got shape {tuple(image.shape)}"
        )
    _, in_h, in_w = image.shape
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        new_h = max(1, int(in_h * size))
        new_w = max(1, int(in_w * size))
    else:
        new_h, new_w = size

    if (new_h, new_w) == (in_h, in_w):
        return image.clone() if copy else image

    kwargs = {}
    if interp in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    out = F.interpolate(image[None], size=(new_h, new_w), mode=interp, **kwargs)[0]
    return out


def _torch_remap_relative(image, dx, dy, interp="bilinear"):
    """Backward-warp a CHW image by per-pixel offsets (dx, dy), zero outside the image."""
    if image.ndim != 3:
        raise ValueError(
            f"_torch_remap_relative expects a 3D CHW tensor, got shape {tuple(image.shape)}"
        )
    if dx.shape != dy.shape:
        raise ValueError(
            f"_torch_remap_relative: dx and dy must match, got {tuple(dx.shape)} vs {tuple(dy.shape)}"
        )
    _, h, w = image.shape

    x_abs = dx + torch.arange(w, device=dx.device, dtype=dx.dtype)
    y_abs = dy + torch.arange(h, device=dy.device, dtype=dy.dtype)[:, None]

    x_norm = (x_abs / (w - 1)) * 2 - 1
    y_norm = (y_abs / (h - 1)) * 2 - 1

    grid = torch.stack([x_norm, y_norm], dim=-1)[None].to(image.dtype)
    out = F.grid_sample(
        image[None], grid, mode=interp, align_corners=True, padding_mode="zeros"
    )[0]
    return out


def _torch_scatter_add_relative(image, dx, dy):
    """Scatter-add a CHW image to its floor(dx, dy)-offset positions, dropping out-of-bounds targets."""
    if image.ndim != 3:
        raise ValueError(
            f"_torch_scatter_add_relative expects a 3D CHW tensor, got shape {tuple(image.shape)}"
        )
    in_c, in_h, in_w = image.shape
    if dx.shape != (in_h, in_w) or dy.shape != (in_h, in_w):
        raise ValueError(
            f"_torch_scatter_add_relative: dx/dy must be ({in_h}, {in_w}), "
            f"got dx={tuple(dx.shape)} dy={tuple(dy.shape)}"
        )

    x = dx.long() + torch.arange(in_w, device=dx.device, dtype=torch.long)
    y = dy.long() + torch.arange(in_h, device=dy.device, dtype=torch.long)[:, None]

    valid = ((y >= 0) & (y < in_h) & (x >= 0) & (x < in_w)).reshape(-1)
    indices = (y * in_w + x).reshape(-1)[valid]

    flat_image = image.reshape(in_c, -1).t()[valid]
    out = torch.zeros((in_h * in_w, in_c), dtype=image.dtype, device=image.device)
    out.index_add_(0, indices, flat_image)
    return out.t().reshape(in_c, in_h, in_w)


def unique_pixels(image):
    """Unique colors (U, C), their counts (U) and the per-pixel color index (H, W) of a CHW tensor."""
    c, h, w = image.shape
    flat = image.reshape(c, -1).t()
    unique_colors, inverse_indices, counts = torch.unique(
        flat, dim=0, return_inverse=True, return_counts=True, sorted=False,
    )
    index_matrix = inverse_indices.view(h, w)
    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """For each unique index, sum the CHW image values at its pixels."""
    u = int(index_matrix.max().item()) + 1
    flat = image.reshape(image.shape[0], -1).t()
    out = torch.zeros((u, flat.shape[1]), dtype=flat.dtype, device=flat.device)
    out.index_add_(0, index_matrix.view(-1), flat)
    return out


def indexed_to_image(index_matrix, unique_colors):
    """Build a CHW image from an index matrix and a (U, C) color table."""
    h, w = index_matrix.shape
    flat = unique_colors[index_matrix.view(-1)]
    return flat.t().reshape(-1, h, w)


def regaussianize(noise):
    """Re-whiten groups of duplicated pixels: scale by 1/sqrt(count) and add zero-mean noise within each group."""
    _, hs, ws = noise.shape
    _, counts, index_matrix = unique_pixels(noise[:1])

    foreign_noise = torch.randn_like(noise)
    summed = sum_indexed_values(foreign_noise, index_matrix)
    meaned = indexed_to_image(index_matrix, summed / counts[:, None])
    zeroed_foreign = foreign_noise - meaned

    counts_image = indexed_to_image(index_matrix, counts[:, None])

    output = noise / counts_image ** 0.5 + zeroed_foreign
    return output, counts_image


def xy_meshgrid_like_image(image):
    """Return a (2, H, W) tensor of (x, y) pixel coordinates matching ``image``."""
    _, h, w = image.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    return torch.stack([x, y])


def noise_to_state(noise):
    """Pack a (C, H, W) noise tensor into a state tensor (3+C, H, W) = [dx, dy, ω, noise]."""
    zeros = torch.zeros_like(noise[:1])
    ones = torch.ones_like(noise[:1])
    return torch.cat([zeros, zeros, ones, noise])


def state_to_noise(state):
    """Unpack the noise channels from a state tensor."""
    return state[3:]


def warp_state(state, flow):
    """Warp a (3+C, H, W) state of [dx, dy, ω, noise] along a (2, H, W) flow."""
    if flow.device != state.device:
        raise ValueError(
            f"warp_state: flow and state must be on the same device, "
            f"got flow={flow.device} state={state.device}"
        )
    if state.ndim != 3:
        raise ValueError(
            f"warp_state: state must be 3D (3+C, H, W), got shape {tuple(state.shape)}"
        )
    xyoc, h, w = state.shape
    if flow.shape != (2, h, w):
        raise ValueError(
            f"warp_state: flow must have shape (2, {h}, {w}), got {tuple(flow.shape)}"
        )
    device = state.device

    x_ch, y_ch = 0, 1
    xy = 2
    xyw = 3
    w_ch = 2
    c = xyoc - xyw
    oc = xyoc - xy
    if c <= 0:
        raise ValueError(
            f"warp_state: state has no noise channels (expected 3+C with C>0, got {xyoc} channels)"
        )
    if not (state[w_ch] > 0).all():
        raise ValueError("warp_state: all weights in state[2] must be > 0")

    grid = xy_meshgrid_like_image(state)

    init = torch.empty_like(state)
    init[:xy] = 0
    init[w_ch] = 1
    init[-c:] = 0

    # expansion: nearest remap along the negated flow
    pre_expand = torch.empty_like(state)
    pre_expand[:xy] = _torch_remap_relative(state[:xy], -flow[0], -flow[1], "nearest")
    pre_expand[-oc:] = _torch_remap_relative(state[-oc:], -flow[0], -flow[1], "nearest")
    pre_expand[w_ch][pre_expand[w_ch] == 0] = 1

    # shrink: scatter-add the state into its new positions
    pre_shrink = state.clone()
    pre_shrink[:xy] += flow

    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (pos[x_ch] >= 0) & (pos[x_ch] < w) & (pos[y_ch] >= 0) & (pos[y_ch] < h)
    pre_shrink = torch.where(~in_bounds[None], init, pre_shrink)

    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy
    pre_shrink[:xy] = 0

    def scat(tensor):
        return _torch_scatter_add_relative(tensor, scat_xy[0], scat_xy[1])

    shrink_mask = scat(torch.ones(1, h, w, dtype=state.dtype, device=device)) > 0

    pre_expand = torch.where(shrink_mask, init, pre_expand)

    # regaussianize both branches together so duplicate groups are counted across them
    concat = torch.cat([pre_shrink, pre_expand], dim=2)
    concat[-c:], counts_image = regaussianize(concat[-c:])
    concat[w_ch] = concat[w_ch] / counts_image[0]
    concat[w_ch] = concat[w_ch].nan_to_num()
    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=2)

    shrink = torch.empty_like(pre_shrink)
    shrink[w_ch] = scat(pre_shrink[w_ch][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy] * pre_shrink[w_ch][None]) / shrink[w_ch][None]
    shrink[-c:] = scat(pre_shrink[-c:] * pre_shrink[w_ch][None]) / scat(
        pre_shrink[w_ch][None] ** 2
    ).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[w_ch] = output[w_ch] / output[w_ch].mean()
    output[w_ch] += 1e-5
    output[w_ch] **= 0.9999
    return output


class NoiseWarper:
    """Warpable noise state that emits gaussian noise per frame."""

    def __init__(self, c, h, w, device, dtype=torch.float32):
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError(
                f"NoiseWarper: c/h/w must all be positive, got c={c} h={h} w={w}"
            )
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        self.dtype = dtype

        noise = torch.randn(c, h, w, dtype=dtype, device=device)
        self._state = noise_to_state(noise)

    @property
    def noise(self):
        n = state_to_noise(self._state)
        weights = self._state[2:3]
        return n * weights / (weights ** 2).sqrt()

    def __call__(self, dx, dy):
        if dx.shape != dy.shape:
            raise ValueError(
                f"NoiseWarper: dx and dy must match, got {tuple(dx.shape)} vs {tuple(dy.shape)}"
            )
        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape

        flow = _torch_resize_chw(flow, (self.h, self.w), "bilinear", copy=True)
        flowh, floww = flow.shape[-2:]

        # upstream scales dx by the height ratio and dy by the width ratio; both ratios are equal here
        flow[0] *= flowh / oflowh
        flow[1] *= floww / ofloww

        self._state = warp_state(self._state, flow)
        return self


class RaftOpticalFlow:
    """Runs a loaded RAFT model on the frames of one video; __call__ returns a (2, H, W) flow."""

    def __init__(self, model):
        self.model = model
        self.state = {}

    def _preprocess(self, image_chw, size):
        image = _torch_resize_chw(image_chw, size, "bilinear", copy=False)
        return (image[None] * 2 - 1).to(self.model.dtype)

    def __call__(self, from_image, to_image):
        """``from_image``, ``to_image``: CHW float tensors in [0, 1] on the model's device."""
        _, h, w = from_image.shape
        # multiples of 8, and at least 128 px so the coarsest of the four correlation levels is 2 px wide
        size = (max(128, h // 8 * 8), max(128, w // 8 * 8))
        flow = self.model(self._preprocess(from_image, size), self._preprocess(to_image, size), self.state)[0]
        if size != (h, w):
            flow = _torch_resize_chw(flow, (h, w), "bilinear", copy=False)
            flow[0] *= w / size[1]
            flow[1] *= h / size[0]
        return flow


def get_noise_from_video(
    video_frames: torch.Tensor,
    raft: RaftOpticalFlow,
    *,
    noise_channels: int = 16,
    resize_frames: float = 0.5,
    resize_flow: int = 8,
    downscale_factor: int = 32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Flow-warped gaussian noise for a (T, H, W, 3) uint8 video, as (T, H', W', noise_channels) float32.

    Frames are scaled by resize_frames before RAFT, the noise state lives at resize_flow times that size
    and is area-pooled by downscale_factor on output.
    """
    if not isinstance(resize_flow, int) or resize_flow < 1:
        raise ValueError(
            f"get_noise_from_video: resize_flow must be a positive int, got {resize_flow!r}"
        )
    if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
        raise ValueError(
            "get_noise_from_video: video_frames must have shape (T, H, W, 3), "
            f"got {tuple(video_frames.shape)}"
        )
    if video_frames.dtype != torch.uint8:
        raise TypeError(
            "get_noise_from_video: video_frames must be uint8 in [0, 255], "
            f"got dtype {video_frames.dtype}"
        )

    if device is None:
        device = comfy.model_management.get_torch_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cpu":
        logging.warning(
            "VOIDWarpedNoise: running get_noise_from_video on CPU; this will be "
            "slow (minutes for ~45 frames).  Use CUDA for interactive use."
        )

    T = video_frames.shape[0]
    frames = video_frames.to(device).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    if resize_frames != 1.0:
        new_h = max(1, int(frames.shape[2] * resize_frames))
        new_w = max(1, int(frames.shape[3] * resize_frames))
        frames = F.interpolate(frames, size=(new_h, new_w), mode="area")

    _, _, H, W = frames.shape
    internal_h = resize_flow * H
    internal_w = resize_flow * W
    if internal_h % downscale_factor or internal_w % downscale_factor:
        logging.warning(
            "VOIDWarpedNoise: internal noise size %dx%d is not divisible by "
            "downscale_factor %d; output noise may have artifacts.",
            internal_h, internal_w, downscale_factor,
        )

    warper = NoiseWarper(
        c=noise_channels, h=internal_h, w=internal_w, device=device,
    )
    down_h = warper.h // downscale_factor
    down_w = warper.w // downscale_factor
    output = torch.empty(
        (T, down_h, down_w, noise_channels), dtype=torch.float32, device=device,
    )

    def downscale(noise_chw):
        # area pooling divides the std by downscale_factor
        down = _torch_resize_chw(noise_chw, 1.0 / downscale_factor, "area", copy=False)
        return down * downscale_factor

    output[0] = downscale(warper.noise).permute(1, 2, 0)

    prev = frames[0]
    for i in range(1, T):
        curr = frames[i]
        flow = raft(prev, curr)
        warper(flow[0], flow[1])
        output[i] = downscale(warper.noise).permute(1, 2, 0)
        prev = curr

    return output
