"""Input/output preprocessing helpers for Depth Anything 3."""

from __future__ import annotations

from typing import Tuple

import torch

import comfy.utils

PATCH_SIZE = 14

# ImageNet normalization constants used during DA3 training.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _round_to_patch(x: int, patch: int = PATCH_SIZE) -> int:
    down = (x // patch) * patch
    up = down + patch
    return up if abs(up - x) <= abs(x - down) else down


def compute_target_size(orig_h: int, orig_w: int, process_res: int, method: str = "upper_bound_resize") -> Tuple[int, int]:
    """Compute (target_h, target_w) for a single image.
    upper_bound_resize: scale longest side to process_res, then round each dim to nearest multiple of 14 (default upstream method).
    lower_bound_resize: scale shortest side to process_res, then round."""

    if method == "upper_bound_resize":
        longest = max(orig_h, orig_w)
        scale = process_res / float(longest)
    elif method == "lower_bound_resize":
        shortest = min(orig_h, orig_w)
        scale = process_res / float(shortest)
    else:
        raise ValueError(f"Unsupported process_res_method: {method}")

    new_w = max(1, _round_to_patch(int(round(orig_w * scale))))
    new_h = max(1, _round_to_patch(int(round(orig_h * scale))))
    return new_h, new_w


def preprocess_image(image: torch.Tensor, process_res: int = 504, method: str = "upper_bound_resize") -> torch.Tensor:
    assert image.ndim == 4 and image.shape[-1] == 3, f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
    B, H, W, _ = image.shape
    target_h, target_w = compute_target_size(H, W, process_res, method)

    # (B, H, W, 3) -> (B, 3, H, W)
    x = image.movedim(-1, 1).contiguous()
    if (target_h, target_w) != (H, W):
        # Upstream uses cv2 INTER_CUBIC (upscale) / INTER_AREA (downscale).
        # Lanczos in ``common_upscale`` is anti-aliased and produces the
        # closest pixel-wise match in a sweep across {bilinear, bicubic,
        # area, lanczos, bislerp}. Used in both directions for simplicity.
        x = comfy.utils.common_upscale(x.float(), target_w, target_h, "lanczos", "disabled",)
    x = x.clamp(0.0, 1.0)

    mean = _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = _IMAGENET_STD.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


# -----------------------------------------------------------------------------
# Output post-processing (sky-aware clipping for Mono/Metric variants)
# -----------------------------------------------------------------------------


def compute_non_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Boolean mask: True for non-sky pixels (sky probability < threshold)."""
    return sky_prediction < threshold


def apply_sky_aware_clip(depth: torch.Tensor, sky: torch.Tensor, threshold: float = 0.3, quantile: float = 0.99) -> torch.Tensor:
    """Clips sky regions to the 99th percentile of non-sky depth. Returns a new depth tensor."""
    non_sky = compute_non_sky_mask(sky, threshold=threshold)
    if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
        return depth.clone()

    non_sky_depth = depth[non_sky]
    if non_sky_depth.numel() > 100_000:
        idx = torch.randint(0, non_sky_depth.numel(), (100_000,), device=non_sky_depth.device)
        sampled = non_sky_depth[idx]
    else:
        sampled = non_sky_depth

    max_depth = torch.quantile(sampled, quantile)
    out = depth.clone()
    out[~non_sky] = max_depth
    return out


def normalize_depth_v2_style(depth: torch.Tensor, sky: torch.Tensor | None = None, low_quantile: float = 0.01, high_quantile: float = 0.99) -> torch.Tensor:
    """V2-style normalization computes percentile bounds over non-sky pixels (when available), then maps depth into [0, 1] with near = white (1.0)."""
    if sky is not None:
        mask = compute_non_sky_mask(sky)
        if mask.any():
            valid = depth[mask]
        else:
            valid = depth.flatten()
    else:
        valid = depth.flatten()

    if valid.numel() > 100_000:
        idx = torch.randint(0, valid.numel(), (100_000,), device=valid.device)
        sample = valid[idx]
    else:
        sample = valid

    lo = torch.quantile(sample, low_quantile)
    hi = torch.quantile(sample, high_quantile)
    rng = (hi - lo).clamp(min=1e-6)
    norm = ((depth - lo) / rng).clamp(0.0, 1.0)
    # Nearer pixels are brighter (1.0)
    norm = 1.0 - norm
    if sky is not None:
        # Sky pixels become black (far / unknown)
        sky_mask = ~compute_non_sky_mask(sky)
        norm = torch.where(sky_mask, torch.zeros_like(norm), norm)
    return norm


def normalize_depth_min_max(depth: torch.Tensor) -> torch.Tensor:
    """Simple per-frame min/max normalization with near=1.0 convention."""
    lo = depth.amin(dim=(-2, -1), keepdim=True)
    hi = depth.amax(dim=(-2, -1), keepdim=True)
    rng = (hi - lo).clamp(min=1e-6)
    return 1.0 - ((depth - lo) / rng).clamp(0.0, 1.0)
