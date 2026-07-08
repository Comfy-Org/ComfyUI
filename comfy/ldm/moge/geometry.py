"""Pure-torch + scipy geometry helpers for MoGe inference and mesh export."""


from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.optimize import least_squares

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: Optional[float] = None,
                             dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """Normalized view-plane UV coordinates with corners at +/-(W, H)/diagonal."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1.0 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def intrinsics_from_focal_center(fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """Assemble (..., 3, 3) intrinsics from broadcastable fx, fy, cx, cy."""
    fx, fy, cx, cy = [torch.as_tensor(v) for v in (fx, fy, cx, cy)]
    fx, fy, cx, cy = torch.broadcast_tensors(fx, fy, cx, cy)
    zero = torch.zeros_like(fx)
    one = torch.ones_like(fx)
    return torch.stack([
        torch.stack([fx,   zero, cx], dim=-1),
        torch.stack([zero, fy,   cy], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)


def depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Back-project a (..., H, W) depth map through K^-1 to (..., H, W, 3) camera-space points.

    Intrinsics use normalized image coords (x in [0, 1] left->right, y in [0, 1] top->bottom).
    """
    H, W = depth.shape[-2:]
    device, dtype = depth.device, depth.dtype
    u = (torch.arange(W, dtype=dtype, device=device) + 0.5) / W
    v = (torch.arange(H, dtype=dtype, device=device) + 0.5) / H
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    pix = torch.stack([grid_u, grid_v, torch.ones_like(grid_u)], dim=-1)
    K_inv = torch.linalg.inv(intrinsics)
    rays = torch.einsum("...ij,hwj->...hwi", K_inv, pix)
    return rays * depth.unsqueeze(-1)


def _solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray,
                         focal: Optional[float] = None) -> Tuple[float, float]:
    """LM-solve for z-shift; when focal is None, also recovers the optimal focal."""
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def fn(shift):
        xy_proj = xy / (z + shift)[:, None]
        f = focal if focal is not None else (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (f * xy_proj - uv).ravel()

    sol = least_squares(fn, x0=0.0, ftol=1e-3, method="lm")
    shift = float(np.asarray(sol["x"]).squeeze())
    if focal is None:
        xy_proj = xy / (z + shift)[:, None]
        focal = float((xy_proj * uv).sum() / np.square(xy_proj).sum())
    return shift, focal


def recover_focal_shift(points: torch.Tensor, mask: Optional[torch.Tensor] = None,
                        focal: Optional[torch.Tensor] = None, downsample_size: Tuple[int, int] = (64, 64)
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover the focal length and z-shift that turn points into a metric point map.

    Optical center is at the image center; returned focal is relative to half the image diagonal.
    Returns (focal, shift) on the same device/dtype as points.
    """
    shape = points.shape
    H, W = shape[-3], shape[-2]
    points_b = points.reshape(-1, H, W, 3)
    mask_b = None if mask is None else mask.reshape(-1, H, W)
    focal_b = None if focal is None else focal.reshape(-1)

    uv = normalized_view_plane_uv(W, H, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points_b.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest").squeeze(0).permute(1, 2, 0)
    mask_lr = None
    if mask_b is not None:
        mask_lr = F.interpolate(mask_b.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1) > 0

    uv_np = uv_lr.detach().cpu().numpy()
    points_np = points_lr.detach().cpu().numpy()
    mask_np = None if mask_lr is None else mask_lr.detach().cpu().numpy()
    focal_np = None if focal_b is None else focal_b.detach().cpu().numpy()

    out_focal: list = []
    out_shift: list = []
    for i in range(points_b.shape[0]):
        if mask_np is None:
            xyz_i = points_np[i].reshape(-1, 3)
            uv_i = uv_np.reshape(-1, 2)
        else:
            sel = mask_np[i]
            if sel.sum() < 2:
                out_focal.append(1.0)
                out_shift.append(0.0)
                continue
            xyz_i = points_np[i][sel]
            uv_i = uv_np[sel]
        if focal_np is None:
            shift_i, focal_i = _solve_optimal_shift(uv_i, xyz_i)
            out_focal.append(focal_i)
        else:
            shift_i, _ = _solve_optimal_shift(uv_i, xyz_i, focal=float(focal_np[i]))
        out_shift.append(shift_i)

    shift_t = torch.tensor(out_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal is None:
        focal_t = torch.tensor(out_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        focal_t = focal.reshape(shape[:-3])
    return focal_t, shift_t


def depth_map_edge(depth: torch.Tensor, atol: Optional[float] = None, rtol: Optional[float] = None, kernel_size: int = 3) -> torch.Tensor:
    """Per-pixel boolean: True where the local depth window's max-min span exceeds atol or rtol*depth."""
    shape = depth.shape
    d = depth.reshape(-1, 1, *shape[-2:])
    pad = kernel_size // 2
    diff = F.max_pool2d(d, kernel_size, stride=1, padding=pad) + F.max_pool2d(-d, kernel_size, stride=1, padding=pad)
    edge = torch.zeros_like(d, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / d.clamp_min(1e-6)).nan_to_num_() > rtol
    return edge.reshape(*shape)


def triangulate_grid_mesh(points: torch.Tensor, mask: Optional[torch.Tensor] = None, decimation: int = 1, discontinuity_threshold: float = 0.04,
                          depth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triangulate a (H, W, 3) point map into (vertices, faces, uvs) on CPU.

    Vertices: pixels with finite coords (passing optional mask).  Quads with four valid corners
    become two triangles.  depth overrides the scalar used for the rtol edge check; pass radial
    depth for panoramas (the default points[..., 2] goes negative below the equator).
    """
    points = points.detach().cpu()
    finite = torch.isfinite(points).all(dim=-1)
    if mask is None:
        mask = finite
    else:
        mask = mask.detach().cpu().to(torch.bool) & finite

    if discontinuity_threshold > 0:
        d = depth.detach().cpu() if depth is not None else points[..., 2]
        # Replace inf with 0 so max-pool doesn't poison neighbourhoods (mask above already excludes those pixels).
        d_finite = torch.where(finite, d, torch.zeros_like(d))
        edge = depth_map_edge(d_finite, rtol=discontinuity_threshold)
        mask = mask & ~edge

    if decimation > 1:
        points = points[::decimation, ::decimation].contiguous()
        mask = mask[::decimation, ::decimation].contiguous()
    H, W = points.shape[:2]

    flat_mask = mask.reshape(-1)
    idx = torch.full((H * W,), -1, dtype=torch.long)
    n_valid = int(flat_mask.sum().item())
    idx[flat_mask] = torch.arange(n_valid, dtype=torch.long)
    idx = idx.reshape(H, W)

    vertices = points.reshape(-1, 3)[flat_mask].contiguous()

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    u = xx.float() / max(W - 1, 1)
    v = yy.float() / max(H - 1, 1)
    uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)[flat_mask].contiguous()

    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    quad_ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[quad_ok], b[quad_ok], c[quad_ok], d[quad_ok]
    faces = torch.cat([torch.stack([a, b, c], dim=-1), torch.stack([a, c, d], dim=-1)], dim=0).contiguous()
    return vertices, faces, uvs
