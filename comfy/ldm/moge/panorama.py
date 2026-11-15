"""Panorama (equirectangular) inference helpers for MoGe.

Splits an equirect into 12 perspective views via an icosahedron camera rig, runs
the model per view, and stitches per-view distance maps back into a single
equirect distance map via a multi-scale Poisson + gradient sparse solve.
Image sampling uses F.grid_sample (GPU); the sparse solve uses lsmr (CPU).
"""


from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import convolve, map_coordinates
from scipy.sparse import vstack, csr_array
from scipy.sparse.linalg import lsmr


def _icosahedron_directions() -> np.ndarray:
    """12 icosahedron-vertex directions (non-normalised, matching upstream's vertex order)."""
    A = (1.0 + np.sqrt(5.0)) / 2.0
    return np.array([
        [0,  1,  A], [0, -1,  A], [0,  1, -A], [0, -1, -A],
        [1,  A,  0], [-1,  A,  0], [1, -A,  0], [-1, -A,  0],
        [A,  0,  1], [A,  0, -1], [-A,  0,  1], [-A,  0, -1],
    ], dtype=np.float32)


def _intrinsics_from_fov(fov_x_rad: float, fov_y_rad: float) -> np.ndarray:
    """Normalised-image (unit-square) K matrix."""
    fx = 0.5 / np.tan(fov_x_rad / 2)
    fy = 0.5 / np.tan(fov_y_rad / 2)
    return np.array([[fx, 0, 0.5], [0, fy, 0.5], [0, 0, 1]], dtype=np.float32)


def _extrinsics_look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-convention world->camera extrinsics for an array of look-at targets (N, 4, 4)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    if target.ndim == 1:
        target = target[None]

    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True).clip(1e-12)
    right = np.cross(fwd, up)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    # Fall back to an arbitrary perpendicular if forward is parallel to up.
    parallel = right_norm.squeeze(-1) < 1e-6
    if parallel.any():
        alt_up = np.array([1, 0, 0], dtype=np.float32)
        right = np.where(parallel[:, None], np.cross(fwd, alt_up), right)
        right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    right = right / right_norm.clip(1e-12)
    new_up = np.cross(fwd, right)

    R = np.stack([right, new_up, fwd], axis=-2)
    t = -np.einsum("nij,j->ni", R, eye)
    E = np.zeros((R.shape[0], 4, 4), dtype=np.float32)
    E[:, :3, :3] = R
    E[:, :3, 3] = t
    E[:, 3, 3] = 1.0
    return E


def get_panorama_cameras() -> Tuple[np.ndarray, List[np.ndarray]]:
    """Returns (extrinsics (12, 4, 4), [intrinsics] * 12) for icosahedron views at 90 deg FoV."""
    targets = _icosahedron_directions()
    eye = np.zeros(3, dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    extrinsics = _extrinsics_look_at(eye, targets, up)
    K = _intrinsics_from_fov(np.deg2rad(90.0), np.deg2rad(90.0))
    return extrinsics, [K] * len(targets)


def spherical_uv_to_directions(uv: np.ndarray) -> np.ndarray:
    """Equirect UV in [0, 1] -> 3D unit-direction (Z up)."""
    theta = (1 - uv[..., 0]) * (2 * np.pi)
    phi = uv[..., 1] * np.pi
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=-1).astype(np.float32)


def directions_to_spherical_uv(directions: np.ndarray) -> np.ndarray:
    """3D direction -> equirect UV in [0, 1]."""
    n = np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-12)
    d = directions / n
    u = 1 - np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi) % 1.0
    v = np.arccos(d[..., 2].clip(-1, 1)) / np.pi
    return np.stack([u, v], axis=-1).astype(np.float32)


def _uv_grid(H: int, W: int) -> np.ndarray:
    """Pixel-center UV grid in [0, 1]; (H, W, 2)."""
    u = (np.arange(W, dtype=np.float32) + 0.5) / W
    v = (np.arange(H, dtype=np.float32) + 0.5) / H
    return np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1)


def _unproject_cv(uv: np.ndarray, depth: np.ndarray,
                  extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project pixels into world coords (OpenCV convention)."""
    pix = np.concatenate([uv, np.ones_like(uv[..., :1])], axis=-1)
    K_inv = np.linalg.inv(intrinsics)
    cam = pix @ K_inv.T * depth[..., None]
    cam_h = np.concatenate([cam, np.ones_like(cam[..., :1])], axis=-1)
    E_inv = np.linalg.inv(extrinsics)
    return (cam_h @ E_inv.T)[..., :3]


def _project_cv(points: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World coords -> (uv, depth) in the camera (OpenCV convention)."""
    pts_h = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    cam = pts_h @ extrinsics.T
    cam_xyz = cam[..., :3]
    depth = cam_xyz[..., 2]
    proj = cam_xyz @ intrinsics.T
    uv = proj[..., :2] / proj[..., 2:3].clip(1e-12)
    return uv.astype(np.float32), depth.astype(np.float32)


def _grid_sample_uv(img_bchw: torch.Tensor, uv: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample img_bchw at UV-in-[0,1] coords uv of shape (B, H, W, 2); replicate-border."""
    grid = uv * 2.0 - 1.0
    return F.grid_sample(img_bchw, grid, mode=mode, padding_mode="border", align_corners=False)


def split_panorama_image(image: torch.Tensor, extrinsics: np.ndarray, intrinsics: List[np.ndarray], resolution: int) -> torch.Tensor:
    """(3, Hp, Wp) equirect on any device -> (N, 3, R, R) perspective crops on the same device."""
    device = image.device
    N = len(extrinsics)
    uv = _uv_grid(resolution, resolution)
    sample_uvs = []
    for i in range(N):
        world = _unproject_cv(uv, np.ones(uv.shape[:-1], dtype=np.float32), extrinsics[i], intrinsics[i])
        sample_uvs.append(directions_to_spherical_uv(world))
    sample_uvs = np.stack(sample_uvs, axis=0)

    img_bchw = image.unsqueeze(0).expand(N, -1, -1, -1).contiguous()
    sample_uvs_t = torch.from_numpy(sample_uvs).to(device=device, dtype=image.dtype)
    return _grid_sample_uv(img_bchw, sample_uvs_t, mode="bilinear")


def _poisson_equation(W: int, H: int, wrap_x: bool = False, wrap_y: bool = False):
    """Sparse Laplacian operator over the H x W grid."""
    grid_index = np.arange(H * W).reshape(H, W)
    grid_index = np.pad(grid_index, ((0, 0), (1, 1)), mode="wrap" if wrap_x else "edge")
    grid_index = np.pad(grid_index, ((1, 1), (0, 0)), mode="wrap" if wrap_y else "edge")

    data = np.array([[-4, 1, 1, 1, 1]], dtype=np.float32).repeat(H * W, axis=0).reshape(-1)
    indices = np.stack([
        grid_index[1:-1, 1:-1],
        grid_index[:-2, 1:-1], grid_index[2:, 1:-1],
        grid_index[1:-1, :-2], grid_index[1:-1, 2:],
    ], axis=-1).reshape(-1)
    indptr = np.arange(0, H * W * 5 + 1, 5)
    return csr_array((data, indices, indptr), shape=(H * W, H * W))


def _grad_equation(W: int, H: int, wrap_x: bool = False, wrap_y: bool = False):
    """Sparse forward-difference operator over the H x W grid."""
    grid_index = np.arange(W * H).reshape(H, W)
    if wrap_x:
        grid_index = np.pad(grid_index, ((0, 0), (0, 1)), mode="wrap")
    if wrap_y:
        grid_index = np.pad(grid_index, ((0, 1), (0, 0)), mode="wrap")

    data = np.concatenate([
        np.concatenate([
            np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
            -np.ones((grid_index.shape[0], grid_index.shape[1] - 1), dtype=np.float32).reshape(-1, 1),
        ], axis=1).reshape(-1),
        np.concatenate([
            np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),
            -np.ones((grid_index.shape[0] - 1, grid_index.shape[1]), dtype=np.float32).reshape(-1, 1),
        ], axis=1).reshape(-1),
    ])
    indices = np.concatenate([
        np.concatenate([grid_index[:, :-1].reshape(-1, 1), grid_index[:, 1:].reshape(-1, 1)], axis=1).reshape(-1),
        np.concatenate([grid_index[:-1, :].reshape(-1, 1), grid_index[1:, :].reshape(-1, 1)], axis=1).reshape(-1),
    ])
    nx = grid_index.shape[0] * (grid_index.shape[1] - 1)
    ny = (grid_index.shape[0] - 1) * grid_index.shape[1]
    indptr = np.arange(0, nx * 2 + ny * 2 + 1, 2)
    return csr_array((data, indices, indptr), shape=(nx + ny, H * W))


def _scipy_remap_bilinear(img: np.ndarray, sample_pixels: np.ndarray, mode: str = "bilinear") -> np.ndarray:
    """Bilinear/nearest sampling at fractional pixel coords; out-of-range clamps to nearest border."""
    H, W = img.shape[:2]
    yy = np.clip(sample_pixels[..., 1], 0, H - 1)
    xx = np.clip(sample_pixels[..., 0], 0, W - 1)
    order = 1 if mode == "bilinear" else 0
    if img.ndim == 2:
        return map_coordinates(img, [yy, xx], order=order, mode="nearest").astype(img.dtype)
    out = np.stack([
        map_coordinates(img[..., c], [yy, xx], order=order, mode="nearest")
        for c in range(img.shape[-1])
    ], axis=-1)
    return out.astype(img.dtype)


def merge_panorama_depth(width: int, height: int,
                         distance_maps: List[np.ndarray], pred_masks: List[np.ndarray],
                         extrinsics: List[np.ndarray], intrinsics: List[np.ndarray],
                         on_view: Optional[Callable[[], None]] = None,
                         on_solve_start: Optional[Callable[[int, int], None]] = None,
                         on_solve_end: Optional[Callable[[int, int], None]] = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch per-view distance maps into a single equirect distance map.

    Recursive multi-scale solve: solves at half resolution first and uses that as the lsmr init
    for the full-resolution solve. Optional callbacks fire per view processed and around each
    lsmr solve so callers can drive a progress bar.
    """

    if max(width, height) > 256:
        coarse_depth, _ = merge_panorama_depth(width // 2, height // 2,
                                               distance_maps, pred_masks, extrinsics, intrinsics,
                                               on_view=on_view,
                                               on_solve_start=on_solve_start,
                                               on_solve_end=on_solve_end)
        t = torch.from_numpy(coarse_depth).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
        depth_init = t.squeeze().numpy().astype(np.float32)
    else:
        depth_init = None

    spherical_directions = spherical_uv_to_directions(_uv_grid(height, width))

    pano_log_grad_maps, pano_grad_masks = [], []
    pano_log_lap_maps, pano_lap_masks = [], []
    pano_pred_masks: List[np.ndarray] = []

    for i in range(len(distance_maps)):
        proj_uv, proj_depth = _project_cv(spherical_directions, extrinsics[i], intrinsics[i])
        proj_valid = (proj_depth > 0) & (proj_uv > 0).all(axis=-1) & (proj_uv < 1).all(axis=-1)

        Hd, Wd = distance_maps[i].shape[:2]
        proj_pixels = np.clip(proj_uv, 0, 1) * np.array([Wd - 1, Hd - 1], dtype=np.float32)

        log_dist = np.log(np.clip(distance_maps[i], 1e-6, None))
        sampled = _scipy_remap_bilinear(log_dist, proj_pixels, mode="bilinear")
        pano_log = np.where(proj_valid, sampled, 0.0).astype(np.float32)

        sampled_mask = _scipy_remap_bilinear(pred_masks[i].astype(np.uint8), proj_pixels, mode="nearest")
        pano_pred = proj_valid & (sampled_mask > 0)

        # Equirect wraps horizontally but not vertically: wrap pad along x, edge pad along y.
        padded = np.pad(pano_log, ((0, 0), (0, 1)), mode="wrap")
        gx, gy = padded[:, :-1] - padded[:, 1:], padded[:-1, :] - padded[1:, :]
        padded_m = np.pad(pano_pred, ((0, 0), (0, 1)), mode="wrap")
        mx, my = padded_m[:, :-1] & padded_m[:, 1:], padded_m[:-1, :] & padded_m[1:, :]
        pano_log_grad_maps.append((gx, gy))
        pano_grad_masks.append((mx, my))

        padded = np.pad(pano_log, ((1, 1), (0, 0)), mode="edge")
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        lap = convolve(padded, lap_kernel)[1:-1, 1:-1]
        padded_m = np.pad(pano_pred, ((1, 1), (0, 0)), mode="edge")
        padded_m = np.pad(padded_m, ((0, 0), (1, 1)), mode="wrap")
        m_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        lap_mask = convolve(padded_m.astype(np.uint8), m_kernel)[1:-1, 1:-1] == 5
        pano_log_lap_maps.append(lap)
        pano_lap_masks.append(lap_mask)
        pano_pred_masks.append(pano_pred)

        if on_view is not None:
            on_view()

    gx = np.stack([m[0] for m in pano_log_grad_maps], axis=0)
    gy = np.stack([m[1] for m in pano_log_grad_maps], axis=0)
    mx = np.stack([m[0] for m in pano_grad_masks], axis=0)
    my = np.stack([m[1] for m in pano_grad_masks], axis=0)
    gx_avg = (gx * mx).sum(axis=0) / mx.sum(axis=0).clip(1e-3)
    gy_avg = (gy * my).sum(axis=0) / my.sum(axis=0).clip(1e-3)

    laps = np.stack(pano_log_lap_maps, axis=0)
    lap_masks = np.stack(pano_lap_masks, axis=0)
    lap_avg = (laps * lap_masks).sum(axis=0) / lap_masks.sum(axis=0).clip(1e-3)

    grad_x_mask = mx.any(axis=0).reshape(-1)
    grad_y_mask = my.any(axis=0).reshape(-1)
    grad_mask = np.concatenate([grad_x_mask, grad_y_mask])
    lap_mask_flat = lap_masks.any(axis=0).reshape(-1)

    A = vstack([
        _grad_equation(width, height, wrap_x=True, wrap_y=False)[grad_mask],
        _poisson_equation(width, height, wrap_x=True, wrap_y=False)[lap_mask_flat],
    ])
    b = np.concatenate([
        gx_avg.reshape(-1)[grad_x_mask],
        gy_avg.reshape(-1)[grad_y_mask],
        lap_avg.reshape(-1)[lap_mask_flat],
    ])
    x0 = np.log(np.clip(depth_init, 1e-6, None)).reshape(-1) if depth_init is not None else None

    if on_solve_start is not None:
        on_solve_start(width, height)
    x, *_ = lsmr(A, b, atol=1e-5, btol=1e-5, x0=x0, show=False)
    if on_solve_end is not None:
        on_solve_end(width, height)

    pano_depth = np.exp(x).reshape(height, width).astype(np.float32)
    pano_mask = np.any(pano_pred_masks, axis=0)
    return pano_depth, pano_mask
