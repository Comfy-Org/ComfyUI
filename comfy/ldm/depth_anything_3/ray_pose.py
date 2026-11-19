"""Ray-to-pose conversion for the multi-view path of Depth Anything 3."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# qr/svd use fp32: CUDA often has no fp16/bf16 kernels for these ops.


def _ql_decomposition(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose A = Q @ L with Q orthogonal and L lower-triangular.
    Implemented in terms of QR by reversing the columns/rows; the standard
    trick from the upstream reference. Inputs A are (3, 3)."""
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device, dtype=A.dtype)
    A_tilde = A @ P
    # CUDA QR is not implemented for fp16/bf16; upcast just for this call.
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde.float())
    Q_tilde = Q_tilde.to(A.dtype)
    R_tilde = R_tilde.to(A.dtype)
    Q = Q_tilde @ P
    L = P @ R_tilde @ P
    d = torch.diag(L)
    sign = torch.sign(d)
    Q = Q * sign[None, :]  # scale columns of Q
    L = L * sign[:, None]  # scale rows of L
    return Q, L


def _homogenize_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


# -----------------------------------------------------------------------------
# Weighted-LSQ + RANSAC homography (batched)
# -----------------------------------------------------------------------------


def _find_homography_weighted_lsq(src_pts: torch.Tensor, dst_pts: torch.Tensor, confident_weight: torch.Tensor,) -> torch.Tensor:
    """Solve a single H with weighted least-squares (DLT)."""
    N = src_pts.shape[0]
    if N < 4:
        raise ValueError("At least 4 points are required to compute a homography.")
    w = confident_weight.sqrt().unsqueeze(1)  # (N, 1)
    x = src_pts[:, 0:1]
    y = src_pts[:, 1:2]
    u = dst_pts[:, 0:1]
    v = dst_pts[:, 1:2]
    zeros = torch.zeros_like(x)
    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=1)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=1)
    A = torch.cat([A1, A2], dim=0)        # (2N, 9)
    # CUDA SVD is not implemented for fp16/bf16; upcast just for this call.
    _, _, Vh = torch.linalg.svd(A.float())
    Vh = Vh.to(A.dtype)
    H = Vh[-1].reshape(3, 3)
    return H / H[-1, -1]


def _find_homography_weighted_lsq_batched(src_pts_batch: torch.Tensor, dst_pts_batch: torch.Tensor, confident_weight_batch: torch.Tensor) -> torch.Tensor:
    """Batched DLT solver. Inputs (B, K, 2) / (B, K); output (B, 3, 3)."""
    B, K, _ = src_pts_batch.shape
    w = confident_weight_batch.sqrt().unsqueeze(2)
    x = src_pts_batch[:, :, 0:1]
    y = src_pts_batch[:, :, 1:2]
    u = dst_pts_batch[:, :, 0:1]
    v = dst_pts_batch[:, :, 1:2]
    zeros = torch.zeros_like(x)
    A1 = torch.cat([-x * w, -y * w, -w, zeros, zeros, zeros, x * u * w, y * u * w, u * w], dim=2)
    A2 = torch.cat([zeros, zeros, zeros, -x * w, -y * w, -w, x * v * w, y * v * w, v * w], dim=2)
    A = torch.cat([A1, A2], dim=1)        # (B, 2K, 9)
    # CUDA SVD is not implemented for fp16/bf16; upcast just for this call.
    _, _, Vh = torch.linalg.svd(A.float())
    Vh = Vh.to(A.dtype)
    H = Vh[:, -1].reshape(B, 3, 3)
    return H / H[:, 2:3, 2:3]


def _ransac_find_homography_weighted_batched(
    src_pts: torch.Tensor,                # (B, N, 2)
    dst_pts: torch.Tensor,                # (B, N, 2)
    confident_weight: torch.Tensor,       # (B, N)
    n_sample: int,
    n_iter: int = 100,
    reproj_threshold: float = 3.0,
    num_sample_for_ransac: int = 8,
    max_inlier_num: int = 10000,
    rand_sample_iters_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched weighted-RANSAC homography estimator. Returns (B, 3, 3) homography matrices."""
    B, N, _ = src_pts.shape
    assert N >= 4
    device = src_pts.device

    sorted_idx = torch.argsort(confident_weight, descending=True, dim=1)
    candidate_idx = sorted_idx[:, :n_sample]                  # (B, n_sample)

    if rand_sample_iters_idx is None:
        rand_sample_iters_idx = torch.stack(
            [torch.randperm(n_sample, device=device)[:num_sample_for_ransac]
             for _ in range(n_iter)],
            dim=0,
        )

    rand_idx = candidate_idx[:, rand_sample_iters_idx]        # (B, n_iter, k)
    b_idx = (
        torch.arange(B, device=device)
        .view(B, 1, 1)
        .expand(B, n_iter, num_sample_for_ransac)
    )
    src_b = src_pts[b_idx, rand_idx]
    dst_b = dst_pts[b_idx, rand_idx]
    w_b = confident_weight[b_idx, rand_idx]

    cB, cN = src_b.shape[:2]
    H_batch = _find_homography_weighted_lsq_batched(
        src_b.flatten(0, 1), dst_b.flatten(0, 1), w_b.flatten(0, 1),
    ).unflatten(0, (cB, cN))                                  # (B, n_iter, 3, 3)

    src_homo = torch.cat([src_pts, torch.ones(B, N, 1, device=device, dtype=src_pts.dtype)], dim=2)
    proj = torch.bmm(
        src_homo.unsqueeze(1).expand(B, n_iter, N, 3).reshape(-1, N, 3),
        H_batch.reshape(-1, 3, 3).transpose(1, 2),
    )                                                          # (B*n_iter, N, 3)
    proj_xy = (proj[:, :, :2] / proj[:, :, 2:3]).reshape(B, n_iter, N, 2)
    err = ((proj_xy - dst_pts.unsqueeze(1)) ** 2).sum(-1).sqrt()  # (B, n_iter, N)
    inlier_mask = err < reproj_threshold
    score = (inlier_mask * confident_weight.unsqueeze(1)).sum(dim=2)
    best_idx = torch.argmax(score, dim=1)
    best_inlier_mask = inlier_mask[torch.arange(B, device=device), best_idx]

    # Refit with the inlier set (per-batch, since the inlier counts vary).
    H_inlier_list = []
    for b in range(B):
        mask = best_inlier_mask[b]
        in_src = src_pts[b][mask]
        in_dst = dst_pts[b][mask]
        in_w = confident_weight[b][mask]
        if in_src.shape[0] < 4:
            # Fall back to identity when RANSAC fails to find enough inliers.
            H_inlier_list.append(torch.eye(3, device=device, dtype=src_pts.dtype))
            continue
        sorted_w = torch.argsort(in_w, descending=True)
        if len(sorted_w) > max_inlier_num:
            keep = max(int(len(sorted_w) * 0.95), max_inlier_num)
            sorted_w = sorted_w[:keep][torch.randperm(keep, device=device)[:max_inlier_num]]
        H_inlier_list.append(
            _find_homography_weighted_lsq(in_src[sorted_w], in_dst[sorted_w], in_w[sorted_w])
        )
    return torch.stack(H_inlier_list, dim=0)


# -----------------------------------------------------------------------------
# Camera-ray utilities
# -----------------------------------------------------------------------------


def _unproject_identity(num_y: int, num_x: int, B: int, S: int, device, dtype) -> torch.Tensor:
    """Camera-space unit rays for an identity intrinsic on a 2x2 image plane."""
    dx = 1.0 / num_x
    dy = 1.0 / num_y
    # Centered camera-space coords directly (skip the K^-1 step since it's
    # just a translation by -1 on x and y when K is identity-with-center=1).
    y = torch.linspace(-(1 - dy), (1 - dy), num_y, device=device, dtype=dtype)
    x = torch.linspace(-(1 - dx), (1 - dx), num_x, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)            # (h, w, 2)
    grid = grid.unsqueeze(0).unsqueeze(0).expand(B, S, num_y, num_x, 2)
    return torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)


def _camray_to_caminfo(
    camray: torch.Tensor,  # (B, S, h, w, 6)
    confidence: Optional[torch.Tensor] = None,  # (B, S, h, w)
    reproj_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-pixel camera rays to per-view (R, T, focal, principal)."""
    if confidence is None:
        confidence = torch.ones_like(camray[..., 0])
    B, S, h, w, _ = camray.shape
    device = camray.device
    dtype = camray.dtype

    rays_target = camray[..., :3]                           # (B, S, h, w, 3)
    rays_origin = _unproject_identity(h, w, B, S, device, dtype)

    # Flatten (B*S, h*w, *) for the RANSAC routine.
    rays_target = rays_target.flatten(0, 1).flatten(1, 2)
    rays_origin = rays_origin.flatten(0, 1).flatten(1, 2)
    weights = confidence.flatten(0, 1).flatten(1, 2).clone()

    # Project to 2D in homogeneous form (the upstream calls this "perspective division").
    z_thresh = 1e-4
    mask = (rays_target[:, :, 2].abs() > z_thresh) & (rays_origin[:, :, 2].abs() > z_thresh)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    src = rays_origin.clone()
    dst = rays_target.clone()
    src[..., 0] = torch.where(mask, src[..., 0] / src[..., 2], src[..., 0])
    src[..., 1] = torch.where(mask, src[..., 1] / src[..., 2], src[..., 1])
    dst[..., 0] = torch.where(mask, dst[..., 0] / dst[..., 2], dst[..., 0])
    dst[..., 1] = torch.where(mask, dst[..., 1] / dst[..., 2], dst[..., 1])
    src = src[..., :2]
    dst = dst[..., :2]

    N = src.shape[1]
    n_iter = 100
    sample_ratio = 0.3
    num_sample_for_ransac = 8
    n_sample = max(num_sample_for_ransac, int(N * sample_ratio))
    rand_idx = torch.stack(
        [torch.randperm(n_sample, device=device)[:num_sample_for_ransac] for _ in range(n_iter)],
        dim=0,
    )

    # Chunk along the view axis to keep peak memory predictable.
    chunk = 2
    A_list = []
    for i in range(0, src.shape[0], chunk):
        A = _ransac_find_homography_weighted_batched(
            src[i:i + chunk], dst[i:i + chunk], weights[i:i + chunk],
            n_sample=n_sample, n_iter=n_iter,
            num_sample_for_ransac=num_sample_for_ransac,
            reproj_threshold=reproj_threshold,
            rand_sample_iters_idx=rand_idx,
            max_inlier_num=8000,
        )
        # Flip sign on dets that come out < 0 (so that the QL produces a
        # right-handed rotation). ``det`` lacks fp16/bf16 CUDA kernels, so
        # do the comparison in fp32.
        flip = torch.linalg.det(A.float()) < 0
        A = torch.where(flip[:, None, None], -A, A)
        A_list.append(A)
    A = torch.cat(A_list, dim=0)                            # (B*S, 3, 3)

    R_list, f_list, pp_list = [], [], []
    for i in range(A.shape[0]):
        R, L = _ql_decomposition(A[i])
        L = L / L[2][2]
        f_list.append(torch.stack((L[0][0], L[1][1])))
        pp_list.append(torch.stack((L[2][0], L[2][1])))
        R_list.append(R)
    R = torch.stack(R_list).reshape(B, S, 3, 3)
    focal = torch.stack(f_list).reshape(B, S, 2)
    pp = torch.stack(pp_list).reshape(B, S, 2)

    # Translation: confidence-weighted average of camray direction(s).
    cf = confidence.flatten(0, 1).flatten(1, 2)
    T = (camray.flatten(0, 1).flatten(1, 2)[..., 3:] * cf.unsqueeze(-1)).sum(dim=1)
    T = T / cf.sum(dim=-1, keepdim=True)
    T = T.reshape(B, S, 3)

    # Match upstream output convention: focal -> 1/focal, pp + 1.
    return R, T, 1.0 / focal, pp + 1.0


def get_extrinsic_from_camray(
    camray: torch.Tensor,  # (B, S, h, w, 6)
    conf: torch.Tensor,  # (B, S, h, w, 1) or (B, S, h, w)
    patch_size_y: int,
    patch_size_x: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrap a 4x4 extrinsic + per-view focal + principal-point output."""
    if conf.ndim == 5 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
    R, T, focal, pp = _camray_to_caminfo(camray, confidence=conf)
    extr = torch.cat([R, T.unsqueeze(-1)], dim=-1)           # (B, S, 3, 4)
    homo_row = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
    homo_row = homo_row.view(1, 1, 1, 4).expand(R.shape[0], R.shape[1], 1, 4)
    extr = torch.cat([extr, homo_row], dim=-2)               # (B, S, 4, 4)
    return extr, focal, pp
