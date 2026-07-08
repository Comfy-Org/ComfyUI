"""Geometry / camera transform helpers for Depth Anything 3."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Affine 4x4 helpers
# -----------------------------------------------------------------------------


def as_homogeneous(ext: torch.Tensor) -> torch.Tensor:
    """Promote (...,3,4) extrinsics to (...,4,4) homogeneous form. No-op when the input is already ``(...,4,4)``."""
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        ones = torch.zeros_like(ext[..., :1, :4])
        ones[..., 0, 3] = 1.0
        return torch.cat([ext, ones], dim=-2)
    raise ValueError(f"Invalid affine shape: {ext.shape}")


def affine_inverse(A: torch.Tensor) -> torch.Tensor:
    """Inverse of an affine matrix ``[R|T; 0 0 0 1]``."""
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


# -----------------------------------------------------------------------------
# Quaternion <-> rotation matrix (xyzw / scalar-last)
# -----------------------------------------------------------------------------


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """sqrt(max(0, x)) with a zero subgradient where x == 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Force the real part of a unit quaternion (xyzw) to be non-negative."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (xyzw) to (...,3,3) rotation matrices."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (...,3,3) rotation matrices to quaternions (xyzw)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    # Reorder rijk -> xyzw (i.e. ijkr).
    out = out[..., [1, 2, 3, 0]]
    return standardize_quaternion(out)


# -----------------------------------------------------------------------------
# Pose-encoding <-> extrinsics + intrinsics
# -----------------------------------------------------------------------------


def extri_intri_to_pose_encoding(extrinsics: torch.Tensor, intrinsics: torch.Tensor, image_size_hw: Tuple[int, int]) -> torch.Tensor:
    """Pack (extr, intr, image_size) into the 9-D pose-encoding vector.
    extrinsics: camera-to-world (c2w) (B,S,4,4) matrices,
    intrinsics: pixel-space (B,S,3,3) matrices,
    image_size_hw: is a (H, W) pair.
    """
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    return torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def pose_encoding_to_extri_intri(pose_encoding: torch.Tensor, image_size_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of extri_intri_to_pose_encoding."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]
    # Normalize to unit quaternion. CameraDec outputs raw values; a near-zero
    # quaternion causes two_s = 2/norm² → inf in quat_to_mat → NaN extrinsics.
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)
    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device, dtype=pose_encoding.dtype)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics
