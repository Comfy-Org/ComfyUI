# The bbox/affine math (xyxy<->cs, get_warp_matrices) is the standard
# top-down pose-estimation crop pipeline from MMPose (Apache 2.0):
# https://github.com/open-mmlab/mmpose — same algorithm as UDP (CVPR 2020).

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# Bbox + affine math
# All `output_size` / image-shape tuples in this block are (H, W) to match
# the torch.Size convention used everywhere else in the codebase.

def bbox_xyxy2cs(bbox, padding: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """xyxy bbox -> (center, scale) with optional padding multiplier."""
    bbox = torch.as_tensor(bbox, dtype=torch.float32)
    dim = bbox.dim()
    if dim == 1:
        bbox = bbox.unsqueeze(0)
    x1, y1, x2, y2 = bbox[:, 0:1], bbox[:, 1:2], bbox[:, 2:3], bbox[:, 3:4]
    center = torch.cat([x1 + x2, y1 + y2], dim=1) * 0.5
    scale = torch.cat([x2 - x1, y2 - y1], dim=1) * padding
    if dim == 1:
        return center[0], scale[0]
    return center, scale


def fix_aspect_ratio(bbox_scale, aspect_ratio: float) -> torch.Tensor:
    """Pad whichever side is too narrow to hit `aspect_ratio` (w/h)."""
    bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)
    dim = bbox_scale.dim()
    if dim == 1:
        bbox_scale = bbox_scale.unsqueeze(0)
    w, h = bbox_scale[:, 0:1], bbox_scale[:, 1:2]
    out = torch.where(
        w > h * aspect_ratio,
        torch.cat([w, w / aspect_ratio], dim=1),
        torch.cat([h * aspect_ratio, h], dim=1),
    )
    return out[0] if dim == 1 else out


def get_warp_matrices(centers, scales, output_size: Tuple[int, int]) -> torch.Tensor:
    """Batched 2x3 affine matrices mapping each (center, scale) bbox region to
    the output box. `output_size` is (H_out, W_out). With rot=0 the MMPose
    3-point fit reduces to a closed-form isotropic scale + translate.
    """
    centers = torch.as_tensor(centers, dtype=torch.float32)
    scales = torch.as_tensor(scales, dtype=torch.float32)
    if centers.dim() == 1:
        centers = centers.unsqueeze(0)
        scales = scales.unsqueeze(0)
    n = centers.shape[0]
    src_w = scales[:, 0]
    dst_h = float(output_size[0])
    dst_w = float(output_size[1])
    # With rot=0 the warp is just scale + translate (uniform x/y scale based
    # on src_w/dst_w). The closed form drops out of MMPose's 3-point solve.
    s = dst_w / src_w  # (N,)
    mats = torch.zeros((n, 2, 3), dtype=centers.dtype, device=centers.device)
    mats[:, 0, 0] = s
    mats[:, 1, 1] = s
    mats[:, 0, 2] = dst_w * 0.5 - s * centers[:, 0]
    mats[:, 1, 2] = dst_h * 0.5 - s * centers[:, 1]
    return mats # (N, 2, 3)


def warp_affine_batched(
    src_t: torch.Tensor, # (N, C, H_src, W_src) float
    mats: torch.Tensor, # (N, 2, 3) float
    output_size: Tuple[int, int] # (H_out, W_out)
    ) -> torch.Tensor:
    """Apply N forward (src->dst) 2x3 affine warps to N source images in one
    grid_sample call. Kept generic over arbitrary affines (not specialized to
    the scale+translate produced by `get_warp_matrices`) so callers can pass
    rotated/sheared affines; the per-crop 3x3 invert is O(N) of trivial work."""

    H_out, W_out = int(output_size[0]), int(output_size[1])
    N, _, H_src, W_src = src_t.shape
    device = src_t.device

    # Invert each forward affine; grid_sample needs dst->src.
    mats_t = mats.to(device=device, dtype=torch.float32)
    bottom = mats_t.new_tensor([0.0, 0.0, 1.0]).expand(N, 1, 3)
    mats_3 = torch.cat([mats_t, bottom], dim=1)  # (N, 3, 3)
    mats_inv = torch.linalg.inv(mats_3)[:, :2, :]  # (N, 2, 3)

    # Output pixel-center grid (i+0.5, j+0.5).
    ys, xs = torch.meshgrid(
        torch.arange(H_out, dtype=torch.float32, device=device) + 0.5,
        torch.arange(W_out, dtype=torch.float32, device=device) + 0.5,
        indexing="ij",
    )
    homo = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # (H_out, W_out, 3)
    src_pos = torch.einsum("nkl,ijl->nijk", mats_inv, homo)  # (N, H_out, W_out, 2)
    # Normalize to [-1, 1] grid_sample coords (align_corners=False).
    src_pos[..., 0] = src_pos[..., 0] / W_src * 2 - 1
    src_pos[..., 1] = src_pos[..., 1] / H_src * 2 - 1

    return F.grid_sample(src_t, src_pos, mode="bilinear", padding_mode="zeros", align_corners=False)


# Batch construction (one prediction over N person crops from a single image)

def prepare_batch(
    img,                            # (H, W, 3) uint8 torch tensor or list of such tensors
    boxes,                          # (N, 4) xyxy (numpy or torch)
    input_size: Tuple[int, int],    # (W, H) of the model crop
    bbox_padding: float = 1.25,     # xyxy->cs padding multiplier (1.25 body, 0.9 hand)
    aspect_ratio: float = 0.75,     # w/h of the crop (0.75 matches HMR2/Sapiens)
    masks=None,                     # optional per-person masks
    masks_score=None,               # optional per-person mask scores
    cam_int=None,                   # optional camera intrinsics
) -> Dict:
    """Build the batch dict the SAM3DBody forward expects, doing the N crops in one batched `grid_sample` call."""

    is_multi_image = isinstance(img, list)
    if is_multi_image:
        assert len(img) == boxes.shape[0]
        height, width = img[0].shape[:2]
    else:
        height, width = img.shape[:2]

    n = int(boxes.shape[0])
    assert n > 0, "prepare_batch needs at least one box"

    W_out, H_out = int(input_size[0]), int(input_size[1])

    # Per-box bbox math (cheap, vectorized, CPU).
    centers, scales = bbox_xyxy2cs(boxes, padding=bbox_padding)
    # Two passes: first hits the upstream bbox aspect (e.g. 0.75 HMR2/Sapiens
    # convention), second pads further if the model crop's W_out/H_out differs
    # from that. When they match (common case) the second call is a no-op.
    scales = fix_aspect_ratio(scales, aspect_ratio)
    scales = fix_aspect_ratio(scales, W_out / H_out)
    mats = get_warp_matrices(centers, scales, (H_out, W_out))  # (N, 2, 3)

    # Stack source images into a contiguous (N, 3, H, W) tensor on CPU.
    if is_multi_image:
        src_t = torch.stack(list(img), dim=0)
    else:
        src_t = img.unsqueeze(0).expand(n, -1, -1, -1)
    src_t = src_t.permute(0, 3, 1, 2).contiguous().float()  # (N, 3, H, W) in [0, 255]

    warped_t = warp_affine_batched(src_t, mats, (H_out, W_out))  # (N, 3, H_out, W_out)
    # Float warp -> floor (matches the legacy uint8 round-trip) -> /255.
    img_t = torch.floor(warped_t).clamp_(0.0, 255.0) / 255.0  # (N, 3, H_out, W_out) in [0, 1]

    # Masks: zero-init when missing, otherwise stack and warp through the same matrices.
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    if masks is None:
        mask_t = torch.zeros((n, H_out, W_out), dtype=torch.float32)
        mask_score_t = torch.zeros((n,), dtype=torch.float32)
    else:
        # masks is an array of N items, each (H, W) or (H, W, 1).
        masks_t = torch.stack([torch.as_tensor(masks[i]) for i in range(n)], dim=0)
        if masks_t.dim() == 4 and masks_t.shape[-1] == 1:
            masks_t = masks_t[..., 0]
        masks_src_t = masks_t.float().unsqueeze(1)  # (N, 1, H, W) in [0, 255]
        warped_masks = warp_affine_batched(masks_src_t, mats, (H_out, W_out))
        mask_t = torch.floor(warped_masks.squeeze(1)).clamp_(0.0, 255.0)
        if masks_score is not None:
            mask_score_t = torch.as_tensor([masks_score[i] for i in range(n)], dtype=torch.float32)
        else:
            mask_score_t = torch.ones((n,), dtype=torch.float32)

    img_size_t = torch.tensor([W_out, H_out], dtype=torch.float32).expand(n, 2).contiguous()

    batch = {
        "img": img_t.unsqueeze(0),                  # (1, N, 3, H_out, W_out)
        "img_size": img_size_t.unsqueeze(0),        # (1, N, 2)
        "bbox_center": centers.unsqueeze(0),        # (1, N, 2)
        "bbox_scale": scales.unsqueeze(0),          # (1, N, 2)
        "bbox": boxes_t.unsqueeze(0),               # (1, N, 4)
        "affine_trans": mats.unsqueeze(0),          # (1, N, 2, 3)
        "mask": mask_t.unsqueeze(0).unsqueeze(2),   # (1, N, 1, H_out, W_out)
        "mask_score": mask_score_t.unsqueeze(0),    # (1, N)
        "person_valid": torch.ones((1, n), dtype=torch.float32),
    }

    if cam_int is not None:
        batch["cam_int"] = cam_int.to(batch["img"])
    else:
        # Default intrinsics: focal = sqrt(W^2 + H^2), principal point = image center.
        f = (height ** 2 + width ** 2) ** 0.5
        batch["cam_int"] = torch.tensor(
            [[[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]]],
        ).to(batch["img"])

    return batch


# Geometry utils

def rot6d_to_rotmat(
        x: torch.Tensor     # (B, 6) batch of 6-D rotation representations.
    ) -> torch.Tensor:      # (B, 3, 3) rotation matrices.
    """6D continuous rotation rep (Zhou et al., CVPR 2019) -> 3x3 rotation matrix."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def perspective_projection(
        x: torch.Tensor,    # (B, N, 3) 3D points in camera coords.
        K: torch.Tensor     # (B, 3, 3) camera intrinsics.
    ) -> torch.Tensor:      # (B, N, 2) 2D image-plane projections.
    """Project 3D points (already in camera frame) through intrinsics K."""
    y = x / x[:, :, -1].unsqueeze(-1)            # perspective divide
    y = torch.einsum("bij,bkj->bki", K, y)       # apply intrinsics
    return y[:, :, :2]


# Rotation conversions, behavior mirrors the roma library (https://github.com/naver/roma)

def _axis_rotmat(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices around a single coordinate axis. Shape (..., 3, 3)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero,
                zero, cos, -sin,
                zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin,
                zero, one, zero,
                -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero,
                sin, cos, zero,
                zero, zero, one)
    else:
        raise ValueError(f"Invalid axis {axis!r}; expected X/Y/Z.")
    return torch.stack(flat, dim=-1).reshape(angle.shape + (3, 3))


def euler_to_rotmat(convention: str, angles: torch.Tensor) -> torch.Tensor:
    """Euler angles -> rotation matrix, matching roma's case-keyed convention."""
    axes = convention.upper()
    R0 = _axis_rotmat(axes[0], angles[..., 0])
    R1 = _axis_rotmat(axes[1], angles[..., 1])
    R2 = _axis_rotmat(axes[2], angles[..., 2])
    if convention.islower():
        return R2 @ R1 @ R0
    return R0 @ R1 @ R2


def _index_from_letter(letter: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[letter]


def _angle_from_tan(
    axis: str,
    other_axis: str,
    data: torch.Tensor,
    horizontal: bool,
    tait_bryan: bool,
) -> torch.Tensor:
    """Extract an outer Euler angle from a row/column of a rotation matrix.

    Adapted from PyTorch3D's matrix_to_euler_angles helper.
    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ("XY", "YZ", "ZX")
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _matrix_to_euler_intrinsic(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """Decompose a rotation matrix into intrinsic Euler angles (uppercase abc).

    Adapted from PyTorch3D's matrix_to_euler_angles.
    """
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        sign = -1.0 if (i0 - i2) in (-1, 2) else 1.0
        central = torch.asin(matrix[..., i0, i2] * sign)
    else:
        central = torch.acos(matrix[..., i0, i0])

    out = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(out, dim=-1)


def rotmat_to_euler(convention: str, matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> Euler angles, inverse of :func:`euler_to_rotmat`.

    PyTorch3D's matrix_to_euler_angles uses the convention R = R_a R_b R_c for
    convention "abc"; that matches roma's UPPERCASE ordering directly. For
    roma's lowercase, the matrix is reversed (R_c R_b R_a), so we decompose
    with the reversed convention and flip the angles back to axis order.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) rotation matrix, got {tuple(matrix.shape)}.")
    if convention.isupper():
        return _matrix_to_euler_intrinsic(matrix, convention)
    decomposed = _matrix_to_euler_intrinsic(matrix, convention.upper()[::-1])
    return decomposed.flip(-1)


def unitquat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (x, y, z, w) -> rotation matrix.

    Matches roma.unitquat_to_rotmat (scalar-last). The quaternion is assumed to be normalized.

    Args:
        quat: (..., 4) unit quaternion.
    Returns:
        (..., 3, 3) rotation matrix.
    """
    x, y, z, w = quat.unbind(dim=-1)
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    one = torch.ones_like(w)
    flat = (
        one - (tyy + tzz),  txy - twz,           txz + twy,
        txy + twz,          one - (txx + tzz),   tyz - twx,
        txz - twy,          tyz + twx,           one - (txx + tyy),
    )
    return torch.stack(flat, dim=-1).reshape(quat.shape[:-1] + (3, 3))
