"""BVH export for SAM 3D Body pose_data.

BVH stores explicit bone OFFSETs per joint, so standard importers reconstruct
anatomical bone orientations directly (unlike glTF). We skip the rig's joint 0
(static world anchor) and use joint 1 as the ROOT (6 channels: XYZ pos + ZXY
rot); other joints get 3 channels. Rotations are intrinsic Z-X-Y Euler degrees.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np

from .glb_shared import (
    Rig,
    bone_locals_from_globals,
    collect_tracks,
    global_skel_state_from_pose_data,
    quat_sign_fix_per_joint,
    unflip,
)


def _quat_to_zxy_euler_deg(quat: np.ndarray) -> np.ndarray:
    """xyzw quat → intrinsic Z-X-Y Euler degrees, returned as (..., 3) in
    (z, x, y) order to match BVH's `CHANNELS Zrotation Xrotation Yrotation`."""
    q = np.asarray(quat, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # ZXY decomposition R = Rz(c)·Rx(a)·Ry(b):
    #   M[2][1] = 2(yz + xw)        → sin(a)
    #   M[0][1] = 2(xy - zw)        → -cos(a) sin(c)
    #   M[1][1] = 1 - 2(x² + z²)    → cos(a) cos(c)
    #   M[2][0] = 2(xz - yw)        → -cos(a) sin(b)
    #   M[2][2] = 1 - 2(x² + y²)    → cos(a) cos(b)
    M21 = np.clip(2.0 * (y * z + x * w), -1.0, 1.0)
    M01 = 2.0 * (x * y - z * w)
    M11 = 1.0 - 2.0 * (x * x + z * z)
    M20 = 2.0 * (x * z - y * w)
    M22 = 1.0 - 2.0 * (x * x + y * y)
    a = np.arcsin(M21)
    c = np.arctan2(-M01, M11)
    b = np.arctan2(-M20, M22)
    out = np.stack([np.rad2deg(c), np.rad2deg(a), np.rad2deg(b)], axis=-1)
    return out.astype(np.float32)


def _find_bvh_root(parents: np.ndarray, is_external: bool = False) -> int:
    """First child of the rig's world anchor, dropping the origin→body stick.
    Falls back to the first root joint. External rigs whose root is already the
    articulated body root with multiple child chains keep the root — descending
    into one child would drop the sibling limbs."""
    NJ = parents.shape[0]
    world_anchors = [j for j in range(NJ)
                     if not (0 <= int(parents[j]) < NJ and int(parents[j]) != j)]
    if not world_anchors:
        return 0
    children: List[List[int]] = [[] for _ in range(NJ)]
    for j in range(NJ):
        p = int(parents[j])
        if 0 <= p < NJ and p != j:
            children[p].append(j)
    wa = world_anchors[0]
    if children[wa]:
        if is_external and len(children[wa]) > 1:
            return wa
        return children[wa][0]
    return wa


def _build_children_map(parents: np.ndarray) -> List[List[int]]:
    NJ = parents.shape[0]
    out: List[List[int]] = [[] for _ in range(NJ)]
    for j in range(NJ):
        p = int(parents[j])
        if 0 <= p < NJ and p != j:
            out[p].append(j)
    return out


def build_bvh(
    pose_data: Dict[str, Any],
    model: Any = None,
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    units: str = "cm",
) -> bytes:
    """Build a BVH file from pose_data. Returns UTF-8 text bytes.

    `model` may be None when pose_data carries a `_skeleton_override` (external
    rigs); the rig hierarchy/offsets/bind come from the override. `units` is
    "cm" (default) or "m" — affects OFFSET/root-position, not rotations.
    """
    if units not in ("cm", "m"):
        raise ValueError(f"build_bvh: units must be 'cm' or 'm', got {units!r}")
    unit_scale = 100.0 if units == "cm" else 1.0

    rig = Rig.from_pose_data(pose_data, model)
    is_external = not rig.can_rerun_fk
    NJ = rig.num_joints
    parents = rig.parents
    frames = pose_data["frames"]

    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_bvh: no valid tracks in pose_data")
    person_k, frame_indices = tracks[0]
    n_frames = len(frame_indices)
    if n_frames == 0:
        raise ValueError("build_bvh: track has zero frames")

    body_root = _find_bvh_root(parents, is_external)
    children_map = _build_children_map(parents)

    # Bone OFFSETs = translation_offsets (joint position relative to parent).
    # The BVH root uses its bind world position so the skeleton imports in place.
    bind_global = rig.bind_global_cm                         # (NJ, 8) cm
    bind_pos_m = bind_global[:, :3].astype(np.float64) * 0.01  # (NJ, 3) m
    offset_m = rig.joint_offsets_cm.astype(np.float64) * 0.01

    # DFS order rooted at body_root — matches per-frame channel order.
    bvh_order: List[int] = []
    def _visit(j: int) -> None:
        bvh_order.append(j)
        for c in children_map[j]:
            _visit(c)
    _visit(body_root)

    # Stored pred_global_rots/pred_joint_coords (authoritative); derive locals
    # with body_root as the BVH-space hierarchy root.
    rig_global_m = global_skel_state_from_pose_data(
        pose_data, frame_indices, person_k, NJ,
        joint_coords_y_down=rig.per_frame_y_down,
    )
    rig_global_m[..., 3:7] = quat_sign_fix_per_joint(rig_global_m[..., 3:7])
    bvh_parents = parents.copy()
    bvh_parents[body_root] = -1
    bone_local = bone_locals_from_globals(rig_global_m, bvh_parents)
    # Second pass catches sign discontinuities from the parent-inverse composition.
    bone_local[..., 3:7] = quat_sign_fix_per_joint(bone_local[..., 3:7])

    eulers_deg = _quat_to_zxy_euler_deg(bone_local[..., 3:7])

    if camera_translation in ("absolute", "centered"):
        cam_t = np.stack([
            unflip(np.asarray(frames[t][person_k]["pred_cam_t"], dtype=np.float32))
            for t in frame_indices
        ], axis=0).astype(np.float64)
        if camera_translation == "centered":
            cam_t = cam_t - cam_t[0:1]
        root_pos_m = bind_pos_m[body_root][None, :] + cam_t
    else:
        root_pos_m = np.tile(bind_pos_m[body_root], (n_frames, 1))

    lines: List[str] = ["HIERARCHY"]

    def _emit_joint(j: int, depth: int, is_root: bool) -> None:
        ind = "  " * depth
        keyword = "ROOT" if is_root else "JOINT"
        name = "Hips" if is_root else f"joint_{j:03d}"
        lines.append(f"{ind}{keyword} {name}")
        lines.append(ind + "{")
        o = (bind_pos_m[j] if is_root else offset_m[j]) * unit_scale
        lines.append(f"{ind}  OFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        if is_root:
            lines.append(ind + "  CHANNELS 6 Xposition Yposition Zposition "
                         "Zrotation Xrotation Yrotation")
        else:
            lines.append(ind + "  CHANNELS 3 Zrotation Xrotation Yrotation")
        kids = children_map[j]
        if kids:
            for c in kids:
                _emit_joint(c, depth + 1, is_root=False)
        else:
            # End Site (standard BVH spec) gives leaf bones a drawable length.
            lines.append(ind + "  End Site")
            lines.append(ind + "  {")
            tip = (offset_m[j] * unit_scale) * 0.3
            tip_norm = float(np.linalg.norm(tip))
            if tip_norm < 0.5 * unit_scale * 0.01:  # < 0.5 mm → fall back
                tip = np.array([0.0, 0.05 * unit_scale, 0.0], dtype=np.float64)
            lines.append(f"{ind}    OFFSET {tip[0]:.6f} {tip[1]:.6f} {tip[2]:.6f}")
            lines.append(ind + "  }")
        lines.append(ind + "}")

    _emit_joint(body_root, 0, is_root=True)

    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {1.0 / float(fps):.6f}")

    # Channel matrix per frame: root pos (3) + root rot (3) + non-root rots
    # (3 each), columns in `bvh_order`. savetxt is far faster than f-strings.
    non_root_idx = np.asarray(bvh_order[1:], dtype=np.int64)
    motion = np.concatenate([
        root_pos_m * unit_scale,                                 # (N, 3)
        eulers_deg[:, body_root].astype(np.float64),             # (N, 3)
        eulers_deg[:, non_root_idx, :].reshape(n_frames, -1),    # (N, 3*(NJ-1))
    ], axis=1)
    buf = io.StringIO()
    np.savetxt(buf, motion, fmt="%.6f")
    lines.append(buf.getvalue().rstrip("\n"))

    return ("\n".join(lines) + "\n").encode("utf-8")
