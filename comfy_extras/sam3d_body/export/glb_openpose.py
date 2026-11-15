"""GLB export — OpenPose 18-keypoint visualization mode.

Sourced from pose_data's `pred_keypoints_3d`, independent of the MHR rig. Each
track becomes an armature with a joint per keypoint; sphere markers and limbs
are skinned to those joints. Optional hands (`pred_keypoints_3d` 21..62) and
face landmarks (`pred_vertices` at fixed vertex IDs) extend the same armature.
Shared tables/palettes/mappings live in `glb_shared.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .glb_shared import (
    DWPOSE_HAND_COLORS_21,
    FACE_LANDMARK_COLORS,
    FACE_LANDMARK_TARGETS,
    GLBWriter,
    OPENPOSE18_TO_MHR70,
    OPENPOSE_18_NAMES,
    OPENPOSE_18_PAIRS,
    OPENPOSE_HAND21_NAMES,
    OPENPOSE_HAND21_TO_MHR70_L,
    OPENPOSE_HAND21_TO_MHR70_R,
    OPENPOSE_HAND_COLORS_21,
    OPENPOSE_HAND_PAIRS,
    OPENPOSE_RAINBOW_18,
    SCAIL_KEYPOINT_COLORS_18,
    SCAIL_LIMB_COLORS_17,
    collect_tracks,
    flat_shade_mesh,
    gaussian_smooth_positions,
    make_lit_material,
    quat_sign_fix_per_joint,
    resolve_openpose_keypoints_from_joints,
    rotation_align,
    rotmat_to_quat_np,
    select_face_landmark_vert_ids,
    smooth_shade_mesh,
    unflip,
    uv_sphere_unit,
)


def _finalize_skinned_mesh(
    verts: np.ndarray, faces: np.ndarray,
    joints: np.ndarray, weights: np.ndarray, vert_colors: np.ndarray,
    smooth_shade: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shade a skinned group mesh and pack per-vertex colors. Smooth keeps the
    indexed mesh; flat duplicates verts per face and gathers face-corner colors."""
    if smooth_shade:
        v_f, n_f, f_f, j_f, w_f = smooth_shade_mesh(verts, faces, joints, weights)
        return v_f, n_f, f_f, j_f, w_f, vert_colors.astype(np.float32)
    F = faces.shape[0]
    pre_faces = faces.copy()
    v_f, n_f, f_f, j_f, w_f = flat_shade_mesh(verts, faces, joints, weights)
    c_f = np.zeros((F * 3, 3), dtype=np.float32)
    for k in range(3):
        c_f[k::3] = vert_colors[pre_faces[:, k]]
    return v_f, n_f, f_f, j_f, w_f, c_f


def _pair_colors_from_kp(
    pairs: Tuple[Tuple[int, int], ...], kp_colors: np.ndarray, endpoint: int = 1,
) -> np.ndarray:
    """Per-limb color from `kp_colors`. `endpoint=1` (default) picks the distal
    vertex of each pair — the OpenPose per-finger gradient for base→tip fingers."""
    n = len(pairs)
    out = np.zeros((n, 3), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        out[i] = kp_colors[b if endpoint == 1 else a]
    return out


def _openpose_bind_at_rig_rest(
    pose_data: Dict[str, Any], *,
    include_hands: bool, face_vert_ids: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """OpenPose keypoint positions at the rig's REST pose, from the override's
    `bind_global_m` (joint rest TRS) and `rest_verts_m` (face landmarks).

    Used as the static-bind so the GLB's static POSITION sits at rig origin,
    matching skeletal mode and producing the same rest→scene-frame-0 transition.
    Returns None when the override lacks the needed mappings — caller then falls
    back to per-frame extraction (kp_seq[0])."""
    override = pose_data.get("_skeleton_override") if isinstance(pose_data, dict) else None
    if override is None or "bind_global_m" not in override:
        return None
    op18 = override.get("openpose18_joint_indices")
    if op18 is None:
        return None
    rest_pos = np.asarray(override["bind_global_m"], dtype=np.float32)[:, :3]
    op18_w = override.get("openpose18_joint_weights")
    parts: List[np.ndarray] = [
        resolve_openpose_keypoints_from_joints(
            rest_pos, np.asarray(op18, dtype=np.int64),
            weights=None if op18_w is None else np.asarray(op18_w, dtype=np.float32),
        )
    ]
    if include_hands:
        op21_r = override.get("openpose_hand21_r_joint_indices")
        op21_l = override.get("openpose_hand21_l_joint_indices")
        if op21_r is None or op21_l is None:
            return None
        op21_r_w = override.get("openpose_hand21_r_joint_weights")
        op21_l_w = override.get("openpose_hand21_l_joint_weights")
        parts.append(resolve_openpose_keypoints_from_joints(
            rest_pos, np.asarray(op21_r, dtype=np.int64),
            weights=None if op21_r_w is None else np.asarray(op21_r_w, dtype=np.float32),
        ))
        parts.append(resolve_openpose_keypoints_from_joints(
            rest_pos, np.asarray(op21_l, dtype=np.int64),
            weights=None if op21_l_w is None else np.asarray(op21_l_w, dtype=np.float32),
        ))
    if face_vert_ids is not None:
        rest_verts = override.get("rest_verts_m")
        if rest_verts is None:
            return None
        parts.append(np.asarray(rest_verts, dtype=np.float32)[face_vert_ids])
    return np.concatenate(parts, axis=0).astype(np.float32)


def _extract_openpose_keypoints(
    pose_data: Dict[str, Any], frame_indices: List[int], person_k: int,
) -> np.ndarray:
    """(N, 18, 3) OpenPose keypoints in rig-native Y-up metres.

    External-skeleton path: when the override carries `openpose18_joint_indices`
    ((18, 2) int32), synthesize from each person's `pred_joint_coords` (already
    Y-up, no flip). MHR70 path (default): re-index `pred_keypoints_3d` to COCO-18
    and un-flip y/z (stored y-down by sam3d_body).
    """
    frames = pose_data["frames"]
    N = len(frame_indices)
    out = np.zeros((N, 18, 3), dtype=np.float32)

    override = pose_data.get("_skeleton_override") if isinstance(pose_data, dict) else None
    op18 = override.get("openpose18_joint_indices") if override is not None else None
    if op18 is not None:
        op18 = np.asarray(op18, dtype=np.int64)
        if op18.ndim != 2 or op18.shape != (18, 2):
            raise ValueError(
                "build_glb_openpose: `openpose18_joint_indices` in "
                "`_skeleton_override` must be shape (18, 2); got "
                f"{tuple(op18.shape)}. Each row is (joint_a, joint_b); "
                "use joint_b=-1 for single-joint keypoints."
            )
        op18_w = override.get("openpose18_joint_weights")
        if op18_w is not None:
            op18_w = np.asarray(op18_w, dtype=np.float32)
            if op18_w.shape != (18,):
                raise ValueError(
                    "build_glb_openpose: `openpose18_joint_weights` must be "
                    f"shape (18,); got {tuple(op18_w.shape)}."
                )
        for t_idx, t in enumerate(frame_indices):
            person = frames[t][person_k]
            if "pred_joint_coords" not in person:
                raise ValueError(
                    "build_glb_openpose: external-skeleton path needs "
                    "per-frame `pred_joint_coords` (J, 3) on each person; "
                    f"missing at frame={t}, track={person_k}."
                )
            joints = np.asarray(person["pred_joint_coords"], dtype=np.float32)
            out[t_idx] = resolve_openpose_keypoints_from_joints(
                joints, op18, weights=op18_w,
            )
        return out

    for t_idx, t in enumerate(frame_indices):
        person = frames[t][person_k]
        if "pred_keypoints_3d" not in person:
            # External-skeleton producer without `openpose18_joint_indices`:
            # can't synthesize the 18-keypoint set.
            if override is not None:
                raise ValueError(
                    "build_glb_openpose: this pose_data carries "
                    "`_skeleton_override` but it doesn't include "
                    "`openpose18_joint_indices` and the per-frame person "
                    "dict is missing `pred_keypoints_3d`. Ask the upstream "
                    "node to populate `openpose18_joint_indices` on the "
                    "override (a (18, 2) int32 mapping into its joint list), "
                    "or switch SAM3DBody_ToGLB to `skeletal` mode."
                )
            present_keys = sorted(person.keys())
            raise ValueError(
                "build_glb_openpose: pose_data is missing "
                "`pred_keypoints_3d` (frame=%d, track=%d). Keys present "
                "on this person: %s. Re-run SAM3DBody_Predict — older "
                "saved pose_data may pre-date the field, and any "
                "intermediate node that rebuilds person dicts must "
                "preserve it."
                % (t, person_k, present_keys)
            )
        kp = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
        out[t_idx] = kp[OPENPOSE18_TO_MHR70]
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _extract_openpose_hand_keypoints(
    pose_data: Dict[str, Any], frame_indices: List[int], person_k: int,
) -> np.ndarray:
    """(N, 42, 3) right+left OpenPose hand keypoints (21+21) in rig-native Y-up.

    External-skeleton path: needs `openpose_hand21_{r,l}_joint_indices` ((21, 2)
    int32) in the override, resolved against `pred_joint_coords`. MHR70 path:
    re-orders `pred_keypoints_3d` 21..62 to OpenPose-21 (wrist + 5 fingers)."""
    frames = pose_data["frames"]
    N = len(frame_indices)
    out = np.zeros((N, 42, 3), dtype=np.float32)

    override = pose_data.get("_skeleton_override") if isinstance(pose_data, dict) else None
    op21_r = override.get("openpose_hand21_r_joint_indices") if override is not None else None
    op21_l = override.get("openpose_hand21_l_joint_indices") if override is not None else None
    if override is not None and (op21_r is not None or op21_l is not None):
        if op21_r is None or op21_l is None:
            raise ValueError(
                "build_glb_openpose: external skeleton must supply BOTH "
                "`openpose_hand21_r_joint_indices` and "
                "`openpose_hand21_l_joint_indices` for include_hands=True."
            )
        op21_r = np.asarray(op21_r, dtype=np.int64)
        op21_l = np.asarray(op21_l, dtype=np.int64)
        for arr, side in ((op21_r, "r"), (op21_l, "l")):
            if arr.ndim != 2 or arr.shape != (21, 2):
                raise ValueError(
                    f"build_glb_openpose: `openpose_hand21_{side}_joint_indices` "
                    f"must be shape (21, 2); got {tuple(arr.shape)}."
                )
        op21_r_w = override.get("openpose_hand21_r_joint_weights")
        op21_l_w = override.get("openpose_hand21_l_joint_weights")
        op21_r_w = (np.asarray(op21_r_w, dtype=np.float32)
                    if op21_r_w is not None else None)
        op21_l_w = (np.asarray(op21_l_w, dtype=np.float32)
                    if op21_l_w is not None else None)
        for t_idx, t in enumerate(frame_indices):
            person = frames[t][person_k]
            if "pred_joint_coords" not in person:
                raise ValueError(
                    "build_glb_openpose: external-skeleton path needs "
                    "per-frame `pred_joint_coords` for hands."
                )
            joints = np.asarray(person["pred_joint_coords"], dtype=np.float32)
            out[t_idx, :21] = resolve_openpose_keypoints_from_joints(
                joints, op21_r, weights=op21_r_w,
            )
            out[t_idx, 21:] = resolve_openpose_keypoints_from_joints(
                joints, op21_l, weights=op21_l_w,
            )
        return out

    for t_idx, t in enumerate(frame_indices):
        person = frames[t][person_k]
        if "pred_keypoints_3d" not in person:
            if override is not None:
                raise ValueError(
                    "build_glb_openpose: include_hands=True with an external "
                    "skeleton needs `openpose_hand21_r_joint_indices` and "
                    "`openpose_hand21_l_joint_indices` on `_skeleton_override`. "
                    "Disable hands or ask the upstream node to populate them."
                )
            raise ValueError(
                "build_glb_openpose: pose_data is missing `pred_keypoints_3d`."
            )
        kp = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
        out[t_idx, :21] = kp[OPENPOSE_HAND21_TO_MHR70_R]
        out[t_idx, 21:] = kp[OPENPOSE_HAND21_TO_MHR70_L]
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _extract_face_landmarks_from_verts(
    pose_data: Dict[str, Any], frame_indices: List[int], person_k: int,
    vert_ids: np.ndarray,
) -> np.ndarray:
    """(N, K_face, 3) face landmarks sampled from `pred_vertices` at the given
    vertex IDs, unflipped to Y-up. Per-frame deformation is already baked in."""
    frames = pose_data["frames"]
    N = len(frame_indices)
    K = int(vert_ids.shape[0])
    out = np.zeros((N, K, 3), dtype=np.float32)
    for t_idx, t in enumerate(frame_indices):
        person = frames[t][person_k]
        if "pred_vertices" not in person:
            raise ValueError(
                "build_glb_openpose: face_source='rig' needs `pred_vertices` "
                "on every frame — re-run Predict to populate it."
            )
        v = np.asarray(person["pred_vertices"], dtype=np.float32).reshape(-1, 3)
        out[t_idx] = v[vert_ids]
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _build_openpose_spheres(
    bind_kp_m: np.ndarray, radius_m: float, kp_colors: np.ndarray,
    base_joint_idx: int = 0,
    smooth_shade: bool = False,
    joint_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """UV sphere per keypoint, rigidly skinned to that keypoint's joint and
    vertex-colored from kp_colors. `base_joint_idx` offsets the emitted JOINTS_0
    indices (body=0, right hand=18, …); `joint_indices`, if given, sets explicit
    per-sphere indices so callers can skip keypoints (e.g. SCAIL head dots).
    Returns (verts, normals, faces, joints4, weights4, vert_colors)."""
    sv, sf = uv_sphere_unit()
    K = bind_kp_m.shape[0]
    Nv = sv.shape[0]
    Nf = sf.shape[0]
    out_v = np.zeros((K * Nv, 3), dtype=np.float32)
    out_n = np.zeros((K * Nv, 3), dtype=np.float32)
    out_f = np.zeros((K * Nf, 3), dtype=np.uint32)
    out_j = np.zeros((K * Nv, 4), dtype=np.uint16)
    out_w = np.zeros((K * Nv, 4), dtype=np.float32)
    out_c = np.zeros((K * Nv, 3), dtype=np.float32)
    for j in range(K):
        v_off = j * Nv
        out_v[v_off:v_off + Nv] = sv * radius_m + bind_kp_m[j]
        out_n[v_off:v_off + Nv] = sv
        out_f[j * Nf:(j + 1) * Nf] = sf + v_off
        out_j[v_off:v_off + Nv, 0] = int(joint_indices[j]) if joint_indices is not None else j + base_joint_idx
        out_w[v_off:v_off + Nv, 0] = 1.0
        out_c[v_off:v_off + Nv] = kp_colors[j]
    return _finalize_skinned_mesh(out_v, out_f, out_j, out_w, out_c, smooth_shade)


def _capsule_mesh_local(
    L: float, W: float, *,
    n_cap_lat: Optional[int] = None,
    n_body: Optional[int] = None,
    n_lon: Optional[int] = None,
    end_width_frac: float = 0.3,
    shape: str = "ellipsoid",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-limb mesh in limb-local frame along +Y from y=0 (head) to y=L (tail).

    `shape`:
      - 'ellipsoid' (default): hemisphere tips of radius `W * end_width_frac`,
        ellipsoidal sin(π·u) body profile (fat middle, narrow ends).
      - 'capsule': SCAIL "rig" limb — an OPEN cylinder of constant radius W,
        no caps. Pair with same-radius sphere markers so they cap the ends
        seamlessly (caps would bump out when sphere radius ≠ cap radius).

    A per-limb mesh is needed because cap height depends on width — one
    canonical mesh can't give true hemispheres for arbitrary L:W in ellipsoid.

    Returns (verts (Nv,3), faces (Nf,3), weights (Nv,2) head/tail, sums to 1).
    """
    W = max(1e-6, min(float(W), float(L) * 0.5 - 1e-6))
    if str(shape) == "capsule":
        # Open cylinder, no caps — sphere markers cap the ends (see docstring).
        cap_r = 0.0
        body_r = W
        if n_cap_lat is None:
            n_cap_lat = 0
        if n_body is None:
            n_body = 0
        if n_lon is None:
            n_lon = 16
    elif str(shape) == "ellipsoid":
        end_frac = float(min(0.95, max(0.05, end_width_frac)))
        cap_r = max(1e-7, W * end_frac)
        body_r = W
        # More body rings to sample the sin(π·u) curve.
        if n_cap_lat is None:
            n_cap_lat = 3
        if n_body is None:
            n_body = 7
        if n_lon is None:
            n_lon = 12
    else:
        raise ValueError(
            f"_capsule_mesh_local: unknown shape={shape!r} "
            "(expected 'ellipsoid' or 'capsule')"
        )
    if 2.0 * cap_r >= L:
        cap_r = max(0.0, L * 0.5 - 1e-6)
    body_len = float(L) - 2.0 * cap_r
    n_cap_lat = max(0, int(n_cap_lat))
    n_body = max(0, int(n_body))
    n_lon = max(3, int(n_lon))

    has_caps = n_cap_lat > 0

    verts: List[List[float]] = []
    head_pole = -1
    tail_pole = -1
    head_rings: List[int] = []
    tail_rings: List[int] = []

    if has_caps:
        # Head pole vertex at y=0 (south pole of head hemisphere).
        head_pole = len(verts)
        verts.append([0.0, 0.0, 0.0])
        # Head cap rings (i = 1..n_cap_lat). Last ring (i=n_cap_lat,
        # theta=π/2) is the head-body junction at y=cap_r, r=cap_r.
        for i in range(1, n_cap_lat + 1):
            theta = (np.pi * 0.5) * i / n_cap_lat
            y = cap_r * (1.0 - np.cos(theta))
            r = cap_r * np.sin(theta)
            head_rings.append(len(verts))
            for k in range(n_lon):
                phi = 2.0 * np.pi * k / n_lon
                verts.append([r * float(np.cos(phi)), float(y), r * float(np.sin(phi))])
    else:
        # Open cylinder: no caps, no pole. Add an end ring at y=0 directly.
        head_rings.append(len(verts))
        for k in range(n_lon):
            phi = 2.0 * np.pi * k / n_lon
            verts.append([body_r * float(np.cos(phi)), 0.0, body_r * float(np.sin(phi))])

    # Body intermediate rings (none for 'capsule', n_body=0 by default).
    body_rings: List[int] = []
    is_ellipsoid = str(shape) == "ellipsoid"
    for j in range(1, n_body + 1):
        u = j / (n_body + 1)
        y = cap_r + body_len * u
        if is_ellipsoid:
            r = cap_r + (body_r - cap_r) * float(np.sin(np.pi * u))
        else:
            r = body_r
        body_rings.append(len(verts))
        for k in range(n_lon):
            phi = 2.0 * np.pi * k / n_lon
            verts.append([r * float(np.cos(phi)), float(y), r * float(np.sin(phi))])

    if has_caps:
        # Tail cap rings (i = 0..n_cap_lat-1). First ring (i=0, theta=π/2)
        # is the body-tail junction at y=L-cap_r, r=cap_r; last
        # (i=n_cap_lat-1) is the ring just before the pole.
        for i in range(0, n_cap_lat):
            theta = (np.pi * 0.5) * (n_cap_lat - i) / n_cap_lat
            y = float(L) - cap_r * (1.0 - np.cos(theta))
            r = cap_r * np.sin(theta)
            tail_rings.append(len(verts))
            for k in range(n_lon):
                phi = 2.0 * np.pi * k / n_lon
                verts.append([r * float(np.cos(phi)), float(y), r * float(np.sin(phi))])
        tail_pole = len(verts)
        verts.append([0.0, float(L), 0.0])
    else:
        # Open cylinder end ring at y=L.
        tail_rings.append(len(verts))
        for k in range(n_lon):
            phi = 2.0 * np.pi * k / n_lon
            verts.append([body_r * float(np.cos(phi)), float(L), body_r * float(np.sin(phi))])

    faces: List[List[int]] = []

    if has_caps:
        # Head pole fan — outward (-Y) normal at the south pole.
        r0 = head_rings[0]
        for k in range(n_lon):
            a = r0 + k
            b = r0 + (k + 1) % n_lon
            faces.append([head_pole, a, b])

    # All inter-ring quads in axial order.
    all_rings = head_rings + body_rings + tail_rings
    for i in range(len(all_rings) - 1):
        rl = all_rings[i]
        rh = all_rings[i + 1]
        for k in range(n_lon):
            a = rl + k
            b = rl + (k + 1) % n_lon
            c = rh + (k + 1) % n_lon
            d = rh + k
            faces.append([a, c, b])
            faces.append([a, d, c])

    if has_caps:
        # Tail pole fan — outward (+Y) normal at the north pole.
        rL = tail_rings[-1]
        for k in range(n_lon):
            a = rL + k
            b = rL + (k + 1) % n_lon
            faces.append([tail_pole, b, a])

    v_arr = np.asarray(verts, dtype=np.float32)
    weights = np.zeros((v_arr.shape[0], 2), dtype=np.float32)
    weights[:, 1] = np.clip(v_arr[:, 1] / max(float(L), 1e-12), 0.0, 1.0)
    weights[:, 0] = 1.0 - weights[:, 1]

    return v_arr, np.asarray(faces, dtype=np.uint32), weights


def _scail_redirect_neck_stub(body_kp: np.ndarray) -> np.ndarray:
    """Replace the nose keypoint (idx 0) of a (...,18,3) array with a short
    neck stub (0.6 spine + 0.4 neck→nose), matching the capsule render."""
    out = body_kp.copy()
    neck = body_kp[..., 1, :]
    nose = body_kp[..., 0, :]
    mid_hip = 0.5 * (body_kp[..., 8, :] + body_kp[..., 11, :])

    def _unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-6)

    nose_vec = nose - neck
    nose_len = np.linalg.norm(nose_vec, axis=-1, keepdims=True)
    mixed = _unit(0.6 * _unit(neck - mid_hip) + 0.4 * _unit(nose_vec))
    out[..., 0, :] = neck + mixed * (nose_len * 0.5)
    return out


def _openpose_limb_rest_trs(
    bind_kp_m: np.ndarray, pairs: Tuple[Tuple[int, int], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-limb rest TRS: midpoints (K_pairs, 3) and unit a→b axes (or +Y if
    degenerate). Caller uses midpoints as rest translation, axes for alignment."""
    K_pairs = len(pairs)
    mid = np.zeros((K_pairs, 3), dtype=np.float32)
    axis = np.zeros((K_pairs, 3), dtype=np.float32)
    axis[:, 1] = 1.0
    for k, (a, b) in enumerate(pairs):
        a_pos = bind_kp_m[a]
        b_pos = bind_kp_m[b]
        mid[k] = 0.5 * (a_pos + b_pos)
        d = b_pos - a_pos
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            axis[k] = d / n
    return mid, axis


def _openpose_limb_anim_trs(
    kp_seq: np.ndarray, pairs: Tuple[Tuple[int, int], ...], rest_axes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame limb TRS: anim_mid (N, K_pairs, 3) midpoints and anim_quat
    (N, K_pairs, 4 xyzw) aligning each limb's rest axis to its frame-t axis.
    Drives skin_matrix(t) = T(mid_t)·R_t·T(-mid_rest) — rigid rotation about
    the rest midpoint, no LBS cross-section thinning."""
    N = kp_seq.shape[0]
    K_pairs = len(pairs)
    anim_mid = np.zeros((N, K_pairs, 3), dtype=np.float32)
    R = np.tile(np.eye(3, dtype=np.float32), (N, K_pairs, 1, 1))
    for k, (a, b) in enumerate(pairs):
        ax_rest = rest_axes[k]
        for t in range(N):
            a_pos = kp_seq[t, a]
            b_pos = kp_seq[t, b]
            anim_mid[t, k] = 0.5 * (a_pos + b_pos)
            d = b_pos - a_pos
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                R[t, k] = rotation_align(ax_rest, d / n)
    quat = rotmat_to_quat_np(R).astype(np.float32)
    return anim_mid, quat


def _build_openpose_sticks(
    bind_kp_m: np.ndarray, pairs: Tuple[Tuple[int, int], ...],
    half_width_m: float, pair_colors: np.ndarray,
    limb_joint_base_idx: int = 0,
    shape: str = "ellipsoid",
    smooth_shade: bool = False,
    end_width_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Capsule per limb pair (a, b), each sized to its own length/width so caps
    are true hemispheres regardless of L:W. Ellipsoid mode auto-clamps width to
    `length * 0.1` so short limbs don't look chunky.

    Rigid (weight=1) binding to a per-limb joint at `limb_joint_base_idx +
    limb_idx`, which the caller animates with midpoint translation + rotation
    (avoids LBS thinning). Returns (verts, normals, faces, joints4, weights4,
    vert_colors)."""
    canonical = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    out_v_chunks: List[np.ndarray] = []
    out_f_chunks: List[np.ndarray] = []
    out_j_chunks: List[np.ndarray] = []
    out_w_chunks: List[np.ndarray] = []
    out_c_chunks: List[np.ndarray] = []
    v_total = 0
    WIDTH_RATIO = 0.1
    MIN_WIDTH = 0.001
    is_capsule = str(shape) == "capsule"
    for limb_idx, (a, b) in enumerate(pairs):
        head = bind_kp_m[a]
        tail = bind_kp_m[b]
        direction = tail - head
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        unit_dir = direction / length
        R = rotation_align(canonical, unit_dir)
        if is_capsule:
            # Uniform radius — every bone the same width (clamped internally).
            half_width_eff = max(MIN_WIDTH, half_width_m)
        else:
            # Auto-thin so short face/ear limbs aren't chunky next to body limbs.
            half_width_eff = max(MIN_WIDTH, min(length * WIDTH_RATIO, half_width_m))

        v_local, f_local, _weights_unused = _capsule_mesh_local(
            length, half_width_eff, shape=shape, end_width_frac=end_width_frac,
        )
        v_world = v_local @ R.T + head
        Nv = v_local.shape[0]

        # Rigid binding to the per-limb joint; the 2-bone weights are discarded
        # (translation-only under LBS, would thin the cross-section).
        j_arr = np.zeros((Nv, 4), dtype=np.uint16)
        j_arr[:, 0] = limb_idx + limb_joint_base_idx
        w_arr = np.zeros((Nv, 4), dtype=np.float32)
        w_arr[:, 0] = 1.0
        c_arr = np.tile(pair_colors[limb_idx], (Nv, 1)).astype(np.float32)

        out_v_chunks.append(v_world)
        out_f_chunks.append(f_local + v_total)
        out_j_chunks.append(j_arr)
        out_w_chunks.append(w_arr)
        out_c_chunks.append(c_arr)
        v_total += Nv

    if not out_v_chunks:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32), np.zeros((0, 4), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))

    verts = np.concatenate(out_v_chunks, axis=0)
    faces = np.concatenate(out_f_chunks, axis=0)
    joints = np.concatenate(out_j_chunks, axis=0)
    weights = np.concatenate(out_w_chunks, axis=0)
    colors = np.concatenate(out_c_chunks, axis=0)
    return _finalize_skinned_mesh(verts, faces, joints, weights, colors, smooth_shade)


def build_glb_openpose(
    pose_data: Dict[str, Any],
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    marker_radius_m: float = 0.025,
    stick_radius_m: float = 0.008,
    include_hands: bool = False,
    hand_marker_radius_m: float = 0.0,
    hand_stick_radius_m: float = 0.0,
    hand_color_style: str = "dwpose",
    face_style: str = "disabled",
    face_marker_radius_m: float = 0.0,
    palette: str = "openpose",
    shape: str = "ellipsoid",
    smooth_shade: bool = False,
    material_roughness: float = 0.85,
    material_double_sided: bool = False,
    stick_end_width_frac: float = 0.6,
    bone_smooth_window: int = 0,
) -> bytes:
    """Build a GLB of an OpenPose-style 3D skeleton — sphere markers per keypoint
    plus colored sticks between limb pairs, one armature per track. Body from
    `pred_keypoints_3d`; optional hands (same source) and face landmarks
    (`pred_vertices`) extend each armature.

    Args:
        include_hands: append the 21+21 OpenPose hand keypoints per track.
        hand_marker_radius_m: hand sphere radius. 0 = auto = 0.4 × marker_radius_m.
        hand_stick_radius_m: hand limb half-width. 0 = auto = 0.5 × stick_radius_m.
        hand_color_style: 'dwpose' (default) = solid-blue dots + rainbow sticks;
            'openpose' = rainbow dots AND sticks.
        face_style: 'disabled' (default) | 'full' (~30 contour pts) | 'eyes_mouth'
            (eyes + outer-lip subset); sampled at vertex IDs from
            `canonical_colors["positions"]`.
        face_marker_radius_m: face landmark sphere radius. 0 = auto = 0.3 ×
            marker_radius_m. Rendered as dots only, no contour lines.
        palette: 'openpose' = rainbow gradient per keypoint; 'scail' = warm right
            / cool left, grey centerline, distinct per-limb colors.
    """
    is_scail = str(palette) == "scail"
    # SCAIL drops the face bones (13..16) and eye/ear spheres; keeps nose (idx 0,
    # the neck-stub tip) to cap the open cylinder. Matches the capsule render.
    body_pairs = OPENPOSE_18_PAIRS[:13] if is_scail else OPENPOSE_18_PAIRS
    body_sphere_kp = (np.arange(14, dtype=np.int64)
                      if is_scail else np.arange(18, dtype=np.int64))
    if is_scail:
        body_sphere_colors = SCAIL_KEYPOINT_COLORS_18
        body_stick_colors = SCAIL_LIMB_COLORS_17
    elif str(palette) == "openpose":
        # Same rainbow array drives both spheres and sticks.
        body_sphere_colors = OPENPOSE_RAINBOW_18
        body_stick_colors = OPENPOSE_RAINBOW_18
    else:
        raise ValueError(
            f"build_glb_openpose: unknown palette={palette!r} "
            "(expected 'openpose' or 'scail')"
        )

    if float(hand_marker_radius_m) <= 0.0:
        hand_marker_radius_m = float(marker_radius_m) * 0.4
    if float(hand_stick_radius_m) <= 0.0:
        hand_stick_radius_m = float(stick_radius_m) * 0.5
    if float(face_marker_radius_m) <= 0.0:
        face_marker_radius_m = float(marker_radius_m) * 0.3
    if hand_color_style == "dwpose":
        hand_sphere_colors = DWPOSE_HAND_COLORS_21
    elif hand_color_style == "openpose":
        hand_sphere_colors = OPENPOSE_HAND_COLORS_21
    else:
        raise ValueError(
            f"build_glb_openpose: unknown hand_color_style="
            f"{hand_color_style!r} (expected 'dwpose' or 'openpose')"
        )
    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_glb_openpose: no valid tracks in pose_data")

    # Eyes (6..13) + outer-lip ring (19..22) from FACE_LANDMARK_TARGETS.
    _EYES_MOUTH_IDX = np.array([6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22], dtype=np.int64)
    face_vert_ids: Optional[np.ndarray] = None
    face_target_idx = np.arange(len(FACE_LANDMARK_TARGETS), dtype=np.int64)
    if face_style in ("full", "eyes_mouth"):
        canonical_colors = pose_data.get("canonical_colors") or {}
        positions = canonical_colors.get("positions")
        if positions is None:
            raise ValueError(
                "build_glb_openpose: face_style needs "
                "pose_data['canonical_colors']['positions'] (computed at "
                "model load and attached by Predict). Ensure the SAM3DBody "
                "Loader+Predict ran upstream of this node."
            )
        if face_style == "eyes_mouth":
            face_target_idx = _EYES_MOUTH_IDX
        face_vert_ids = select_face_landmark_vert_ids(
            np.asarray(positions),
            face_mask=canonical_colors.get("face_mask"),
        )[face_target_idx]
    elif face_style != "disabled":
        raise ValueError(
            f"build_glb_openpose: unknown face_style={face_style!r} "
            "(expected 'disabled', 'full', or 'eyes_mouth')"
        )

    K_body = 18
    K_hands = 42 if include_hands else 0
    K_face = int(face_vert_ids.shape[0]) if face_vert_ids is not None else 0
    K = K_body + K_hands + K_face

    # Limb counts: one joint per stick pair. Limb joints carry translation +
    # rotation so each capsule rotates rigidly with its limb (no LBS thinning).
    K_body_limbs = len(body_pairs)
    K_hand_limbs = len(OPENPOSE_HAND_PAIRS) if include_hands else 0
    K_limbs = K_body_limbs + 2 * K_hand_limbs  # face has no sticks

    # Joint name list mirrors the keypoint stacking order: body → hands → face.
    joint_names: List[str] = [f"openpose_{n}" for n in OPENPOSE_18_NAMES]
    if include_hands:
        joint_names.extend([f"openpose_R_{n}" for n in OPENPOSE_HAND21_NAMES])
        joint_names.extend([f"openpose_L_{n}" for n in OPENPOSE_HAND21_NAMES])
    if K_face > 0:
        joint_names.extend([f"openpose_face_{FACE_LANDMARK_TARGETS[i][0]}"
                            for i in face_target_idx])

    # Limb joint names, stacked body → R-hand → L-hand to match the limb
    # joint ordering in skin.joints (after the K keypoint joints).
    limb_names: List[str] = [
        f"openpose_limb_{OPENPOSE_18_NAMES[a]}_{OPENPOSE_18_NAMES[b]}"
        for (a, b) in body_pairs
    ]
    if include_hands:
        for side in ("R", "L"):
            for (a, b) in OPENPOSE_HAND_PAIRS:
                limb_names.append(
                    f"openpose_{side}hand_limb_"
                    f"{OPENPOSE_HAND21_NAMES[a]}_{OPENPOSE_HAND21_NAMES[b]}"
                )

    w = GLBWriter()
    nodes: List[dict] = []
    meshes: List[dict] = []
    skins: List[dict] = []
    materials: List[dict] = []
    animations: List[dict] = []
    scene_root_indices: List[int] = []

    samplers: List[dict] = []
    channels: List[dict] = []
    for track_i, (person_k, frame_indices) in enumerate(tracks):
        body_seq = _extract_openpose_keypoints(pose_data, frame_indices, person_k)
        n_frames = body_seq.shape[0]
        if n_frames == 0:
            continue

        seq_chunks: List[np.ndarray] = [body_seq]
        if include_hands:
            seq_chunks.append(_extract_openpose_hand_keypoints(
                pose_data, frame_indices, person_k))
        if face_vert_ids is not None:
            seq_chunks.append(_extract_face_landmarks_from_verts(
                pose_data, frame_indices, person_k, face_vert_ids))
        kp_seq = np.concatenate(seq_chunks, axis=1)  # (N, K, 3)
        if bone_smooth_window and bone_smooth_window > 1:
            kp_seq = gaussian_smooth_positions(kp_seq, int(bone_smooth_window))

        # Static-bind = rig REST pose when available, else frame 0. The rest
        # bind keeps static POSITION at rig origin so viewers auto-center there
        # and the motion is visible (see _openpose_bind_at_rig_rest).
        bind_kp_m_rest = _openpose_bind_at_rig_rest(
            pose_data, include_hands=include_hands, face_vert_ids=face_vert_ids,
        )
        bind_kp_m = (bind_kp_m_rest if bind_kp_m_rest is not None
                     else kp_seq[0].astype(np.float32))

        if is_scail:  # nose → neck stub, matching the capsule render
            kp_seq[:, :K_body] = _scail_redirect_neck_stub(kp_seq[:, :K_body])
            bind_kp_m[:K_body] = _scail_redirect_neck_stub(bind_kp_m[:K_body])

        person_root: Dict[str, Any] = {"name": f"track{track_i:02d}", "children": []}
        nodes.append(person_root)
        person_root_idx = len(nodes) - 1
        scene_root_indices.append(person_root_idx)

        # K keypoint joint nodes (spheres bind here, translation only).
        joint_node_indices: List[int] = []
        for j in range(K):
            nodes.append({
                "name": joint_names[j],
                "translation": bind_kp_m[j].tolist(),
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            })
            joint_node_indices.append(len(nodes) - 1)
        person_root["children"].extend(joint_node_indices)

        # Per-limb rest + per-frame TRS; sticks bind rigidly to these joints.
        limb_rest_mids_list: List[np.ndarray] = []
        limb_rest_axes_list: List[np.ndarray] = []
        limb_anim_mids_list: List[np.ndarray] = []
        limb_anim_quats_list: List[np.ndarray] = []
        rmid_b, raxis_b = _openpose_limb_rest_trs(bind_kp_m[:K_body], body_pairs)
        amid_b, aquat_b = _openpose_limb_anim_trs(kp_seq[:, :K_body], body_pairs, raxis_b)
        limb_rest_mids_list.append(rmid_b)
        limb_rest_axes_list.append(raxis_b)
        limb_anim_mids_list.append(amid_b)
        limb_anim_quats_list.append(aquat_b)
        if include_hands:
            for h_off in (K_body, K_body + 21):
                rmid_h, raxis_h = _openpose_limb_rest_trs(
                    bind_kp_m[h_off:h_off + 21], OPENPOSE_HAND_PAIRS,
                )
                amid_h, aquat_h = _openpose_limb_anim_trs(
                    kp_seq[:, h_off:h_off + 21], OPENPOSE_HAND_PAIRS, raxis_h,
                )
                limb_rest_mids_list.append(rmid_h)
                limb_rest_axes_list.append(raxis_h)
                limb_anim_mids_list.append(amid_h)
                limb_anim_quats_list.append(aquat_h)
        limb_rest_mids = np.concatenate(limb_rest_mids_list, axis=0)
        limb_anim_mids = np.concatenate(limb_anim_mids_list, axis=1)
        limb_anim_quats = np.concatenate(limb_anim_quats_list, axis=1)
        # Hemisphere-align consecutive quats so LINEAR interp takes the short path.
        limb_anim_quats = quat_sign_fix_per_joint(limb_anim_quats).astype(np.float32)

        limb_joint_indices: List[int] = []
        for k in range(K_limbs):
            nodes.append({
                "name": limb_names[k],
                "translation": limb_rest_mids[k].tolist(),
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            })
            limb_joint_indices.append(len(nodes) - 1)
        person_root["children"].extend(limb_joint_indices)

        # Combined skin: keypoint joints then limb joints; IBM = T(-rest) for
        # both, yielding identity skin_matrix at rest.
        all_joint_indices = joint_node_indices + limb_joint_indices
        ibm = np.tile(np.eye(4, dtype=np.float32), (K + K_limbs, 1, 1))
        ibm[:K, :3, 3] = -bind_kp_m
        if K_limbs > 0:
            ibm[K:K + K_limbs, :3, 3] = -limb_rest_mids
        ibm_acc = w.add_mat4_f32(ibm.transpose(0, 2, 1).astype(np.float32))
        skins.append({
            "joints": all_joint_indices,
            "inverseBindMatrices": ibm_acc,
            "skeleton": person_root_idx,
        })
        skin_idx = len(skins) - 1

        # Per-group geometry. Spheres bind to keypoint joints [0, K); sticks to
        # limb joints [K, K+K_limbs). Stacked body → R-hand → L-hand → face.
        group_meshes: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray]] = []
        sp = _build_openpose_spheres(
            bind_kp_m[body_sphere_kp], float(marker_radius_m),
            body_sphere_colors[body_sphere_kp], base_joint_idx=0,
            smooth_shade=smooth_shade,
            joint_indices=body_sphere_kp,
        )
        st = _build_openpose_sticks(
            bind_kp_m[:K_body], body_pairs, float(stick_radius_m),
            body_stick_colors, limb_joint_base_idx=K,  # body limbs start at K
            shape=shape,
            smooth_shade=smooth_shade,
            end_width_frac=stick_end_width_frac,
        )
        group_meshes.append(sp)
        group_meshes.append(st)

        if include_hands:
            # Hand sticks stay rainbow per-finger; only dots switch under 'dwpose'.
            hand_pair_colors = _pair_colors_from_kp(
                OPENPOSE_HAND_PAIRS, OPENPOSE_HAND_COLORS_21, endpoint=1,
            )
            for hand_i, h_off in enumerate((K_body, K_body + 21)):  # right, then left
                h_bind = bind_kp_m[h_off:h_off + 21]
                group_meshes.append(_build_openpose_spheres(
                    h_bind, float(hand_marker_radius_m),
                    hand_sphere_colors, base_joint_idx=h_off,
                    smooth_shade=smooth_shade,
                ))
                group_meshes.append(_build_openpose_sticks(
                    h_bind, OPENPOSE_HAND_PAIRS, float(hand_stick_radius_m),
                    hand_pair_colors,
                    limb_joint_base_idx=K + K_body_limbs + hand_i * K_hand_limbs,
                    shape=shape,
                    smooth_shade=smooth_shade,
                    end_width_frac=stick_end_width_frac,
                ))

        if K_face > 0:
            f_off = K_body + K_hands
            f_bind = bind_kp_m[f_off:f_off + K_face]
            # DWPose face = dots only, no contour lines.
            group_meshes.append(_build_openpose_spheres(
                f_bind, float(face_marker_radius_m),
                FACE_LANDMARK_COLORS, base_joint_idx=f_off,
                smooth_shade=smooth_shade,
            ))

        primitives: List[dict] = []
        for (v_arr, n_arr, f_arr, j_arr, w_arr, c_arr) in group_meshes:
            if v_arr.shape[0] == 0:
                continue
            attrs = {
                "POSITION": w.add_vec3_f32(v_arr),
                "NORMAL": w.add_vec3_f32(n_arr),
                "JOINTS_0": w.add_joints_u16(j_arr),
                "WEIGHTS_0": w.add_weights_f32(w_arr),
                "COLOR_0": w.add_vec3_f32(c_arr),
            }
            materials.append(make_lit_material(
                roughness=material_roughness,
                double_sided=material_double_sided,
            ))
            primitives.append({
                "attributes": attrs,
                "indices": w.add_indices_u32(f_arr.reshape(-1)),
                "mode": 4,
                "material": len(materials) - 1,
            })
        if not primitives:
            continue
        meshes.append({"primitives": primitives})
        nodes.append({
            "name": f"track{track_i:02d}_openpose",
            "mesh": len(meshes) - 1,
            "skin": skin_idx,
        })
        person_root["children"].append(len(nodes) - 1)

        times = np.asarray(frame_indices, dtype=np.float32) / float(fps)
        time_acc = w.add_scalar_f32(times)
        for j in range(K):
            t_j = kp_seq[:, j, :].astype(np.float32)
            if (np.ptp(t_j, axis=0) < 1e-6).all():
                nodes[joint_node_indices[j]]["translation"] = t_j[0].tolist()
                continue
            acc = w.add_vec3_f32_anim(t_j)
            samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
            channels.append({
                "sampler": len(samplers) - 1,
                "target": {"node": joint_node_indices[j], "path": "translation"},
            })

        # Per-limb-joint translation + rotation; stationary limbs bake their
        # constant TRS into the node instead of an animation channel.
        for k in range(K_limbs):
            t_k = limb_anim_mids[:, k, :].astype(np.float32)
            if (np.ptp(t_k, axis=0) < 1e-6).all():
                nodes[limb_joint_indices[k]]["translation"] = t_k[0].tolist()
            else:
                acc = w.add_vec3_f32_anim(t_k)
                samplers.append({"input": time_acc, "output": acc,
                                 "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": limb_joint_indices[k], "path": "translation"},
                })
            q_k = limb_anim_quats[:, k, :].astype(np.float32)
            # Plain ptp is fine — signs already aligned by quat_sign_fix_per_joint.
            if (np.ptp(q_k, axis=0) < 1e-6).all():
                nodes[limb_joint_indices[k]]["rotation"] = q_k[0].tolist()
            else:
                acc = w.add_vec4_f32(q_k)
                samplers.append({"input": time_acc, "output": acc,
                                 "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": limb_joint_indices[k], "path": "rotation"},
                })

        if camera_translation != "off":
            frames = pose_data["frames"]
            cam_t = np.stack([
                unflip(np.asarray(frames[t][person_k]["pred_cam_t"], dtype=np.float32))
                for t in frame_indices
            ], axis=0)
            if camera_translation == "centered" and cam_t.shape[0] > 0:
                cam_t = cam_t - cam_t[0:1]
            if (np.ptp(cam_t, axis=0) < 1e-6).all():
                person_root["translation"] = cam_t[0].tolist()
            else:
                acc = w.add_vec3_f32_anim(cam_t)
                samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": person_root_idx, "path": "translation"},
                })

    if samplers:
        animations.append({
            "name": "all_tracks",
            "samplers": samplers, "channels": channels,
        })

    if not scene_root_indices:
        raise ValueError("build_glb_openpose: produced no tracks")

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "ComfyUI-SAM3DBody"},
        "scene": 0,
        "scenes": [{"nodes": scene_root_indices}],
        "nodes": nodes,
        "meshes": meshes,
        "skins": skins,
    }
    if materials:
        gltf["materials"] = materials
    if animations:
        gltf["animations"] = animations
    return w.to_bytes(gltf)
