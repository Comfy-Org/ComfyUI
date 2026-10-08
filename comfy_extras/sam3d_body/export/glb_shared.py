"""Shared GLB export helpers for SAM 3D Body pose_data.

Skeletal mode rebuilds the MHR 127-bone rig: per-frame local TRS from
param_transform on mhr_model_params, rest verts from a zero-pose forward,
sparse skinning compacted to glTF's 4-influence form, expression re-exposed as
72 morph targets. Camera-y-down data is un-flipped to glTF Y-up. Pose
correctives are dropped (glTF skinning can't represent them), so extreme joint
angles differ from the SAM3DBody renderer by the corrective amount.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from comfy_extras.sam3d_body.rasterizer import rainbow_colors_from_canonical

# fp32-rounded ln(2); exp(x * _LN2) matches the rig's own 2**x bit-for-bit.
_LN2 = 0.6931471824645996


# Quaternion / rotation helpers (xyzw, matching MHR rig)

def _euler_xyz_to_quat_np(angles: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> (x, y, z, w). Mirrors mhr_rig._euler_xyz_to_quat."""
    roll, pitch, yaw = angles[..., 0], angles[..., 1], angles[..., 2]
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return np.stack([x, y, z, w], axis=-1)


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """xyzw product. Mirrors mhr_rig._quat_multiply."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by unit xyzw q. Mirrors mhr_rig._quat_rotate."""
    axis = q[..., :3]
    r = q[..., 3:4]
    av = np.cross(axis, v, axis=-1)
    aav = np.cross(axis, av, axis=-1)
    return v + 2.0 * (av * r + aav)


def _skel_state_inverse_np(skel_state: np.ndarray) -> np.ndarray:
    """Inverse of (t, q, s). Normalizes q first so non-unit input is OK."""
    t = skel_state[..., :3]
    q = skel_state[..., 3:7]
    s = skel_state[..., 7:8]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    s_safe = np.where(np.abs(s) > 1e-12, s, 1.0)
    s_inv = 1.0 / s_safe
    q_inv = np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)
    t_inv = -s_inv * _quat_rotate_np(q_inv, t)
    return np.concatenate([t_inv, q_inv, s_inv], axis=-1)


def _skel_state_compose_np(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """s1 ∘ s2. Mirrors mhr_rig._skel_multiply."""
    t1 = s1[..., :3]
    q1 = s1[..., 3:7]
    sc1 = s1[..., 7:8]

    t2 = s2[..., :3]
    q2 = s2[..., 3:7]
    sc2 = s2[..., 7:8]
    # Defensive normalization to match the rig's `F.normalize` calls.
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    q2 = q2 / np.maximum(np.linalg.norm(q2, axis=-1, keepdims=True), 1e-12)
    t_res = t1 + sc1 * _quat_rotate_np(q1, t2)
    q_res = _quat_multiply_np(q1, q2)
    s_res = sc1 * sc2
    return np.concatenate([t_res, q_res, s_res], axis=-1)


def _gaussian_smooth_time(arr: np.ndarray, window: int) -> np.ndarray:
    """Edge-replicate Gaussian smoothing along time (sigma = window/4). float64."""
    a = np.asarray(arr, dtype=np.float64)
    n = a.shape[0]
    half = window // 2
    sigma = max(0.5, window / 4.0)
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-x * x / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    padded = np.concatenate([
        np.broadcast_to(a[:1], (half,) + a.shape[1:]),
        a,
        np.broadcast_to(a[-1:], (half,) + a.shape[1:]),
    ], axis=0)
    out = np.zeros_like(a)
    for k, w in enumerate(kernel):
        out += w * padded[k:k + n]
    return out


def gaussian_smooth_quats(q_seq: np.ndarray, window: int) -> np.ndarray:
    """Smooth a (N, NJ, 4) quaternion sequence along time: sign-align per joint,
    convolve per-component, renormalize. Calms bone spikes at extreme poses."""
    if window <= 1 or q_seq.shape[0] < 2:
        return q_seq
    out = _gaussian_smooth_time(quat_sign_fix_per_joint(q_seq), window)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def gaussian_smooth_positions(seq: np.ndarray, window: int) -> np.ndarray:
    """Smooth a (N, K, 3) position sequence along time. Calms jittery keypoint
    tracks before the openpose rig derives sphere translations + limb TRS."""
    if window <= 1 or seq.shape[0] < 2:
        return seq
    return _gaussian_smooth_time(seq, window).astype(np.float32)


def quat_sign_fix_per_joint(q_seq: np.ndarray) -> np.ndarray:
    """Walk (N, NJ, 4) along time, flipping sign when consecutive frames sit on
    opposite hemispheres. Avoids long-path slerp glitches. fp64 internally."""
    out = np.array(q_seq, dtype=np.float64, copy=True)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.maximum(norms, 1e-12)
    for t in range(1, out.shape[0]):
        dots = (out[t - 1] * out[t]).sum(axis=-1)
        sign = np.where(dots < 0.0, -1.0, 1.0)[:, None]
        out[t] = out[t] * sign
    return out.astype(np.float32)


def bone_locals_from_globals(rig_global: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Globals (N, NJ, 8) + parents -> per-bone local TRS so FK reproduces
    rig_global. local = inverse(parent_global) ∘ child_global, robust to
    hierarchy-convention mismatches in `parents`."""
    N, NJ, _ = rig_global.shape
    bone_local = np.zeros_like(rig_global)
    for j in range(NJ):
        p = int(parents[j])
        if 0 <= p < NJ and p != j:
            parent_g = rig_global[:, p]
            parent_g_inv = _skel_state_inverse_np(parent_g)
            bone_local[:, j] = _skel_state_compose_np(parent_g_inv, rig_global[:, j])
        else:
            bone_local[:, j] = rig_global[:, j]
    return bone_local


def _quat_to_mat3_np(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / n, 0.0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - s * (y * y + z * z)
    R[..., 0, 1] = s * (x * y - z * w)
    R[..., 0, 2] = s * (x * z + y * w)
    R[..., 1, 0] = s * (x * y + z * w)
    R[..., 1, 1] = 1 - s * (x * x + z * z)
    R[..., 1, 2] = s * (y * z - x * w)
    R[..., 2, 0] = s * (x * z - y * w)
    R[..., 2, 1] = s * (y * z + x * w)
    R[..., 2, 2] = 1 - s * (x * x + y * y)
    return R


def collect_tracks(pose_data: Dict[str, Any], track_index: int) -> List[Tuple[int, List[int]]]:
    """List of (person_index, frame_indices). track_index == -1 means every
    present track; empty tracks dropped. Same person index = same subject."""
    frames = pose_data["frames"]
    max_p = max((len(f) for f in frames), default=0)
    if max_p == 0:
        return []
    if track_index >= 0:
        if track_index >= max_p:
            return []
        wanted = [track_index]
    else:
        wanted = list(range(max_p))

    tracks: List[Tuple[int, List[int]]] = []
    for k in wanted:
        valid = [t for t, fr in enumerate(frames) if k < len(fr)]
        if valid:
            tracks.append((k, valid))
    return tracks


# glTF binary builder


_FLOAT = 5126
_USHORT = 5123
_UINT = 5125
_BYTE_ARRAY = 34962
_BYTE_ELEMENT = 34963


def _pad4(buf: bytes, fill: bytes = b"\x00") -> bytes:
    n = (4 - (len(buf) % 4)) % 4
    return buf + fill * n


class GLBWriter:
    """Builds a single .glb from incremental accessor/bufferView additions."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.bufferViews: List[dict] = []
        self.accessors: List[dict] = []

    def _add_view(self, data: bytes, *, target: Optional[int] = None) -> int:
        offset = len(self._buffer)
        self._buffer += data
        # 4-byte align so subsequent views start on a boundary.
        pad = (4 - (offset + len(data)) % 4) % 4
        if pad:
            self._buffer += b"\x00" * pad
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bufferViews.append(view)
        return len(self.bufferViews) - 1

    def add_vec3_f32(self, arr: np.ndarray, *, target: int = _BYTE_ARRAY) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes(), target=target)
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "VEC3",
            "min": a.min(axis=0).tolist() if a.shape[0] else [0.0, 0.0, 0.0],
            "max": a.max(axis=0).tolist() if a.shape[0] else [0.0, 0.0, 0.0],
        })
        return len(self.accessors) - 1

    def add_vec3_f32_no_minmax(self, arr: np.ndarray) -> int:
        """Morph-target POSITIONs: spec lets us skip min/max."""
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes(), target=_BYTE_ARRAY)
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "VEC3",
        })
        return len(self.accessors) - 1

    def add_indices_u32(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.uint32).reshape(-1)
        view_idx = self._add_view(a.tobytes(), target=_BYTE_ELEMENT)
        self.accessors.append({
            "bufferView": view_idx, "componentType": _UINT,
            "count": int(a.size), "type": "SCALAR",
        })
        return len(self.accessors) - 1

    def add_scalar_f32(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        view_idx = self._add_view(a.tobytes())
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": int(a.size), "type": "SCALAR",
            "min": [float(a.min())] if a.size else [0.0],
            "max": [float(a.max())] if a.size else [0.0],
        })
        return len(self.accessors) - 1

    def add_scalar_f32_flat(self, arr: np.ndarray, count: int) -> int:
        """Animation-output scalars: `count` is keyframes, not floats (morph
        weight tracks store N_morph weights per keyframe)."""
        a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        view_idx = self._add_view(a.tobytes())
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": int(count), "type": "SCALAR",
        })
        return len(self.accessors) - 1

    def add_vec3_f32_anim(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes())
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "VEC3",
        })
        return len(self.accessors) - 1

    def add_vec4_f32(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes())
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "VEC4",
        })
        return len(self.accessors) - 1

    def add_mat4_f32(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes())
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "MAT4",
        })
        return len(self.accessors) - 1

    def add_joints_u16(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.uint16)
        view_idx = self._add_view(a.tobytes(), target=_BYTE_ARRAY)
        self.accessors.append({
            "bufferView": view_idx, "componentType": _USHORT,
            "count": a.shape[0], "type": "VEC4",
        })
        return len(self.accessors) - 1

    def add_weights_f32(self, arr: np.ndarray) -> int:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        view_idx = self._add_view(a.tobytes(), target=_BYTE_ARRAY)
        self.accessors.append({
            "bufferView": view_idx, "componentType": _FLOAT,
            "count": a.shape[0], "type": "VEC4",
        })
        return len(self.accessors) - 1

    def to_bytes(self, gltf: dict) -> bytes:
        gltf["buffers"] = [{"byteLength": len(self._buffer)}]
        gltf["bufferViews"] = self.bufferViews
        gltf["accessors"] = self.accessors

        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_padded = _pad4(json_bytes, fill=b" ")
        bin_padded = _pad4(bytes(self._buffer))

        total = 12 + 8 + len(json_padded) + 8 + len(bin_padded)
        header = struct.pack("<4sII", b"glTF", 2, total)
        json_chunk = struct.pack("<II", len(json_padded), 0x4E4F534A) + json_padded
        bin_chunk = struct.pack("<II", len(bin_padded), 0x004E4942) + bin_padded

        return header + json_chunk + bin_chunk


# Inverse of mhr_head's `verts[..., [1, 2]] *= -1`: camera-y-down → glTF Y-up.
def unflip(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float32, copy=True)
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


_BAKEABLE_SHADERS = {
    "default", "rainbow",
    "rainbow_face_normal", "rainbow_face_semantic",
}


def bake_vertex_colors(
    canonical_colors: Optional[Dict[str, np.ndarray]],
    shader: str,
    rainbow_tilt_x_deg: float,
    rainbow_tilt_z_deg: float,
    pastel_mix: float,
) -> Optional[np.ndarray]:
    """Per-vertex RGB matching the renderer's shader preset. Returns (N_v, 3)
    float32 in [0, 1], or None for `default` (use the viewer's material)."""
    if shader == "default" or canonical_colors is None:
        return None

    positions = np.asarray(canonical_colors["positions"], dtype=np.float32)

    vcolor = rainbow_colors_from_canonical(
        positions, tilt_x_deg=rainbow_tilt_x_deg, tilt_z_deg=rainbow_tilt_z_deg,
    ).copy()
    if shader in ("rainbow_face_normal", "rainbow_face_semantic"):
        face_mask = canonical_colors.get("face_mask")
        if face_mask is not None and np.asarray(face_mask).any():
            if shader == "rainbow_face_normal":
                norm = np.asarray(canonical_colors["norm"], dtype=np.float32)
                vcolor[face_mask] = norm[face_mask]
            else:  # rainbow_face_semantic
                sem = np.asarray(canonical_colors["face_region_rgb"], dtype=np.float32)
                assigned = sem.sum(axis=1) > 0
                vcolor[assigned] = sem[assigned]

    # SCAIL-style per-person pastel mix toward white (track 0 = full color).
    pm = max(0.0, min(1.0, float(pastel_mix)))
    if pm > 0:
        vcolor = vcolor * (1.0 - pm) + pm
    return np.clip(vcolor, 0.0, 1.0).astype(np.float32)


def compute_pastel_mix(track_i: int, falloff: float) -> float:
    """SCAIL-style desaturation: track 0 = 0.0, track k = 1 - falloff^k."""
    f = max(0.1, min(1.0, float(falloff)))
    return 0.0 if track_i == 0 else (1.0 - f ** track_i)


def compute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    vn = np.zeros_like(verts, dtype=np.float32)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln < 1e-8] = 1.0
    return (vn / ln).astype(np.float32)


def _parents_from_pmi(rig: Any) -> np.ndarray:
    """Parent index per joint from skel_pmi ((2, 266): row 0 child, row 1
    parent, split into BFS levels by skel_pmi_buffer_sizes). Roots = -1."""
    NJ = int(rig.NUM_JOINTS)
    pmi = rig.skel_pmi.cpu().numpy()
    sizes = rig.skel_pmi_buffer_sizes.cpu().numpy().tolist()
    parents = np.full(NJ, -1, dtype=np.int32)
    offset = 0
    for sz in sizes:
        if sz > 0:
            src = pmi[0, offset:offset + sz].astype(np.int64)
            tgt = pmi[1, offset:offset + sz].astype(np.int64)
            parents[src] = tgt
        offset += sz
    return parents


def _get_skeleton_override(pose_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return ``_skeleton_override`` dict if present. Non-MHR skeletons supply
    this to bypass MHR rig extraction (see ComfyUI-Kimodo).

    Required keys:
        parents:             (NJ,) int32, -1 = root
        bind_global_m:       (NJ, 8) f32 — [t.xyz | q.xyzw | scale], meters
        lbs_compact_joints:  (V, 8) uint16 — pre-compacted skin influences
        lbs_compact_weights: (V, 8) f32
        lbs_compact_max_inf: int — actual max influences (≤ 8)
        rest_verts_m:        (V, 3) f32
        faces:               (F, 3) uint32

    Optional (enable openpose mode on external rigs):
        per_frame_y_down:    bool — False if pred_joint_coords are already Y-up
                             (kimodo). Default True (MHR).
        openpose18_joint_indices:        (18, 2) int32 — body keypoint → (a, b)
                             joints, resolved against `pred_joint_coords`.
                             b == -1 = single joint, else midpoint of (a, b).
        openpose18_joint_weights:        (18,) f32 — blend w: w*a + (1-w)*b
                             (default 0.5; outside [0,1] extrapolates; ignored
                             when b == -1).
        openpose_hand21_{r,l}_joint_indices: (21, 2) int32 — per-hand keypoint
                             maps; both required for include_hands=True.
        openpose_hand21_{r,l}_joint_weights: (21,) f32 — optional, same as above.
    """
    if pose_data is None:
        return None
    return pose_data.get("_skeleton_override")


def extract_rig_static(model: Any, pose_data: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Static rig buffers as numpy. If `pose_data` carries `_skeleton_override`,
    use that instead of MHR-specific `model.head_pose.mhr` buffers."""
    override = _get_skeleton_override(pose_data)
    if override is not None:
        # External rig: skin pre-compacted, bind global supplied directly.
        parents = np.asarray(override["parents"], dtype=np.int32)
        rest_v = np.asarray(override["rest_verts_m"], dtype=np.float32)
        # BVH needs parent-relative bone offsets (cm); derive from bind globals.
        bind_global_m = np.asarray(override["bind_global_m"], dtype=np.float32)
        local_bind = bone_locals_from_globals(bind_global_m[None], parents)[0]
        joint_translation_offsets = (local_bind[:, :3] * 100.0).astype(np.float32)
        return {
            "parents": parents,
            "parents_pmi": parents,
            "joint_translation_offsets": joint_translation_offsets,  # (NJ, 3) cm
            "lbs_compact_joints": np.asarray(override["lbs_compact_joints"], dtype=np.uint16),
            "lbs_compact_weights": np.asarray(override["lbs_compact_weights"], dtype=np.float32),
            "lbs_compact_max_inf": int(override.get("lbs_compact_max_inf", 4)),
            "faces": np.asarray(override["faces"], dtype=np.uint32),
            "num_joints": int(parents.shape[0]),
            "num_verts": int(rest_v.shape[0]),
            "num_expr": 0,
            "num_shape": 0,
            "_external": True,
        }

    inner = model.model if hasattr(model, "model") else model
    rig = inner.head_pose.mhr
    head = inner.head_pose

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy()

    # `skel_joint_parents` encodes the anatomical hierarchy; pmi-derived order
    # is BFS-optimized for parallel FK and may include traversal quirks.
    explicit_parents = _np(rig.skel_joint_parents).astype(np.int32)
    return {
        "parents": explicit_parents,                                   # (127,) int32, -1 = root
        "parents_pmi": _parents_from_pmi(rig),                         # kept for FK-related uses
        "joint_translation_offsets": _np(rig.skel_joint_translation_offsets),  # (127, 3) cm
        "joint_prerotations": _np(rig.skel_joint_prerotations),        # (127, 4) xyzw
        "param_transform": _np(rig.param_transform),                   # (889, 249)
        "lbs_inverse_bind_pose": _np(rig.lbs_inverse_bind_pose),       # (127, 8)
        "lbs_skin_weights": _np(rig.lbs_skin_weights),                 # (NNZ,)
        "lbs_skin_indices": _np(rig.lbs_skin_indices).astype(np.int64),  # (NNZ,)
        "lbs_vert_indices": _np(rig.lbs_vert_indices).astype(np.int64),  # (NNZ,)
        "expr_basis": _np(rig.expr_basis),                             # (72, 18439, 3)
        "faces": _np(head.faces).astype(np.uint32),                    # (36874, 3)
        "num_joints": int(rig.NUM_JOINTS),
        "num_verts": int(rig.NUM_VERTS),
        "num_expr": int(rig.NUM_EXPR),
        "num_shape": int(rig.NUM_IDENTITY),
        "_external": False,
    }


def compact_skin_to_n(
    skin_indices: np.ndarray, vert_indices: np.ndarray, weights: np.ndarray,
    num_verts: int, max_inf: int = 8,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sparse (joint, vert, weight) triplets -> dense (joints, weights) of shape
    (V, max_inf), keeping the largest influences and renormalizing. `actual_max`
    lets the caller skip JOINTS_1/WEIGHTS_1 when nothing exceeds 4 influences."""
    joints = np.zeros((num_verts, max_inf), dtype=np.uint16)
    out_w = np.zeros((num_verts, max_inf), dtype=np.float32)
    counts = np.zeros(num_verts, dtype=np.int32)

    if vert_indices.size:
        # Group by vert, weights descending within each group.
        order = np.lexsort((-weights, vert_indices))
        vi_sorted = vert_indices[order]
        sk_sorted = skin_indices[order]
        w_sorted = weights[order]

        # Per-row rank within its vertex group (0 at each group start).
        n = vi_sorted.size
        group_start = np.empty(n, dtype=bool)
        group_start[0] = True
        np.not_equal(vi_sorted[1:], vi_sorted[:-1], out=group_start[1:])
        pos = np.arange(n, dtype=np.int64)
        group_start_pos = np.maximum.accumulate(np.where(group_start, pos, 0))
        rank = pos - group_start_pos

        keep = rank < max_inf
        vk = vi_sorted[keep]
        rk = rank[keep]
        joints[vk, rk] = sk_sorted[keep].astype(np.uint16, copy=False)
        out_w[vk, rk] = w_sorted[keep].astype(np.float32, copy=False)

        true_counts = np.bincount(vi_sorted, minlength=num_verts)
        np.minimum(true_counts, max_inf, out=counts, casting="unsafe")

    sums = out_w.sum(axis=1, keepdims=True)
    nz = sums.squeeze(-1) > 0
    out_w[nz] /= sums[nz]
    zero_w = ~nz
    if zero_w.any():
        out_w[zero_w, 0] = 1.0
    actual_max = int(counts.max()) if counts.size else 0
    return joints, out_w, actual_max


def zero_pose_rest_verts(
    model: Any, shape_params: np.ndarray, expr_zero: bool = True,
    pose_data: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Zero pose + this subject's shape -> rest verts (V, 3) in rig-native Y-up
    meters. External path returns `rest_verts_m` directly."""
    override = _get_skeleton_override(pose_data)
    if override is not None:
        return np.asarray(override["rest_verts_m"], dtype=np.float32)
    inner = model.model if hasattr(model, "model") else model
    head = inner.head_pose
    rig = head.mhr
    device = rig.scale_mean.device if hasattr(rig, "scale_mean") else next(rig.parameters()).device
    dtype = next(rig.parameters()).dtype

    sp = torch.from_numpy(np.ascontiguousarray(shape_params, dtype=np.float32)).to(device)
    if sp.ndim == 1:
        sp = sp.unsqueeze(0)
    # rig.forward(shape, model_params, expr); zero pose + zero expr.
    model_params = torch.zeros(1, 204, device=device, dtype=dtype)
    expr = torch.zeros(1, 72, device=device, dtype=dtype)
    verts, _ = rig(sp.to(dtype), model_params, expr, apply_correctives=False)
    verts_m = verts[0].cpu().float().numpy() / 100.0  # cm -> m
    return verts_m.astype(np.float32)


def global_skel_state_per_frame(
    model: Any, mhr_model_params: np.ndarray,
) -> np.ndarray:
    """Rig FK over a batch of mhr_model_params -> (N, NJ, 8) = (t cm, q xyzw,
    scale). Bones are shape/expression-independent, so pass zeros."""
    inner = model.model if hasattr(model, "model") else model
    rig = inner.head_pose.mhr
    device = next(rig.parameters()).device
    dtype = next(rig.parameters()).dtype

    N = mhr_model_params.shape[0]
    mp = torch.from_numpy(np.ascontiguousarray(mhr_model_params, dtype=np.float32)).to(device=device, dtype=dtype)
    sp = torch.zeros(N, rig.NUM_IDENTITY, device=device, dtype=dtype)
    expr = torch.zeros(N, rig.NUM_EXPR, device=device, dtype=dtype)

    _, skel_state = rig(sp, mp, expr, apply_correctives=False)
    return skel_state.cpu().float().numpy()  # (N, NJ, 8) cm


def rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) xyzw. Shepperd 1978, largest-component pick.
    Cross-frame sign-fixing is the caller's job."""
    shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3).astype(np.float64)
    M = Rf.shape[0]
    q = np.zeros((M, 4), dtype=np.float64)

    trace = Rf[:, 0, 0] + Rf[:, 1, 1] + Rf[:, 2, 2]
    m1 = trace > 0
    if m1.any():
        S = np.sqrt(trace[m1] + 1.0) * 2.0
        q[m1, 3] = 0.25 * S
        q[m1, 0] = (Rf[m1, 2, 1] - Rf[m1, 1, 2]) / S
        q[m1, 1] = (Rf[m1, 0, 2] - Rf[m1, 2, 0]) / S
        q[m1, 2] = (Rf[m1, 1, 0] - Rf[m1, 0, 1]) / S

    rest = ~m1
    m2 = rest & (Rf[:, 0, 0] > Rf[:, 1, 1]) & (Rf[:, 0, 0] > Rf[:, 2, 2])
    if m2.any():
        S = np.sqrt(1.0 + Rf[m2, 0, 0] - Rf[m2, 1, 1] - Rf[m2, 2, 2]) * 2.0
        q[m2, 3] = (Rf[m2, 2, 1] - Rf[m2, 1, 2]) / S
        q[m2, 0] = 0.25 * S
        q[m2, 1] = (Rf[m2, 0, 1] + Rf[m2, 1, 0]) / S
        q[m2, 2] = (Rf[m2, 0, 2] + Rf[m2, 2, 0]) / S

    m3 = rest & ~m2 & (Rf[:, 1, 1] > Rf[:, 2, 2])
    if m3.any():
        S = np.sqrt(1.0 + Rf[m3, 1, 1] - Rf[m3, 0, 0] - Rf[m3, 2, 2]) * 2.0
        q[m3, 3] = (Rf[m3, 0, 2] - Rf[m3, 2, 0]) / S
        q[m3, 0] = (Rf[m3, 0, 1] + Rf[m3, 1, 0]) / S
        q[m3, 1] = 0.25 * S
        q[m3, 2] = (Rf[m3, 1, 2] + Rf[m3, 2, 1]) / S

    m4 = rest & ~m2 & ~m3
    if m4.any():
        S = np.sqrt(1.0 + Rf[m4, 2, 2] - Rf[m4, 0, 0] - Rf[m4, 1, 1]) * 2.0
        q[m4, 3] = (Rf[m4, 1, 0] - Rf[m4, 0, 1]) / S
        q[m4, 0] = (Rf[m4, 0, 2] + Rf[m4, 2, 0]) / S
        q[m4, 1] = (Rf[m4, 1, 2] + Rf[m4, 2, 1]) / S
        q[m4, 2] = 0.25 * S

    return q.reshape(shape + (4,)).astype(np.float32)


def global_skel_state_from_pose_data(
    pose_data: Dict[str, Any], frame_indices: List[int], person_k: int,
    NJ: int, *, joint_coords_y_down: bool = True,
) -> np.ndarray:
    """Per-frame skel_state from stored pred_global_rots + pred_joint_coords,
    bypassing rig.forward. Returns (N, NJ, 8) in meters, MHR-native frame.

    pred_global_rots are MHR-native. pred_joint_coords are y-down for MHR
    (un-flipped when `joint_coords_y_down=True`); external rigs store y-up
    (pass False). Scale defaults to 1 (not preserved in pose_data)."""
    frames = pose_data["frames"]
    N = len(frame_indices)
    rotmat = np.zeros((N, NJ, 3, 3), dtype=np.float32)
    coords = np.zeros((N, NJ, 3), dtype=np.float32)
    for t_idx, t in enumerate(frame_indices):
        person = frames[t][person_k]
        rotmat[t_idx] = np.asarray(person["pred_global_rots"], dtype=np.float32)[:NJ]
        coords[t_idx] = np.asarray(person["pred_joint_coords"], dtype=np.float32)[:NJ]
    if joint_coords_y_down:
        coords[..., 1] *= -1.0
        coords[..., 2] *= -1.0
    quat = rotmat_to_quat_np(rotmat)
    skel_state = np.zeros((N, NJ, 8), dtype=np.float32)
    skel_state[..., :3] = coords
    skel_state[..., 3:7] = quat
    skel_state[..., 7] = 1.0
    return skel_state


def bind_skel_state(model: Any, pose_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Rig FK with all-zero params -> bind-pose global skel state (NJ, 8) in cm,
    used as bones' static TRS. External rig: convert `bind_global_m` m -> cm."""
    override = _get_skeleton_override(pose_data)
    if override is not None:
        bind_m = np.asarray(override["bind_global_m"], dtype=np.float32).copy()
        bind_m[:, :3] *= 100.0
        return bind_m
    zero_mp = np.zeros((1, 204), dtype=np.float32)
    return global_skel_state_per_frame(model, zero_mp)[0]


@dataclass
class Rig:
    """Normalized static rig for the GLB/BVH exporters, source-independent: MHR
    model or inline `pose_data["_skeleton_override"]` (external rigs). Consumers
    never branch on the source. Only `rest_verts_m` is source-dependent — MHR
    expands it from `shape_params`; external rigs ship it fixed.
    """
    parents: np.ndarray             # (NJ,) int32, -1 = root
    joint_offsets_cm: np.ndarray    # (NJ, 3) parent-relative bind offsets, cm
    bind_global_cm: np.ndarray      # (NJ, 8) bind global [t cm | q xyzw | s]
    lbs_joints: np.ndarray          # (V, 8) uint16 — compacted skin influences
    lbs_weights: np.ndarray         # (V, 8) f32
    lbs_max_inf: int                # ≤ 8; lets callers skip JOINTS_1 when ≤ 4
    faces: np.ndarray               # (F, 3) uint32
    num_joints: int
    num_verts: int
    num_expr: int                   # 0 = no face morphs
    per_frame_y_down: bool          # pred_joint_coords stored y-down (MHR) vs y-up (external)
    can_rerun_fk: bool              # True = per-frame FK re-runnable from mhr_model_params
    expr_basis: Optional[np.ndarray] = None      # (E, V, 3) cm — MHR only
    _model: Any = None
    _rest_override: Optional[np.ndarray] = None  # (V, 3) m — external only

    @property
    def bind_global_m(self) -> np.ndarray:
        b = self.bind_global_cm.astype(np.float32).copy()
        b[:, :3] *= 0.01
        return b

    def rest_verts_m(self, shape_params: np.ndarray) -> np.ndarray:
        """Zero-pose rest verts (V, 3) in rig-native Y-up metres."""
        if self._rest_override is not None:
            return self._rest_override
        return zero_pose_rest_verts(self._model, shape_params)

    @classmethod
    def from_pose_data(cls, pose_data: Optional[Dict[str, Any]], model: Any = None) -> "Rig":
        rs = extract_rig_static(model, pose_data)
        external = bool(rs.get("_external", False))
        if external:
            joints8 = np.asarray(rs["lbs_compact_joints"], dtype=np.uint16)
            weights8 = np.asarray(rs["lbs_compact_weights"], dtype=np.float32)
            max_inf = int(rs["lbs_compact_max_inf"])
            override = _get_skeleton_override(pose_data) or {}
            per_y_down = bool(override.get("per_frame_y_down", False))
            rest_override = np.asarray(override["rest_verts_m"], dtype=np.float32)
            expr_basis = None
        else:
            joints8, weights8, max_inf = compact_skin_to_n(
                rs["lbs_skin_indices"], rs["lbs_vert_indices"],
                rs["lbs_skin_weights"], int(rs["num_verts"]), max_inf=8,
            )
            per_y_down = True
            rest_override = None
            expr_basis = rs["expr_basis"] if int(rs["num_expr"]) > 0 else None
        return cls(
            parents=np.asarray(rs["parents"], dtype=np.int32),
            joint_offsets_cm=np.asarray(rs["joint_translation_offsets"], dtype=np.float32),
            bind_global_cm=np.asarray(bind_skel_state(model, pose_data), dtype=np.float32),
            lbs_joints=joints8, lbs_weights=weights8, lbs_max_inf=max_inf,
            faces=np.asarray(rs["faces"], dtype=np.uint32),
            num_joints=int(rs["num_joints"]), num_verts=int(rs["num_verts"]),
            num_expr=int(rs["num_expr"]),
            per_frame_y_down=per_y_down, can_rerun_fk=not external,
            expr_basis=expr_basis, _model=model, _rest_override=rest_override,
        )


def ibp_from_bind_global(bind_skel_state_m: np.ndarray) -> np.ndarray:
    """Inverse-bind MAT4 from the rig's bind global (meters). IBP[j] =
    inverse(FK over bind local TRS), as glTF skinning needs. Returns (NJ, 4, 4)
    column-major."""
    NJ = bind_skel_state_m.shape[0]
    t = bind_skel_state_m[:, :3].astype(np.float32)
    q = bind_skel_state_m[:, 3:7].astype(np.float32)
    s = bind_skel_state_m[:, 7].astype(np.float32)
    # Forward bind M = T * R * S (uniform scale): [s*R | t; 0 | 1]
    R = _quat_to_mat3_np(q)
    M = np.zeros((NJ, 4, 4), dtype=np.float32)
    M[:, :3, :3] = R * s[:, None, None]
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    # fp64 4x4 invert per joint for stability, back to fp32.
    M_inv = np.linalg.inv(M.astype(np.float64)).astype(np.float32)
    # glTF MAT4 accessor is column-major.
    return M_inv.transpose(0, 2, 1).astype(np.float32)


def _local_trs_per_frame(
    rig_static: Dict[str, np.ndarray], mhr_model_params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (local_t[N, 127, 3], local_q[N, 127, 4 xyzw], local_s[N, 127])
    in rig-native frame, meters. Mirrors mhr_rig.forward without skinning."""
    pt = rig_static["param_transform"]            # (889, 249) = (127*7, 204+45)
    t_off = rig_static["joint_translation_offsets"]  # (127, 3)  cm
    q_pre = rig_static["joint_prerotations"]      # (127, 4)
    NJ = rig_static["num_joints"]

    N = mhr_model_params.shape[0]
    cat_in = np.zeros((N, pt.shape[1]), dtype=np.float32)
    cat_in[:, :mhr_model_params.shape[1]] = mhr_model_params.astype(np.float32)
    # joint_parameters[n, d] = sum_i pt[d, i] * cat_in[n, i]
    jp = cat_in @ pt.T
    jp = jp.reshape(N, NJ, 7)

    local_t_cm = jp[..., :3] + t_off[None]
    local_q_raw = _euler_xyz_to_quat_np(jp[..., 3:6])
    local_q = _quat_multiply_np(q_pre[None], local_q_raw)
    local_s = np.exp(jp[..., 6] * _LN2)

    # rig-cm -> glTF-meters
    return (local_t_cm * 0.01).astype(np.float32), local_q.astype(np.float32), local_s.astype(np.float32)


def _ibp_to_mat4(ibp_skel: np.ndarray) -> np.ndarray:
    """(127, 8) IBP skel-state -> (127, 4, 4) column-major MAT4, t in meters."""
    NJ = ibp_skel.shape[0]
    t = ibp_skel[:, :3] * 0.01      # cm -> m
    q = ibp_skel[:, 3:7]
    s = ibp_skel[:, 7]
    R = _quat_to_mat3_np(q)
    M = np.zeros((NJ, 4, 4), dtype=np.float32)
    M[:, :3, :3] = R * s[:, None, None]
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    return M.transpose(0, 2, 1).astype(np.float32)


def uv_sphere_unit(n_lat: int = 9, n_lon: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Unit UV sphere, poles ±Y. `n_lat` odd so a ring lands at the equator;
    n_lon=16 matches the capsule cylinder so end rings meet flush."""
    verts: List[List[float]] = [[0.0, -1.0, 0.0]]  # south pole at index 0
    for i in range(1, n_lat + 1):
        lat = -0.5 * np.pi + np.pi * i / (n_lat + 1)
        y = float(np.sin(lat))
        r = float(np.cos(lat))
        for k in range(n_lon):
            phi = 2.0 * np.pi * k / n_lon
            verts.append([r * float(np.cos(phi)), y, r * float(np.sin(phi))])
    north_idx = len(verts)
    verts.append([0.0, 1.0, 0.0])

    faces: List[List[int]] = []
    # South cap — winding gives -Y outward normal.
    south_ring = 1
    for k in range(n_lon):
        a = south_ring + k
        b = south_ring + (k + 1) % n_lon
        faces.append([0, a, b])
    # Inter-ring quads, outward radial.
    for i in range(n_lat - 1):
        rl = 1 + i * n_lon
        rh = 1 + (i + 1) * n_lon
        for k in range(n_lon):
            a = rl + k
            b = rl + (k + 1) % n_lon
            c = rh + (k + 1) % n_lon
            d = rh + k
            faces.append([a, c, b])
            faces.append([a, d, c])
    # North cap — winding gives +Y outward normal.
    rL = 1 + (n_lat - 1) * n_lon
    for k in range(n_lon):
        a = rL + k
        b = rL + (k + 1) % n_lon
        faces.append([north_idx, b, a])

    return (np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32))


def flat_shade_mesh(
    verts: np.ndarray, faces: np.ndarray, joints: np.ndarray, weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat-shade by duplicating verts per face; each triangle gets 3 unique
    verts sharing its face normal. Skinning attrs duplicated alongside."""
    F = faces.shape[0]
    new_v = np.zeros((F * 3, 3), dtype=np.float32)
    new_n = np.zeros((F * 3, 3), dtype=np.float32)
    new_j = np.zeros((F * 3, 4), dtype=np.uint16)
    new_w = np.zeros((F * 3, 4), dtype=np.float32)
    new_f = np.arange(F * 3, dtype=np.uint32).reshape(F, 3)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.where(fn_len > 1e-8, fn / np.maximum(fn_len, 1e-12), np.array([[0.0, 1.0, 0.0]]))
    for k in range(3):
        new_v[k::3] = verts[faces[:, k]]
        new_n[k::3] = fn
        new_j[k::3] = joints[faces[:, k]]
        new_w[k::3] = weights[faces[:, k]]
    return new_v, new_n, new_f, new_j, new_w


def smooth_shade_mesh(
    verts: np.ndarray, faces: np.ndarray, joints: np.ndarray, weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted per-vertex normals. Geometry/skinning/indexing pass through
    unchanged so vertex colors stay aligned. Orphan verts get +Y fallback."""
    Nv = int(verts.shape[0])
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    vn = np.zeros((Nv, 3), dtype=np.float32)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(ln > 1e-8, vn / np.maximum(ln, 1e-12), np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
    return (
        verts.astype(np.float32),
        vn.astype(np.float32),
        faces.astype(np.uint32),
        joints,
        weights,
    )


def rotation_align(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit `from_vec` to unit `to_vec`."""
    cos_t = float(np.dot(from_vec, to_vec))
    cross = np.cross(from_vec, to_vec)
    sin_t = float(np.linalg.norm(cross))
    if sin_t < 1e-8:
        if cos_t > 0:
            return np.eye(3, dtype=np.float32)
        # Anti-aligned: 180° around any perpendicular. For ≈+Y, use X.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    axis = cross / sin_t
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + sin_t * K + (1.0 - cos_t) * (K @ K)).astype(np.float32)


def make_lit_material(
    roughness: float = 0.85, double_sided: bool = False, opacity: float = 1.0,
) -> dict:
    """Lit PBR material using vertex COLOR_0. Dielectric (metallic=0) so colors
    stay readable; roughness 0.85 suits rainbow body meshes, 0.3 the glossy
    SCAIL rig. opacity < 1 switches to alpha-blend."""
    a = float(max(0.0, min(1.0, opacity)))
    mat = {
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, a],
            "metallicFactor": 0.0,
            "roughnessFactor": float(max(0.0, min(1.0, roughness))),
        },
    }
    if a < 1.0:
        mat["alphaMode"] = "BLEND"
    if double_sided:
        mat["doubleSided"] = True
    return mat


# OpenPose 18-keypoint viz (independent of MHR rig — uses pred_keypoints_3d,
# the model's regressed surface keypoints).


OPENPOSE_18_NAMES = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar",
)

# COCO-18 OpenPose -> MHR70. Subset of `MHR70_TO_OPENPOSE` in
# comfy/ldm/sam3d/mhr70.py (no toes/heels).
OPENPOSE18_TO_MHR70 = np.array([
    0,   # 0  Nose
    69,  # 1  Neck
    6,   # 2  RShoulder
    8,   # 3  RElbow
    41,  # 4  RWrist
    5,   # 5  LShoulder
    7,   # 6  LElbow
    62,  # 7  LWrist
    10,  # 8  RHip
    12,  # 9  RKnee
    14,  # 10 RAnkle
    9,   # 11 LHip
    11,  # 12 LKnee
    13,  # 13 LAnkle
    2,   # 14 REye
    1,   # 15 LEye
    4,   # 16 REar
    3,   # 17 LEar
], dtype=np.int64)

# OpenPose limb pairs + rainbow palette delegate to the canonical DWPose tables
# carried by `comfy_extras.pose.keypoint_draw.KeypointDraw` (also used by nodes_sdpose).
# `body_limbSeq` is 1-indexed there; we use 0-indexed throughout this module.
from comfy_extras.pose.keypoint_draw import KeypointDraw as _KeypointDraw
_KD = _KeypointDraw()
OPENPOSE_18_PAIRS = tuple((a - 1, b - 1) for a, b in _KD.body_limbSeq)
OPENPOSE_RAINBOW_18 = (np.array(_KD.colors, dtype=np.float32) / 255.0)


# SCAIL-Pose limb palette (17 limbs in `OPENPOSE_18_PAIRS` order): warm =
# right side, cool = left, grey centerline, pink/violet face. Matches
# ComfyUI-SCAIL-Pose's `nlf_render.py::ordered_colors_255`.
SCAIL_LIMB_COLORS_17 = (np.array([
    [255,   0,   0],  # 0  Neck → R.Shoulder       (Red)
    [  0, 255, 255],  # 1  Neck → L.Shoulder       (Cyan)
    [255,  85,   0],  # 2  R.Shoulder → R.Elbow    (Orange)
    [255, 170,   0],  # 3  R.Elbow → R.Wrist       (Golden Orange)
    [  0, 170, 255],  # 4  L.Shoulder → L.Elbow    (Sky Blue)
    [  0,  85, 255],  # 5  L.Elbow → L.Wrist       (Medium Blue)
    [180, 255,   0],  # 6  Neck → R.Hip            (Yellow-Green)
    [  0, 255,   0],  # 7  R.Hip → R.Knee          (Bright Green)
    [  0, 255,  85],  # 8  R.Knee → R.Ankle        (Light Green-Blue)
    [  0,   0, 255],  # 9  Neck → L.Hip            (Pure Blue)
    [ 85,   0, 255],  # 10 L.Hip → L.Knee          (Purple-Blue)
    [170,   0, 255],  # 11 L.Knee → L.Ankle        (Medium Purple)
    [150, 150, 150],  # 12 Neck → Nose             (Grey)
    [255,   0, 170],  # 13 Nose → R.Eye            (Pink-Magenta)
    [ 50,   0, 255],  # 14 R.Eye → R.Ear           (Dark Violet)
    [255,   0, 170],  # 15 Nose → L.Eye            (Pink-Magenta)
    [ 50,   0, 255],  # 16 L.Eye → L.Ear           (Dark Violet)
], dtype=np.float32) / 255.0)


def _scail_keypoint_colors_18(limb_pairs: Tuple[Tuple[int, int], ...] = None) -> np.ndarray:
    """18 keypoint colors derived from 17 SCAIL limb colors. Each kp inherits
    the first limb where it's the distal endpoint; mid-grey otherwise (only
    the neck/nose root in OpenPose-18)."""
    pairs = limb_pairs if limb_pairs is not None else OPENPOSE_18_PAIRS
    out = np.tile(np.array([0.6, 0.6, 0.6], dtype=np.float32), (18, 1))
    for limb_i, (_, b) in enumerate(pairs):
        if (out[b] == 0.6).all():
            out[b] = SCAIL_LIMB_COLORS_17[limb_i]
    return out


SCAIL_KEYPOINT_COLORS_18 = _scail_keypoint_colors_18()


# OpenPose hand: 21 kp per hand = wrist + 5 fingers × 4 joints (proximal→distal).
# MHR70 stores fingers as (tip, joint1, joint2, joint3=MCP) so we reverse each
# 4-tuple. See comfy/ldm/sam3d/mhr70.py.
OPENPOSE_HAND21_NAMES = (
    "wrist",
    "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4",
)

OPENPOSE_HAND21_TO_MHR70_R = np.array([
    41,                  # 0  right_wrist
    24, 23, 22, 21,      # thumb base→tip
    28, 27, 26, 25,      # index
    32, 31, 30, 29,      # middle
    36, 35, 34, 33,      # ring
    40, 39, 38, 37,      # pinky
], dtype=np.int64)

OPENPOSE_HAND21_TO_MHR70_L = np.array([
    62,                  # 0  left_wrist
    45, 44, 43, 42,      # thumb base→tip
    49, 48, 47, 46,      # index
    53, 52, 51, 50,      # middle
    57, 56, 55, 54,      # ring
    61, 60, 59, 58,      # pinky
], dtype=np.int64)

# OpenPose hand limbs: 5 chains × 4 bones, delegated to KeypointDraw.hand_edges.
OPENPOSE_HAND_PAIRS = tuple(tuple(e) for e in _KD.hand_edges)

# OpenPose hand colors (poseParameters.cpp::HAND_COLORS_RENDER): wrist grey,
# then per-finger base→tip gradient red/yellow/green/cyan/magenta.
OPENPOSE_HAND_COLORS_21 = (np.array([
    [100, 100, 100],
    [100,   0,   0], [150,   0,   0], [200,   0,   0], [255,   0,   0],
    [100, 100,   0], [150, 150,   0], [200, 200,   0], [255, 255,   0],
    [  0, 100,  50], [  0, 150,  75], [  0, 200, 100], [  0, 255, 125],
    [  0, 100, 100], [  0, 150, 150], [  0, 200, 200], [  0, 255, 255],
    [100,   0, 100], [150,   0, 150], [200,   0, 200], [255,   0, 255],
], dtype=np.float32) / 255.0)

# DWPose: solid blue hand dots, rainbow per-finger bones (matches
# controlnet_aux/dwpose/util.py::draw_handpose).
DWPOSE_HAND_COLORS_21 = np.tile(
    np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (21, 1)
)


def resolve_openpose_keypoints_from_joints(
    joints: np.ndarray, mapping: np.ndarray, weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(K, 2) joint-index map resolved against (J, D) joint positions -> (K, D).
    Row (a, b): b == -1 uses joints[a]; b >= 0 returns w*joints[a]+(1-w)*joints[b]
    (w defaults 0.5 = midpoint; w outside [0, 1] extrapolates past the segment)."""
    a = mapping[:, 0].astype(np.int64)
    b = mapping[:, 1].astype(np.int64)
    pos_a = joints[a]
    has_b = b >= 0
    if not has_b.any():
        return pos_a.astype(np.float32, copy=False)
    b_safe = np.where(has_b, b, a)
    pos_b = joints[b_safe]
    if weights is None:
        w_a = np.where(has_b, 0.5, 1.0).astype(np.float32)
    else:
        w_a = np.where(has_b, np.asarray(weights, dtype=np.float32), 1.0)
    w_b = (1.0 - w_a).astype(np.float32)
    out = pos_a * w_a[:, None] + pos_b * w_b[:, None]
    return out.astype(np.float32, copy=False)


# part -> (override map key, override weight key, MHR70 reindex map)
_OPENPOSE_RENDER_MAPS = {
    "body": ("openpose18_joint_indices", "openpose18_joint_weights", OPENPOSE18_TO_MHR70),
    "hand_r": ("openpose_hand21_r_joint_indices", "openpose_hand21_r_joint_weights", OPENPOSE_HAND21_TO_MHR70_R),
    "hand_l": ("openpose_hand21_l_joint_indices", "openpose_hand21_l_joint_weights", OPENPOSE_HAND21_TO_MHR70_L),
}


def openpose_render_keypoints(
    person: Dict[str, Any], pose_data: Optional[Dict[str, Any]], part: str,
    *, dim: int, H: int = 0, W: int = 0,
) -> Optional[np.ndarray]:
    """OpenPose keypoints for one person, op-layout, camera frame (Y-down).
    `part` in {'body','hand_r','hand_l'}. dim=3 -> (K, 3) metres pre-cam_t-add;
    dim=2 -> (K, 2) pixels. Returns None when source data is missing.

    External rigs resolve from `pred_joint_coords` (Y-up -> flipped to Y-down);
    MHR reindexes stored `pred_keypoints_{3d,2d}` via the MHR70 map."""
    map_key, w_key, mhr_map = _OPENPOSE_RENDER_MAPS[part]
    override = _get_skeleton_override(pose_data)
    ext_map = override.get(map_key) if override is not None else None

    if ext_map is not None:
        joints = person.get("pred_joint_coords")
        if joints is None:
            return None
        w = override.get(w_key)
        kp3d = resolve_openpose_keypoints_from_joints(
            np.asarray(joints, dtype=np.float32),
            np.asarray(ext_map, dtype=np.int64),
            None if w is None else np.asarray(w, dtype=np.float32),
        ).copy()
        kp3d[:, 1] *= -1.0   # rig-native Y-up -> camera Y-down
        kp3d[:, 2] *= -1.0
        if dim == 3:
            return kp3d
        cam_t = person.get("pred_cam_t")
        focal = person.get("focal_length")
        if cam_t is None or focal is None:
            return None
        pts3 = kp3d + np.asarray(cam_t, dtype=np.float32).reshape(1, 3)
        z = np.maximum(pts3[:, 2:3], 1e-6)
        f = float(np.asarray(focal, dtype=np.float32).reshape(-1)[0])
        xy = pts3[:, :2] * f + np.array([W * 0.5, H * 0.5], dtype=np.float32)[None, :] * z
        return (xy / z).astype(np.float32)

    key = "pred_keypoints_3d" if dim == 3 else "pred_keypoints_2d"
    kp_full = person.get(key)
    if kp_full is None:
        return None
    kp_full = np.asarray(kp_full, dtype=np.float32)
    if kp_full.ndim != 2 or kp_full.shape[0] < 70:
        return None
    return kp_full[mhr_map]


# Face landmarks (face_source="rig"). MHR has no face bones, so landmarks are
# sourced from `pred_vertices` at vertex IDs picked by NN against the target xyz
# below. Tweak targets if landmarks land off-surface.

# (name, target_xyz) in MHR canonical Y-up meters.
FACE_LANDMARK_TARGETS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    # Brows — 3 per side, outer→inner
    ("r_brow_outer", (-0.058, 1.690, 0.090)),
    ("r_brow_mid",   (-0.040, 1.695, 0.105)),
    ("r_brow_inner", (-0.020, 1.692, 0.115)),
    ("l_brow_inner", (+0.020, 1.692, 0.115)),
    ("l_brow_mid",   (+0.040, 1.695, 0.105)),
    ("l_brow_outer", (+0.058, 1.690, 0.090)),
    # Right eye — outer/top/inner/bottom
    ("r_eye_outer", (-0.058, 1.660, 0.085)),
    ("r_eye_top",   (-0.040, 1.673, 0.090)),
    ("r_eye_inner", (-0.022, 1.665, 0.092)),
    ("r_eye_bot",   (-0.040, 1.652, 0.090)),
    # Left eye
    ("l_eye_outer", (+0.058, 1.660, 0.085)),
    ("l_eye_top",   (+0.040, 1.673, 0.090)),
    ("l_eye_inner", (+0.022, 1.665, 0.092)),
    ("l_eye_bot",   (+0.040, 1.652, 0.090)),
    # Nose
    ("nose_bridge", (0.000, 1.660, 0.110)),
    ("nose_mid",    (0.000, 1.620, 0.125)),
    ("nose_tip",    (0.000, 1.585, 0.135)),
    ("nostril_r",   (-0.014, 1.580, 0.115)),
    ("nostril_l",   (+0.014, 1.580, 0.115)),
    # Mouth — 4 outer-lip points
    ("mouth_r_corner", (-0.030, 1.540, 0.105)),
    ("upper_lip_mid",  (+0.000, 1.555, 0.115)),
    ("mouth_l_corner", (+0.030, 1.540, 0.105)),
    ("lower_lip_mid",  (+0.000, 1.530, 0.110)),
    # Chin + jaw line — Y raised so NN search lands on chin tip / jaw underside
    # (above the jaw-neck boundary at y~1.47) instead of throat verts.
    ("chin",       (0.000, 1.498, 0.108)),
    ("r_jaw_low",  (-0.038, 1.512, 0.100)),
    ("r_jaw_mid",  (-0.062, 1.535, 0.080)),
    ("r_jaw_high", (-0.078, 1.562, 0.060)),
    ("l_jaw_low",  (+0.038, 1.512, 0.100)),
    ("l_jaw_mid",  (+0.062, 1.535, 0.080)),
    ("l_jaw_high", (+0.078, 1.562, 0.060)),
)

# Solid white face landmarks — matches DWPose, reads cleanly against the
# rainbow body palette.
def _face_landmark_colors() -> np.ndarray:
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.tile(white, (len(FACE_LANDMARK_TARGETS), 1))


FACE_LANDMARK_COLORS: np.ndarray = _face_landmark_colors()


def select_face_landmark_vert_ids(
    canonical_positions: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick MHR head vertex IDs for each `FACE_LANDMARK_TARGETS` by NN in
    canonical positions, restricted to `face_mask` verts (expression-deforming)
    when available, else a position bbox (less reliable around the chin/jaw)."""
    P = np.asarray(canonical_positions, dtype=np.float32).reshape(-1, 3)
    if face_mask is not None and np.asarray(face_mask).any():
        valid = np.where(np.asarray(face_mask).reshape(-1))[0]
    else:
        head_mask = (P[:, 1] > 1.47) & (np.abs(P[:, 0]) < 0.11) & (P[:, 2] > 0.04)
        valid = np.where(head_mask)[0]
    if valid.size == 0:
        raise ValueError(
            "select_face_landmark_vert_ids: no head verts matched the "
            "canonical filter — check that pose_data.canonical_colors "
            "holds the MHR rest-pose positions / face_mask."
        )
    P_valid = P[valid]
    out = np.empty(len(FACE_LANDMARK_TARGETS), dtype=np.int64)
    for i, (_, xyz) in enumerate(FACE_LANDMARK_TARGETS):
        target = np.asarray(xyz, dtype=np.float32)
        d2 = np.sum((P_valid - target) ** 2, axis=1)
        out[i] = int(valid[int(d2.argmin())])
    return out
