"""Face expression for SAM 3D Body.

Pipeline: comfy_extras.mediapipe.face_landmarker → 52 ARKit blendshapes →
72-dim MHR expr_params (mapping inlined below).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import comfy.model_management


# Bypass deadzone — jaw signals are clean (open or not)
_NOISE_FREE_BLENDSHAPES = {"jawOpen", "jawForward", "jawLeft", "jawRight"}

# Per-region gain — MP magnitudes vary by family (jaw up to 1.0, eye/brow
# rarely past 0.3), so a single global gain over/underdrives.
_REGION_PREFIXES = {
    "mouth": ("jaw", "mouth"),
    "eye":   ("eye",),
    "brow":  ("brow", "cheek", "nose"),  # cheek/nose read as upper-face
}


def _region_of(arkit_name: str) -> str:
    for region, prefixes in _REGION_PREFIXES.items():
        for p in prefixes:
            if arkit_name.startswith(p):
                return region
    return "other"


# MHR axis → ARKit driver(s). Each axis collects 1-3 (name, weight) entries;
# the consumer takes max() across them so primary + aux contributions don't
# stack. MHR's 72 expression axes ship as anonymous `shape_c_N` channels in
# the upstream FBX (no semantic names), so this table is hand-derived by
# visual inspection of which axis each ARKit shape drives. Axes 2/3 and
# 12/13 are filled by aux routes only. ARKit shapes with no MHR analog are simply absent.
_AXIS_TO_ARKIT: Dict[int, List[Tuple[str, float]]] = {
    0:  [("browDownLeft",        1.0)],
    1:  [("browDownRight",       1.0)],
    2:  [("cheekPuff",           1.0)],
    3:  [("cheekPuff",           1.0)],
    4:  [("cheekSquintLeft",     1.0)],
    5:  [("cheekSquintRight",    1.0)],
    6:  [("mouthStretchLeft",    1.0)],
    7:  [("mouthStretchRight",   1.0)],
    8:  [("mouthShrugLower",     1.0)],
    9:  [("mouthShrugUpper",     1.0)],
    10: [("mouthDimpleLeft",     1.0)],
    11: [("mouthDimpleRight",    1.0)],
    12: [("eyeLookDownLeft",     0.3)],
    13: [("eyeLookDownRight",    0.3)],
    14: [("eyeBlinkLeft",        1.0)],
    15: [("eyeBlinkRight",       1.0)],
    16: [("eyeLookOutLeft",      1.0)],
    17: [("eyeLookInRight",      1.0)],
    18: [("eyeLookInLeft",       1.0)],
    19: [("eyeLookOutRight",     1.0)],
    22: [("eyeLookUpLeft",       1.0), ("browInnerUp", 0.5)],
    23: [("eyeLookUpRight",      1.0), ("browInnerUp", 0.5)],
    24: [("jawOpen",             1.0), ("mouthLowerDownLeft", 0.3), ("mouthLowerDownRight", 0.3)],
    25: [("jawLeft",             1.0)],
    26: [("jawRight",            1.0)],
    27: [("jawForward",          1.0)],
    28: [("eyeSquintLeft",       1.0)],
    29: [("eyeSquintRight",      1.0)],
    32: [("mouthSmileLeft",      1.0)],
    33: [("mouthSmileRight",     1.0)],
    40: [("mouthLeft",           1.0)],
    41: [("mouthRight",          1.0)],
    42: [("mouthFrownLeft",      1.0)],
    43: [("mouthFrownRight",     1.0)],
    54: [("mouthLowerDownLeft",  1.0)],
    55: [("mouthLowerDownRight", 1.0)],
    60: [("noseSneerLeft",       1.0)],
    61: [("noseSneerRight",      1.0)],
    66: [("browOuterUpLeft",     1.0)],
    67: [("browOuterUpRight",    1.0)],
    68: [("eyeWideLeft",         1.0)],
    69: [("eyeWideRight",        1.0)],
    70: [("mouthUpperUpLeft",    1.0)],
    71: [("mouthUpperUpRight",   1.0)],
}


def _deadzone(x: float, threshold: float) -> float:
    """Zero below threshold, linearly remap (threshold..1] → (0..1] so
    amplification doesn't promote MP's per-blendshape noise floor."""
    if threshold <= 0.0:
        return x
    if x <= threshold:
        return 0.0
    return (x - threshold) / (1.0 - threshold)


def arkit_to_expr_params(
    blendshape_coefs: Dict[str, float],
    strength: float = 1.0,
    mouth_strength: float = 1.0,
    eye_strength: float = 1.0,
    brow_strength: float = 1.0,
    input_threshold: float = 0.0,
    n_axes: int = 72,
) -> np.ndarray:
    """Map MediaPipe's 52 ARKit blendshapes to MHR's 72 expr_params axes.
    Multiple ARKit names per axis combine via max() so primary + aux routes
    don't double up."""
    expr = np.zeros(n_axes, dtype=np.float32)
    region_scale = {
        "mouth": float(mouth_strength), "eye": float(eye_strength),
        "brow":  float(brow_strength),  "other": 1.0,
    }
    thr = float(input_threshold)
    for axis, routes in _AXIS_TO_ARKIT.items():
        best = 0.0
        for name, weight in routes:
            raw = float(blendshape_coefs.get(name, 0.0))
            name_thr = 0.0 if name in _NOISE_FREE_BLENDSHAPES else thr
            raw = _deadzone(raw, name_thr)
            c = raw * region_scale[_region_of(name)] * float(weight)
            if c > best:
                best = c
        expr[axis] = best * strength
    return expr


def subtract_per_clip_baseline(
    per_frame_coefs: List[Optional[Dict[str, float]]],
    percentile: float = 5.0,
) -> List[Optional[Dict[str, float]]]:
    """Subtract per-blendshape p`percentile` baseline, clamp at 0. Adapts to
    per-subject MP bias (e.g. resting browOuterUp ~0.15 → permanent surprise
    under brow_strength=2.0) that a global deadzone can't catch."""
    if percentile <= 0.0:
        return list(per_frame_coefs)

    names: set = set()
    for c in per_frame_coefs:
        if c is not None:
            names.update(c.keys())

    baselines: Dict[str, float] = {}
    for n in names:
        vals = [c[n] for c in per_frame_coefs if c is not None and n in c]
        if vals:
            baselines[n] = float(np.percentile(vals, percentile))

    return [
        None if c is None
        else {n: max(0.0, float(v) - baselines.get(n, 0.0)) for n, v in c.items()}
        for c in per_frame_coefs
    ]


def smooth_blendshape_series(
    per_frame_coefs: List[Optional[Dict[str, float]]],
    window: int = 7,
    sigma: Optional[float] = None,
) -> List[Optional[Dict[str, float]]]:
    """Gaussian-smooth each coefficient across time. MP per-frame output swings
    30-70% on static faces; smoothing pre-mapping cleans better than smoothing
    mesh verts. None frames pass through unchanged."""
    if window <= 1:
        return list(per_frame_coefs)
    if window % 2 == 0:
        window += 1
    if sigma is None:
        sigma = max(1.0, window / 5.0)

    x = np.arange(window) - (window - 1) / 2.0
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()

    names: set = set()
    for c in per_frame_coefs:
        if c is not None:
            names.update(c.keys())
    if not names:
        return list(per_frame_coefs)

    N = len(per_frame_coefs)
    pad = window // 2
    out: List[Optional[Dict[str, float]]] = [None] * N

    for name in names:
        series = np.zeros(N, dtype=np.float32)
        mask = np.zeros(N, dtype=bool)
        for i, c in enumerate(per_frame_coefs):
            if c is not None:
                series[i] = float(c.get(name, 0.0))
                mask[i] = True
        if not mask.any():
            continue
        if not mask.all():
            idx = np.arange(N)
            series = np.interp(idx, idx[mask], series[mask])
        padded = np.concatenate(
            [np.repeat(series[:1], pad), series, np.repeat(series[-1:], pad)]
        )
        filt = np.zeros_like(series)
        for i, w in enumerate(k):
            filt += w * padded[i: i + N]
        for i in range(N):
            if per_frame_coefs[i] is None:
                continue
            if out[i] is None:
                out[i] = {}
            out[i][name] = float(filt[i])
    return out


def fill_detection_gaps(
    per_frame_coefs: List[Optional[Dict[str, float]]],
    method: str = "interpolate",
    max_gap: int = 12,
) -> List[Optional[Dict[str, float]]]:
    """Fill missing per-frame dicts so the signal doesn't slam to zero at
    undetected frames. method: 'interpolate' | 'hold' | 'zeros'.

    `max_gap` applies to 'interpolate' and 'hold' — gaps longer than that stay
    None (don't fake too far). 'zeros' ignores `max_gap` on purpose: the goal
    there is to relax to neutral on every miss, no matter how long, otherwise
    long undetected runs would inherit Predict's per-frame expression."""
    if method == "zeros":
        names: set = set()
        for c in per_frame_coefs:
            if c is not None:
                names.update(c.keys())
        zero = {n: 0.0 for n in names}
        return [dict(zero) if c is None else c for c in per_frame_coefs]

    N = len(per_frame_coefs)
    detected = [i for i, c in enumerate(per_frame_coefs) if c is not None]
    if not detected:
        return list(per_frame_coefs)

    out: List[Optional[Dict[str, float]]] = list(per_frame_coefs)

    for fi in range(N):
        if out[fi] is not None:
            continue
        prev_i = next((k for k in range(fi - 1, -1, -1) if per_frame_coefs[k] is not None), None)
        next_i = next((k for k in range(fi + 1, N) if per_frame_coefs[k] is not None), None)
        if prev_i is None and next_i is None:
            continue
        max_dist = max(
            (fi - prev_i) if prev_i is not None else 10**9,
            (next_i - fi) if next_i is not None else 10**9,
        )
        if max_dist > max_gap:
            continue

        if method == "hold":
            src = per_frame_coefs[prev_i] if prev_i is not None else per_frame_coefs[next_i]
            out[fi] = dict(src)
        elif method == "interpolate":
            if prev_i is None:
                out[fi] = dict(per_frame_coefs[next_i])
            elif next_i is None:
                out[fi] = dict(per_frame_coefs[prev_i])
            else:
                w = (fi - prev_i) / (next_i - prev_i)
                a = per_frame_coefs[prev_i]
                b = per_frame_coefs[next_i]
                keys = set(a.keys()) | set(b.keys())
                out[fi] = {k: (1.0 - w) * a.get(k, 0.0) + w * b.get(k, 0.0) for k in keys}
    return out


def detect_faces_in_crop(
    inner, image_rgb_uint8: np.ndarray, crop_xyxy: np.ndarray, num_faces: int = 1,
) -> List[dict]:
    """Run detection on a sub-region; remap bbox+landmarks back to full-image
    coords. Helps small/distant faces that fall below BlazeFace's min size."""
    H, W = image_rgb_uint8.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in crop_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    crop = np.ascontiguousarray(image_rgb_uint8[y1:y2, x1:x2])
    faces = inner.face_landmarker.detect_batch([crop], num_faces=num_faces)[0]
    bbox_off = np.array([x1, y1, x1, y1], dtype=np.float32)
    xy_off = np.array([x1, y1], dtype=np.float32)
    for f in faces:
        f["bbox_xyxy"] = f["bbox_xyxy"] + bbox_off
        f["landmarks_xy"] = f["landmarks_xy"] + xy_off
    return faces


# Crop helpers — feed MP a tight head region so it doesn't downsample the face
# to 192px for full-frame detection.

def _expand_bbox(bbox_xyxy: np.ndarray, factor: float, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * (x2 - x1) * factor, 0.5 * (y2 - y1) * factor
    return np.array([
        max(0.0, cx - hw), max(0.0, cy - hh),
        min(float(W), cx + hw), min(float(H), cy + hh),
    ], dtype=np.float32)


def head_region_crop(
    person_bbox: np.ndarray, expand: float, W: int, H: int, head_h_frac: float = 0.4,
) -> np.ndarray:
    """Crop upper `head_h_frac` of a body bbox — cropping the whole body wastes
    BlazeFace's 128² input on body pixels."""
    x1, y1, x2, y2 = (float(v) for v in person_bbox)
    body_h = y2 - y1
    if body_h <= 0 or x2 - x1 <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return _expand_bbox(np.array([x1, y1, x2, y1 + body_h * head_h_frac]), expand, W, H)


# mhr70 convention: first five kp are COCO-style face landmarks in pixel coords.
_FACE_KP_INDICES = (0, 1, 2, 3, 4)  # nose, L-eye, R-eye, L-ear, R-ear


def head_crop_from_keypoints(
    pred_keypoints_2d: np.ndarray, expand: float, W: int, H: int,
) -> Optional[np.ndarray]:
    """Head crop from SAM3D nose/eyes/ears kp. HEAD_FIT pads forehead/chin
    since these only span the central face. None if <2 kp in-frame."""
    if pred_keypoints_2d is None:
        return None
    kp = np.asarray(pred_keypoints_2d, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] <= max(_FACE_KP_INDICES):
        return None
    face = kp[list(_FACE_KP_INDICES), :2]
    in_frame = (face[:, 0] > 0) & (face[:, 1] > 0) & (face[:, 0] < W) & (face[:, 1] < H)
    valid = face[in_frame]
    if len(valid) < 2:
        return None
    x1, x2 = float(valid[:, 0].min()), float(valid[:, 0].max())
    y1, y2 = float(valid[:, 1].min()), float(valid[:, 1].max())
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    span = max(x2 - x1, y2 - y1, 1.0)
    half = 0.5 * span * 1.8 * float(expand)  # 1.8 = pad forehead+chin
    return np.array([
        max(0.0, cx - half), max(0.0, cy - half),
        min(float(W), cx + half), min(float(H), cy + half),
    ], dtype=np.float32)


# Face → person assignment when running full-frame detection.

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aw, ah = max(0.0, a[2] - a[0]), max(0.0, a[3] - a[1])
    bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def assign_faces_to_persons(
    face_bboxes: List[np.ndarray], person_bboxes: List[np.ndarray], min_iou: float = 0.01,
) -> List[Optional[int]]:
    if not face_bboxes or not person_bboxes:
        return [None] * len(person_bboxes)
    assigned: List[Optional[int]] = [None] * len(person_bboxes)
    used: set = set()
    # Larger persons first — bigger bbox correlates with detectable face.
    order = sorted(range(len(person_bboxes)),
                   key=lambda p: -((person_bboxes[p][2] - person_bboxes[p][0])
                                   * (person_bboxes[p][3] - person_bboxes[p][1])))
    for pi in order:
        best_iou = min_iou
        best_fi = None
        pb = person_bboxes[pi]
        for fi, fb in enumerate(face_bboxes):
            if fi in used:
                continue
            cx, cy = 0.5 * (fb[0] + fb[2]), 0.5 * (fb[1] + fb[3])
            inside = (pb[0] <= cx <= pb[2]) and (pb[1] <= cy <= pb[3])
            score = max(_iou_xyxy(fb, pb), 0.5 if inside else 0.0)
            if score > best_iou:
                best_iou = score
                best_fi = fi
        if best_fi is not None:
            assigned[pi] = best_fi
            used.add(best_fi)
    return assigned


# Re-run MHR forward after writing expr_params back into pose_frames; updates
# pred_vertices / pred_keypoints_2d/3d / pred_joint_coords / pred_global_rots.

def regenerate_mesh_from_params(inner, pose_frames: List[List[Dict[str, Any]]]) -> None:
    """Re-run MHR forward and write verts/kp3d/kp2d/joint back in place.
    Drives MHR via euler params directly because hand refinement zeroes
    pred_pose_raw."""
    device = comfy.model_management.get_torch_device()
    head = inner.head_pose

    B = len(pose_frames)
    max_p = max((len(f) for f in pose_frames), default=0)

    for pid in range(max_p):
        grots, bpps, hands, shapes, scales, exprs, cam_ts, fls = [], [], [], [], [], [], [], []
        present: List[bool] = []
        for fi in range(B):
            if pid >= len(pose_frames[fi]):
                present.append(False)
                continue
            p = pose_frames[fi][pid]
            needed = ("global_rot", "body_pose_params", "hand_pose_params",
                      "shape_params", "scale_params", "expr_params",
                      "pred_cam_t", "focal_length")
            if any(p.get(k) is None for k in needed):
                present.append(False)
                continue
            grots.append(np.asarray(p["global_rot"], dtype=np.float32))
            bpps.append(np.asarray(p["body_pose_params"], dtype=np.float32))
            hands.append(np.asarray(p["hand_pose_params"], dtype=np.float32))
            shapes.append(np.asarray(p["shape_params"], dtype=np.float32))
            scales.append(np.asarray(p["scale_params"], dtype=np.float32))
            exprs.append(np.asarray(p["expr_params"], dtype=np.float32))
            cam_ts.append(np.asarray(p["pred_cam_t"], dtype=np.float32))
            fls.append(float(np.asarray(p["focal_length"]).reshape(-1)[0]))
            present.append(True)
        if not any(present):
            continue

        global_rot_euler = torch.from_numpy(np.stack(grots)).to(device)
        body_pose_euler = torch.from_numpy(np.stack(bpps)).to(device)
        hand_t = torch.from_numpy(np.stack(hands)).to(device)
        shape_t = torch.from_numpy(np.stack(shapes)).to(device)
        scale_t = torch.from_numpy(np.stack(scales)).to(device)
        expr_t = torch.from_numpy(np.stack(exprs)).to(device)
        cam_t_t = torch.from_numpy(np.stack(cam_ts)).to(device)
        f_t = torch.tensor(fls, device=device, dtype=torch.float32)

        verts, kp3d_full, joint_coords, _, joint_rotmats = head.mhr_forward(
            global_trans=torch.zeros_like(global_rot_euler),
            global_rot=global_rot_euler,
            body_pose_params=body_pose_euler,
            hand_pose_params=hand_t,
            scale_params=scale_t,
            shape_params=shape_t,
            expr_params=expr_t,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )
        # y/z flip matches head_pose.forward (camera-y-down convention).
        verts = verts.clone()
        verts[..., [1, 2]] *= -1
        kp3d = kp3d_full[:, :70].clone()
        kp3d[..., [1, 2]] *= -1
        # 238 sapiens face landmarks (70:308) — track retargeted expression
        # so openpose face dots follow new mouth/eye/brow shape.
        kp3d_face = kp3d_full[:, 70:].clone()
        kp3d_face[..., [1, 2]] *= -1
        joint_coords = joint_coords.clone()
        joint_coords[..., [1, 2]] *= -1

        # Recover principal point from any raw frame for reprojection.
        cx = cy = 0.0
        for fi in range(B):
            if not present[fi]:
                continue
            raw = pose_frames[fi][pid]
            kp2d_r = np.asarray(raw["pred_keypoints_2d"], dtype=np.float32)
            kp3d_r = np.asarray(raw["pred_keypoints_3d"], dtype=np.float32)
            ct_r = np.asarray(raw["pred_cam_t"], dtype=np.float32)
            fl_r = float(np.asarray(raw["focal_length"]).reshape(-1)[0])
            x, y, z = kp3d_r[0] + ct_r
            cx = float(kp2d_r[0, 0] - fl_r * x / max(z, 1e-6))
            cy = float(kp2d_r[0, 1] - fl_r * y / max(z, 1e-6))
            break

        def _project_kp(kp3d_local: torch.Tensor) -> torch.Tensor:
            kp3d_cam = kp3d_local + cam_t_t.unsqueeze(1)
            u = f_t[:, None] * kp3d_cam[..., 0] / kp3d_cam[..., 2].clamp(min=1e-6) + cx
            v = f_t[:, None] * kp3d_cam[..., 1] / kp3d_cam[..., 2].clamp(min=1e-6) + cy
            return torch.stack([u, v], dim=-1)

        kp2d = _project_kp(kp3d)
        kp2d_face = _project_kp(kp3d_face)

        verts_np = verts.float().cpu().numpy()
        kp3d_np = kp3d.float().cpu().numpy()
        kp2d_np = kp2d.float().cpu().numpy()
        kp3d_face_np = kp3d_face.float().cpu().numpy()
        kp2d_face_np = kp2d_face.float().cpu().numpy()
        jc_np = joint_coords.float().cpu().numpy()
        jrot_np = joint_rotmats.float().cpu().numpy()

        fi_active = 0
        for fi in range(B):
            if not present[fi]:
                continue
            pose_frames[fi][pid] = dict(pose_frames[fi][pid])
            p = pose_frames[fi][pid]
            p["pred_vertices"] = verts_np[fi_active]
            p["pred_keypoints_3d"] = kp3d_np[fi_active]
            p["pred_keypoints_2d"] = kp2d_np[fi_active]
            p["pred_face_keypoints_3d"] = kp3d_face_np[fi_active]
            p["pred_face_keypoints_2d"] = kp2d_face_np[fi_active]
            p["pred_joint_coords"] = jc_np[fi_active]
            p["pred_global_rots"] = jrot_np[fi_active]
            fi_active += 1
