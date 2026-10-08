import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from comfy.ldm.sam3d_body.utils import prepare_batch
from comfy.ldm.sam3.tracker import unpack_masks
from comfy.ldm.sam3d_body.model.model import SAM3DBody


import comfy.model_management
import comfy.utils

from tqdm import tqdm

def _bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """xyxy bounds of a binary mask, with sub-5px speckles filtered out."""
    m = mask[..., 0] if mask.dim() == 3 else mask
    m_bool = m > 0
    t = m_bool.to(torch.float32)[None, None]
    eroded = -F.max_pool2d(-t, kernel_size=5, stride=1, padding=2)
    eroded_mask = eroded[0, 0] > 0.5
    keep = torch.where(eroded_mask.any(), eroded_mask, m_bool)

    H, W = keep.shape
    xs = torch.arange(W, device=keep.device)
    ys = torch.arange(H, device=keep.device)
    cols = keep.any(dim=0)
    rows = keep.any(dim=1)
    bbox = torch.stack([
        torch.where(cols, xs, W).amin(),
        torch.where(rows, ys, H).amin(),
        torch.where(cols, xs + 1, 0).amax(),
        torch.where(rows, ys + 1, 0).amax(),
    ])
    full_frame = bbox.new_tensor([0, 0, W, H])
    return torch.where(m_bool.any(), bbox, full_frame).to(torch.float32)


def inputs_from_sam3_track(track_data, B: int, H: int, W: int):
    """Unpack SAM3_TRACK_DATA into per-frame per-object bboxes + masks at image
    resolution. Returns (None, None) on empty track / frame-count mismatch."""

    packed = track_data.get("packed_masks") if isinstance(track_data, dict) else None
    if packed is None:
        return None, None
    N, K = packed.shape[0], packed.shape[1]
    if N != B or K == 0:
        return None, None

    device = comfy.model_management.get_torch_device()
    unpacked = unpack_masks(packed.to(device))  # (N, K, Hm, Wm) bool
    Hm, Wm = unpacked.shape[2], unpacked.shape[3]
    resized = F.interpolate(
        unpacked.float().reshape(N * K, 1, Hm, Wm),
        size=(H, W), mode="bilinear", align_corners=False,
    )
    arr_gpu = (resized > 0.5).to(torch.uint8).reshape(N, K, H, W)
    arr = arr_gpu.cpu()

    per_frame_masks = [arr[f, :, :, :, None].contiguous() for f in range(N)]
    per_frame_bboxes = []
    for f in range(N):
        derived = []
        for k in range(K):
            # Erosion + argmax bbox on GPU; CPU max_pool2d over N*K full-res masks is slow.
            b = _bbox_from_mask(arr_gpu[f, k])
            derived.append(b.cpu())
        per_frame_bboxes.append(torch.stack(derived, dim=0))
    return per_frame_bboxes, per_frame_masks


# Soft budget for the batched Predict path
BATCHED_CROPS_PER_CHUNK = 64


def _quat_to_mat_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    """(3,3) rotation from a wxyz quaternion; columns are the rotated axes."""
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def cam_int_from_fov(height: int, width: int, fov_degrees: float) -> Optional[torch.Tensor]:
    """(1,3,3) intrinsic matrix from a vertical FOV in degrees. Matches MoGe2's
    convention (vertical focal for both axes). Returns None for fov<=0 so the
    caller falls back to prepare_batch's diagonal-focal default."""
    if fov_degrees <= 0:
        return None
    focal = height / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return torch.tensor(
        [[[focal, 0.0, width / 2.0],
          [0.0, focal, height / 2.0],
          [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )


def apply_camera_override(mhr_pose_data: Dict[str, Any], camera_info: Dict[str, Any],
                          H: int, W: int) -> Dict[str, Any]:
    """Re-project every frame's pose through a Load3D 6DOF camera (position/
    target/zoom + optional FOV). Returns a new mhr_pose_data, unchanged on
    empty/invalid input."""
    first_frame = mhr_pose_data["frames"][0] if mhr_pose_data["frames"] else []
    if not first_frame:
        return mhr_pose_data
    # Per-person rig root (pred_cam_t) and body centroid (mesh mean), in camera space.
    roots, centroids = [], []
    for p in first_frame:
        cam_t = p.get("pred_cam_t")
        if cam_t is None:
            continue
        cam_t = np.asarray(cam_t, dtype=np.float32).reshape(3)
        roots.append(cam_t)
        v = p.get("pred_vertices")
        centroids.append(np.asarray(v, dtype=np.float32).reshape(-1, 3).mean(axis=0) + cam_t
                         if v is not None else cam_t)
    if not roots:
        return mhr_pose_data
    subj_root = np.mean(np.stack(roots, axis=0), axis=0)
    subj_centroid = np.mean(np.stack(centroids, axis=0), axis=0)

    # Meter-scale, so Three.js coords map 1:1 (Three.js Y-up → flip Y,Z)
    pos = camera_info.get("position") or {}
    tgt = camera_info.get("target") or {}
    pos_v = np.array([float(pos.get("x", 0.0)), -float(pos.get("y", 5.0)), -float(pos.get("z", 0.0))], dtype=np.float32)
    tgt_v = np.array([float(tgt.get("x", 0.0)), -float(tgt.get("y", 0.0)), -float(tgt.get("z", 0.0))], dtype=np.float32)
    offset = pos_v - tgt_v
    has_offset = float(np.linalg.norm(offset)) >= 1e-6
    q = camera_info.get("quaternion")
    if not has_offset and not q:
        return mhr_pose_data  # no viewpoint and no orientation -> nothing to apply

    zoom = float(camera_info.get("zoom", 1.0)) or 1.0
    # SAM3D roots near the feet. A target at the origin -> center the body centroid
    if float(np.linalg.norm(tgt_v)) < 1e-6:
        target = subj_centroid
    else:
        target = subj_root + tgt_v

    if q:
        mv = lambda v: np.array([v[0], -v[1], -v[2]], dtype=np.float32)
        norm = lambda v: v / max(1e-6, float(np.linalg.norm(v)))
        Rc = _quat_to_mat_wxyz(
            float(q.get("w", 1.0)), float(q.get("x", 0.0)),
            float(q.get("y", 0.0)), float(q.get("z", 0.0)),
        )  # columns = camera world axes
        x_axis = norm(mv(Rc[:, 0]))   # camera +X -> image right
        y_axis = norm(mv(-Rc[:, 1]))  # image +Y is down -> negative of camera up
        z_axis = norm(mv(-Rc[:, 2]))  # camera looks down local -Z -> view direction
    else:
        # x degenerates only when looking straight along world-up -> world +X.
        z_axis = -offset / float(np.linalg.norm(offset))
        x_axis = np.cross(z_axis, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        x_norm = float(np.linalg.norm(x_axis))
        x_axis = x_axis / x_norm if x_norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)

    # Eye: dolly along the offset; rotation-only camera keeps the predicted
    # viewing distance so only orientation/roll changes.
    if has_offset:
        eye = target + offset / max(0.01, zoom)
    else:
        d = max(0.1, float(target[2]))
        eye = target - z_axis * (d / max(0.01, zoom))

    # Lens: camera FoV if given, else the SAM3D predicted focal. Three.js fov
    # is vertical → focal from image height.
    cam_fov = float(camera_info.get("fov", 0.0) or 0.0)
    if cam_fov > 0:
        new_focal = float(H) / (2.0 * float(np.tan(np.deg2rad(cam_fov) / 2.0)))
    else:
        f0 = first_frame[0].get("focal_length")
        new_focal = (float(np.asarray(f0, dtype=np.float32).reshape(-1)[0]) if f0 is not None
                     else float(H) / (2.0 * float(np.tan(np.deg2rad(50.0) / 2.0))))

    center = np.array([W * 0.5, H * 0.5], dtype=np.float32)
    reproj = {"pred_keypoints_3d": "pred_keypoints_2d", "pred_face_keypoints_3d": "pred_face_keypoints_2d"}
    # External rigs store pred_joint_coords Y-up; transform them through the
    # camera too (in camera space, then back to Y-up) so they follow the override.
    override = mhr_pose_data.get("_skeleton_override")
    joints_y_up = override is not None and not bool(override.get("per_frame_y_down", False))
    new_frames: List[List[Dict[str, Any]]] = []
    for frame in mhr_pose_data["frames"]:
        scaled = []
        for p in frame:
            p = dict(p)
            cam_t = p.get("pred_cam_t")
            if cam_t is None:
                scaled.append(p)
                continue
            cam_t = np.asarray(cam_t, dtype=np.float32).reshape(3)
            for k in ("pred_keypoints_3d", "pred_vertices", "pred_face_keypoints_3d"):
                v = p.get(k)
                if v is None:
                    continue
                cam = (np.asarray(v, dtype=np.float32) + cam_t - eye) @ R.T
                p[k] = cam.astype(np.float32)
                if k in reproj:  # re-project the new 3D to 2D image coords
                    z = np.maximum(cam[..., 2:3], 1e-6)
                    p[reproj[k]] = (cam[..., :2] * new_focal / z + center).astype(np.float32)
            jc = p.get("pred_joint_coords")
            if jc is not None:
                jc = np.asarray(jc, dtype=np.float32).copy()
                if joints_y_up:
                    jc[..., 1] *= -1.0
                    jc[..., 2] *= -1.0
                jc = (jc + cam_t - eye) @ R.T
                if joints_y_up:
                    jc[..., 1] *= -1.0
                    jc[..., 2] *= -1.0
                p["pred_joint_coords"] = jc.astype(np.float32)
            p["pred_cam_t"] = np.zeros(3, dtype=np.float32)
            p["focal_length"] = np.array(new_focal, dtype=np.float32)
            scaled.append(p)
        new_frames.append(scaled)
    out = dict(mhr_pose_data)
    out["frames"] = new_frames
    return out


def run_batched_single_chunk(inner: SAM3DBody, frames_rgb: List[torch.Tensor], per_frame_boxes: List[torch.Tensor],
    per_frame_masks: Optional[List[torch.Tensor]], image_size: Tuple[int, int], inference_type: str, K: int,
    cam_int: Optional[torch.Tensor] = None) -> List[List[Dict[str, Any]]]:
    """Run a SINGLE chunk of frames through run_inference in one forward."""
    N = len(frames_rgb)
    total = N * K

    boxes_stacked = torch.stack(
        [per_frame_boxes[f][k] for f in range(N) for k in range(K)], dim=0
    )
    img_per_crop = [frames_rgb[f] for f in range(N) for _ in range(K)]

    if per_frame_masks is not None:
        # One mask but multiple bboxes per frame → each bbox gets the same mask.
        flat_masks = []
        for f in range(N):
            mf = per_frame_masks[f]
            if mf.shape[0] == 1 and K > 1:
                mf = mf.repeat_interleave(K, dim=0)
            flat_masks.extend([mf[k] for k in range(K)])
        masks_stacked = torch.stack(flat_masks, dim=0)
        masks_score = torch.ones(total, dtype=torch.float32)
    else:
        masks_stacked = None
        masks_score = None

    batch = prepare_batch(
        img_per_crop, boxes_stacked,
        input_size=image_size,
        masks=masks_stacked, masks_score=masks_score, cam_int=cam_int,
    )
    device = comfy.model_management.get_torch_device()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    outputs = inner.run_inference(
        img_per_crop,
        batch,
        inference_type=inference_type,
        thresh_wrist_angle=1.4,
    )

    if inference_type == "full":
        pose_output, batch_lhand, batch_rhand, _, _ = outputs
    else:
        pose_output = outputs
        batch_lhand = batch_rhand = None

    out = {k: v.float().cpu().numpy() for k, v in pose_output["mhr"].items()
           if v is not None and k != "faces"}

    # Snapshot batch['bbox'] to CPU before we release `batch` references
    batch_bbox_cpu = batch["bbox"][0].cpu().numpy()
    lhand_bboxes = rhand_bboxes = None
    if inference_type == "full" and batch_lhand is not None and batch_rhand is not None:
        lhand_bboxes = _bboxes_from_center_scale(batch_lhand)
        rhand_bboxes = _bboxes_from_center_scale(batch_rhand)

    del pose_output, batch, batch_lhand, batch_rhand, outputs

    frames_out: List[List[Dict[str, Any]]] = []
    for f in range(N):
        persons: List[Dict[str, Any]] = []
        for k in range(K):
            idx = f * K + k
            p: Dict[str, Any] = {
                "bbox": batch_bbox_cpu[idx],
                "focal_length": out["focal_length"][idx],
                "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                "pred_vertices": out["pred_vertices"][idx],
                "pred_cam_t": out["pred_cam_t"][idx],
                "pred_pose_raw": out["pred_pose_raw"][idx],
                "global_rot": out["global_rot"][idx],
                "body_pose_params": out["body_pose"][idx],
                "hand_pose_params": out["hand"][idx],
                "scale_params": out["scale"][idx],
                "shape_params": out["shape"][idx],
                "expr_params": out["face"][idx],
                "mask": (per_frame_masks[f][k] if per_frame_masks[f].shape[0] > 1 else per_frame_masks[f][0])
                        if per_frame_masks is not None else None,
                "pred_joint_coords": out["pred_joint_coords"][idx],
                "pred_global_rots": out["joint_global_rots"][idx],
                "mhr_model_params": out["mhr_model_params"][idx],
                # 238 face landmarks from sapiens-308 (indices 70..308 of the pre-slice keypoint tensor).
                "pred_face_keypoints_3d": out["pred_face_keypoints_3d"][idx] if "pred_face_keypoints_3d" in out else None,
                "pred_face_keypoints_2d": out["pred_face_keypoints_2d"][idx] if "pred_face_keypoints_2d" in out else None,
            }
            if lhand_bboxes is not None:
                p["lhand_bbox"] = lhand_bboxes[idx]
                p["rhand_bbox"] = rhand_bboxes[idx]
            persons.append(p)
        frames_out.append(persons)
    return frames_out


def run_batched_frames(
    inner: SAM3DBody,
    frames_rgb: List[torch.Tensor],
    per_frame_boxes: List[torch.Tensor],
    per_frame_masks: Optional[List[torch.Tensor]],
    image_size: Tuple[int, int],
    inference_type: str,
    cam_int: Optional[torch.Tensor] = None,
    pbar: Optional[comfy.utils.ProgressBar] = None,
    crops_per_chunk: int = BATCHED_CROPS_PER_CHUNK,
) -> List[List[Dict[str, Any]]]:
    """Run the clip through chunked batched run_inference calls.

    Supports K persons per frame (K must be the same across frames — padded
    externally). Splits frames into chunks so chunk_frames * K <= budget; each
    chunk is one body forward + optional hand forwards over its person-crops
    """
    N = len(frames_rgb)
    assert N > 0, "empty frame list"
    K_set = {len(b) for b in per_frame_boxes}
    assert len(K_set) == 1, f"batched path requires same bbox count per frame, got {K_set}"
    K = K_set.pop()
    assert K >= 1, "need at least one bbox per frame"

    chunk_frames = max(1, crops_per_chunk // K)
    results: List[List[Dict[str, Any]]] = []
    with tqdm(total=N, desc="SAM3D body inference") as t:
        for start in range(0, N, chunk_frames):
            end = min(N, start + chunk_frames)
            sub_frames = frames_rgb[start:end]
            sub_boxes = per_frame_boxes[start:end]
            sub_masks = None if per_frame_masks is None else per_frame_masks[start:end]
            chunk_result = run_batched_single_chunk(
                inner, sub_frames, sub_boxes, sub_masks,
                image_size, inference_type, K,
                cam_int=cam_int,
            )
            results.extend(chunk_result)
            t.update(end - start)
            if pbar is not None:
                pbar.update(end - start)
    return results


def _bboxes_from_center_scale(batch) -> np.ndarray:
    centers = batch["bbox_center"].flatten(0, 1)
    half_scales = batch["bbox_scale"].flatten(0, 1) * 0.5
    return torch.cat([centers - half_scales, centers + half_scales], dim=1).float().cpu().numpy()

# Wire types and small helpers shared across the SAM 3D Body node modules.

def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI image tensor (any shape, float 0..1) → uint8 tensor in [0, 255] on CPU."""
    return (image * 255.0).clamp(0.0, 255.0).to(dtype=torch.uint8, device="cpu")


def compute_canonical_colors(model) -> Dict[str, np.ndarray]:
    """Canonical rest-pose data for shader color lookups: positions (Nv,3),
    norm (Nv,3 in [0,1]), face_mask, head_mask, and face_region_rgb
    (per-region painted color from the .safetensors)."""
    verts = model.head_pose.canonical_vertices().float().cpu().numpy()
    faces = model.head_pose.faces_np()

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
    vn = vn / ln
    norm_map = ((vn + 1.0) * 0.5).astype(np.float32)

    face_mask = _compute_face_mask(model)
    # Head: above jaw-neck (y>1.43) and narrower than shoulders (|x|<0.11).
    # Ears reach |x|≈0.09; shoulders start at |x|≈0.20.
    head_mask = (verts[:, 1] > 1.43) & (np.abs(verts[:, 0]) < 0.11)

    # Painted per-vertex face region RGB ships in the model .safetensors as
    # `head_pose.face_region_rgb` and gets loaded by load_state_dict.
    face_region_rgb = model.head_pose.face_region_rgb.detach().float().cpu().numpy()

    return {
        "positions": verts.astype(np.float32),
        "norm": norm_map,
        "face_mask": face_mask,
        "head_mask": head_mask,
        "face_region_rgb": face_region_rgb,
    }


def compute_hand_vert_mask(model, hand_radius_m: float = 0.15, weight_threshold: float = 0.5) -> np.ndarray:
    """(Nv,) bool mask of hand-region verts. Picks joints within `hand_radius_m`
    of the mhr70 hand keypoint clusters (indices 21..62), then sums sparse LBS
    weights across them; verts above `weight_threshold` are hand verts."""
    head = model.head_pose
    mhr = head.mhr
    device = head.scale_mean.device

    zeros = lambda *s: torch.zeros(1, *s, device=device)
    out = head.mhr_forward(
        global_trans=zeros(3),
        global_rot=zeros(3),
        body_pose_params=zeros(130),
        hand_pose_params=zeros(head.num_hand_comps * 2),
        scale_params=zeros(head.num_scale_comps),
        shape_params=zeros(head.num_shape_comps),
        expr_params=zeros(head.num_face_comps),
        return_keypoints=True,
        return_joint_coords=True,
    )
    # Output order with these flags: (verts, kp, jcoords). See mhr_head.mhr_forward.
    _, kp, jcoords = out[0], out[1], out[2]
    kp = kp[0, :70].cpu().numpy()
    jcoords = jcoords[0].cpu().numpy()

    right_center = kp[21:42].mean(axis=0)
    left_center = kp[42:63].mean(axis=0)
    j_dist_r = np.linalg.norm(jcoords - right_center, axis=1)
    j_dist_l = np.linalg.norm(jcoords - left_center, axis=1)
    is_hand_joint = (j_dist_r < hand_radius_m) | (j_dist_l < hand_radius_m)

    lbs_w = mhr.lbs_skin_weights.cpu().numpy()
    lbs_v = mhr.lbs_vert_indices.cpu().numpy()
    lbs_j = mhr.lbs_skin_indices.cpu().numpy()
    is_hand_joint_f = is_hand_joint.astype(np.float32)

    n_verts = mhr.NUM_VERTS
    hand_mass = np.zeros(n_verts, dtype=np.float32)
    np.add.at(hand_mass, lbs_v, lbs_w * is_hand_joint_f[lbs_j])
    return hand_mass >= weight_threshold


def _compute_face_mask(model, disp_threshold_m: float = 1e-4) -> np.ndarray:
    """(Nv,) bool mask of verts that move with face expression. Sweeps each of
    the 72 expression axes at coef=+1.0 and flags any vert that moves more
    than `disp_threshold_m` for at least one axis."""
    head = model.head_pose
    device = head.scale_mean.device
    num_face = head.num_face_comps

    # One batched forward over [neutral, axis_0 .. axis_71]
    batch = num_face + 1
    zeros = lambda *s: torch.zeros(batch, *s, device=device)
    expr = torch.zeros(batch, num_face, device=device)
    expr[1:] = torch.eye(num_face, device=device)
    verts = head.mhr_forward(
        global_trans=zeros(3),
        global_rot=zeros(3),
        body_pose_params=zeros(130),
        hand_pose_params=zeros(head.num_hand_comps * 2),
        scale_params=zeros(head.num_scale_comps),
        shape_params=zeros(head.num_shape_comps),
        expr_params=expr,
    ).cpu().numpy()  # (batch, Nv, 3)

    disp = np.linalg.norm(verts[1:] - verts[0][None], axis=2)  # (num_face, Nv)
    return (disp > disp_threshold_m).any(axis=0)


def jet_colormap(s: np.ndarray) -> np.ndarray:
    """matplotlib jet, (N,) in [0,1] -> (N, 3) float32 RGB."""
    s = np.asarray(s, dtype=np.float32).clip(0.0, 1.0)
    r = np.interp(s, [0.0, 0.35, 0.66, 0.89, 1.0], [0.0, 0.0, 1.0, 1.0, 0.5])
    g = np.interp(s, [0.0, 0.125, 0.375, 0.64, 0.91, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    b = np.interp(s, [0.0, 0.11, 0.34, 0.65, 1.0], [0.5, 1.0, 1.0, 0.0, 0.0])
    return np.stack([r, g, b], axis=-1).astype(np.float32)
