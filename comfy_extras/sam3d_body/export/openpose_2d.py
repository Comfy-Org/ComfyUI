"""2D OpenPose-style skeleton rendering for SAM 3D Body pose_data.

Body / hand drawing is delegated to `KeypointDraw.draw_wholebody_keypoints`
(shared with SDPose). SAM3D-specific: MHR70 -> DWPose-134 keypoint packing,
plus optional rig-projected face landmarks when `pred_face_keypoints_2d`
isn't present (and arbitrary-count face dots, since sapiens-238 doesn't fit
the DWPose face slot).

Output: (H, W, 3) fp32 torch.Tensor in [0, 1].
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from comfy_extras.pose.keypoint_draw import KeypointDraw

from .glb_shared import (
    OPENPOSE_HAND_COLORS_21,
    openpose_render_keypoints,
    select_face_landmark_vert_ids,
)


_KD = KeypointDraw()
# OpenPose hand palette as a (21, 3) int array (0..255) for KeypointDraw.
_HAND_DOT_PALETTE_OPENPOSE = (OPENPOSE_HAND_COLORS_21 * 255.0).astype(int)


def _project_face_landmarks_2d(
    person: Dict[str, Any], face_vert_ids: np.ndarray, H: int, W: int,
) -> Optional[np.ndarray]:
    """Project `pred_vertices[face_vert_ids]` to 2D using each person's
    pred_cam_t + focal_length. Same projection used by `_replay_mhr_with_overrides`."""
    verts = person.get("pred_vertices")
    cam_t = person.get("pred_cam_t")
    focal = person.get("focal_length")
    if verts is None or cam_t is None or focal is None:
        return None
    verts = np.asarray(verts, dtype=np.float32)
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(3)
    f = float(np.asarray(focal, dtype=np.float32).reshape(-1)[0])
    pts3 = verts[face_vert_ids] + cam_t[None, :]
    z = np.maximum(pts3[:, 2:3], 1e-6)
    xy = pts3[:, :2] * f
    xy = xy + np.array([W * 0.5, H * 0.5], dtype=np.float32)[None, :] * z
    return (xy / z).astype(np.float32)


def _pack_dwpose_134(
    person: Dict[str, Any], pose_data: Dict[str, Any], *,
    include_body: bool, include_hands: bool, H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack a SAM3D person dict into (kp, scores): (134, 2) DWPose-layout
    coords + (134,) confidence. Face slot (24-91) is left zeroed; face dots
    are drawn separately so SAM3D's 238-sapiens / rig-fallback counts work.
    Non-finite or out-of-band entries get score=0 and are filtered downstream.

    Keypoints come from the shared provider: MHR reindexes `pred_keypoints_2d`,
    external rigs (Kimodo) resolve + project from `pred_joint_coords`."""
    kp = np.zeros((134, 2), dtype=np.float32)
    scores = np.zeros(134, dtype=np.float32)

    if include_body:
        body_xy = openpose_render_keypoints(person, pose_data, "body", dim=2, H=H, W=W)
        if body_xy is not None:
            finite = np.isfinite(body_xy).all(axis=1)
            kp[:18][finite] = body_xy[finite]
            scores[:18][finite] = 1.0

    if include_hands:
        for slot_start, part in ((92, "hand_r"), (113, "hand_l")):
            hand_xy = openpose_render_keypoints(person, pose_data, part, dim=2, H=H, W=W)
            if hand_xy is None:
                continue
            finite = np.isfinite(hand_xy).all(axis=1)
            kp[slot_start:slot_start + 21][finite] = hand_xy[finite]
            scores[slot_start:slot_start + 21][finite] = 1.0

    return kp, scores


def _draw_face_dots(
    canvas: np.ndarray, face_xy: np.ndarray, marker_radius_px: int,
) -> None:
    """White face dots, variable count (238 sapiens / ~30 rig-projected)."""
    H, W = canvas.shape[:2]
    pad = int(marker_radius_px)
    centers = []
    for i in range(face_xy.shape[0]):
        x_, y_ = float(face_xy[i, 0]), float(face_xy[i, 1])
        if not (np.isfinite(x_) and np.isfinite(y_)):
            continue
        x, y = int(round(x_)), int(round(y_))
        if x + pad < 0 or x - pad >= W or y + pad < 0 or y - pad >= H:
            continue
        centers.append((x, y))
    _KD.circles(canvas, centers, int(marker_radius_px), (255, 255, 255))


def render_pose_data_openpose(
    pose_data: Dict[str, Any],
    *,
    frame_idx: int,
    W: int,
    H: int,
    background: Optional[torch.Tensor] = None,
    composite: str = "over",
    marker_radius_px: int = 4,
    stick_width_px: int = 4,
    limb_alpha: float = 0.6,
    include_body: bool = True,
    include_hands: bool = False,
    face_style: str = "disabled",
    hand_color_style: str = "dwpose",
    hand_marker_radius_px: int = 0,
    hand_stick_width_px: int = 0,
    face_marker_radius_px: int = 3,
    person_brightness_falloff: float = 0.0,
) -> torch.Tensor:
    """Render a 2D OpenPose-style skeleton onto an (H, W, 3) canvas.

    `composite='over'` paints over `background` (else black canvas).
    `hand_marker_radius_px` / `hand_stick_width_px`: 0 = auto = 0.7x / 0.5x
    of the body sizes.
    `face_style`: 'disabled' / 'full' / 'eyes_mouth'. eyes_mouth falls through
    to the rig fallback since sapiens-238 has no documented subset.
    `person_brightness_falloff` mixes each person's drawn pixels toward white
    by `1 - falloff^k` (track 0 stays vivid). Applied post-draw so per-limb
    alpha blending against the existing canvas remains correct.
    """
    persons = pose_data["frames"][frame_idx]

    if composite == "over" and background is not None:
        bg = background.cpu().numpy()
        canvas = (np.clip(bg, 0.0, 1.0) * 255.0).astype(np.uint8)
        if canvas.shape[:2] != (H, W):
            canvas = np.array(Image.fromarray(canvas).resize((W, H), Image.LANCZOS))
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
    # In-place draw needs a contiguous writable buffer.
    canvas = np.ascontiguousarray(canvas)

    if int(hand_marker_radius_px) <= 0:
        hand_marker_radius_px = max(1, int(round(marker_radius_px * 0.7)))
    if int(hand_stick_width_px) <= 0:
        hand_stick_width_px = max(1, int(round(stick_width_px * 0.5)))

    _EYES_MOUTH_IDX = np.array([6, 7, 8, 9, 10, 11, 12, 13, 19, 20, 21, 22], dtype=np.int64)

    include_face = face_style != "disabled"
    use_rig_only = face_style == "eyes_mouth"

    face_vert_ids: Optional[np.ndarray] = None
    if include_face:
        any_real = (not use_rig_only) and any(
            p.get("pred_face_keypoints_2d") is not None for p in persons
        )
        if not any_real:
            cc = pose_data.get("canonical_colors") or {}
            positions = cc.get("positions")
            if positions is not None:
                face_vert_ids = select_face_landmark_vert_ids(
                    np.asarray(positions), face_mask=cc.get("face_mask"),
                )
                if use_rig_only:
                    face_vert_ids = face_vert_ids[_EYES_MOUTH_IDX]

    hand_dot_color = (
        _HAND_DOT_PALETTE_OPENPOSE if hand_color_style == "openpose"
        else (0, 0, 255)
    )

    falloff = max(0.0, min(1.0, float(person_brightness_falloff)))

    for k, person in enumerate(persons):
        pastel = 0.0 if k == 0 else (1.0 - falloff ** k)
        # Snapshot before this person's strokes so we can identify the pixels
        # they touched and blend just those toward white. Drawing happens
        # against the live canvas first so limb_alpha blends correctly.
        pre = canvas.copy() if pastel > 0 else None

        kp134, scores134 = _pack_dwpose_134(
            person, pose_data, include_body=include_body,
            include_hands=include_hands, H=H, W=W,
        )
        _KD.draw_wholebody_keypoints(
            canvas, kp134, scores=scores134, threshold=0.5,
            draw_body=include_body, draw_feet=False,
            draw_face=False,  # SAM3D draws face dots separately (variable count)
            draw_hands=include_hands,
            stick_width=stick_width_px,
            marker_radius=marker_radius_px,
            hand_stick_width=hand_stick_width_px,
            hand_marker_radius=hand_marker_radius_px,
            limb_alpha=limb_alpha,
            hand_dot_color=hand_dot_color,
        )

        if include_face:
            face_xy = None
            real_face = person.get("pred_face_keypoints_2d")
            if real_face is not None:
                arr = np.asarray(real_face, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    face_xy = arr
            elif face_vert_ids is not None:
                face_xy = _project_face_landmarks_2d(person, face_vert_ids, H, W)
            if face_xy is not None:
                _draw_face_dots(canvas, face_xy, face_marker_radius_px)

        if pre is not None:
            changed = (canvas != pre).any(axis=-1)
            if changed.any():
                touched = canvas[changed].astype(np.float32)
                blended = touched * (1.0 - pastel) + 255.0 * pastel
                canvas[changed] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return torch.from_numpy(canvas.astype(np.float32) / 255.0)
