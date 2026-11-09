"""GLB export — skeletal (real armature) mode.

Rebuilds an Armature with the MHR 127-bone rig: per-frame local TRS from
param_transform on `mhr_model_params`, rest verts from a zero-pose forward,
sparse skinning compacted to glTF's 4-influence form, and facial expression as
72 morph targets driven by `expr_params`. Optional octahedron bone-vis is
rigidly skinned alongside for viewers that don't draw bones. Shared infra lives
in `glb_shared.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .glb_shared import (
    GLBWriter,
    Rig,
    bake_vertex_colors,
    bone_locals_from_globals,
    collect_tracks,
    compute_normals,
    compute_pastel_mix,
    flat_shade_mesh,
    gaussian_smooth_quats,
    global_skel_state_from_pose_data,
    global_skel_state_per_frame,
    ibp_from_bind_global,
    make_lit_material,
    quat_sign_fix_per_joint,
    rotation_align,
    unflip,
)

from comfy_extras.sam3d_body.utils import jet_colormap

def _bone_colors_rgb(bind_pos_m: np.ndarray, scheme: str) -> Optional[np.ndarray]:
    """Per-bone RGB (NJ, 3) float32 in [0, 1]. None for 'white' (default material)."""
    if scheme == "rainbow_y":
        y = bind_pos_m[:, 1].astype(np.float32)
        y_min, y_max = float(y.min()), float(y.max())
        s = np.clip((y - y_min) / max(y_max - y_min, 1e-6), 0.0, 1.0)
        return jet_colormap(s)
    return None


def _octahedron_unit() -> Tuple[np.ndarray, np.ndarray]:
    """Canonical Blender-style bone octahedron: head at origin, tail at +Y, unit
    length, ridge at 1/10 height. 6 verts, 8 triangles, faces wound outward."""
    v = np.array([
        [0.0, 0.0, 0.0],   # 0: head
        [0.0, 1.0, 0.0],   # 1: tail
        [1.0, 0.1, 0.0],   # 2: +X ridge (pre-scale; X/Z scale by half_width)
        [-1.0, 0.1, 0.0],  # 3: -X ridge
        [0.0, 0.1, 1.0],   # 4: +Z ridge
        [0.0, 0.1, -1.0],  # 5: -Z ridge
    ], dtype=np.float32)
    f = np.array([
        # head pyramid: outward = away from bone axis, slightly -Y
        [0, 2, 4], [0, 5, 2], [0, 3, 5], [0, 4, 3],
        # tail pyramid: outward = away from bone axis, slightly +Y
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ], dtype=np.uint32)
    return v, f


def _bone_edges(
    joint_pos_m: np.ndarray, parents: np.ndarray,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """One (parent_idx, child_idx, head_pos, tail_pos) per parent→child edge.
    Skips edges whose parent is a root (world-anchor sticks) and zero-length
    edges."""
    NJ = joint_pos_m.shape[0]
    out: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for c in range(NJ):
        p = int(parents[c])
        if not (0 <= p < NJ and p != c):
            continue
        # Skip world-anchor sticks: parent itself is a root.
        gp = int(parents[p])
        if not (0 <= gp < NJ and gp != p):
            continue
        head = joint_pos_m[p].astype(np.float32)
        tail = joint_pos_m[c].astype(np.float32)
        if float(np.linalg.norm(tail - head)) < 1e-6:
            continue
        out.append((p, c, head, tail))
    return out


def _build_bone_octahedrons_mesh(
    bind_joint_pos_m: np.ndarray, parents: np.ndarray, half_width_m: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One octahedron per parent→child edge. Returns (verts, normals, faces,
    joints, weights, child_idx_per_vert); child_idx feeds per-bone color."""
    base_v, base_f = _octahedron_unit()
    canonical = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    out_v: List[List[float]] = []
    out_n: List[List[float]] = []
    out_f: List[List[int]] = []
    out_j: List[List[int]] = []
    out_w: List[List[float]] = []
    child_per_vert: List[int] = []

    # Width scales with length (capped by half_width_m) so short bones aren't chunky.
    WIDTH_RATIO = 0.1
    MIN_WIDTH = 0.001
    for parent_idx, child_idx, head, tail in _bone_edges(bind_joint_pos_m, parents):
        direction = tail - head
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        unit_dir = direction / length
        R = rotation_align(canonical, unit_dir)

        half_width_eff = max(MIN_WIDTH, min(length * WIDTH_RATIO, half_width_m))
        scale = np.array([half_width_eff, length, half_width_eff], dtype=np.float32)
        v_local = base_v * scale
        v_world = v_local @ R.T + head

        # head pole outward = -Y, tail pole +Y, ridges outward in XZ.
        n_local = np.zeros_like(base_v)
        n_local[0] = [0.0, -1.0, 0.0]
        n_local[1] = [0.0, 1.0, 0.0]
        for k in range(2, 6):
            n = base_v[k].copy()
            n[1] = 0.0
            n_norm = float(np.linalg.norm(n))
            if n_norm > 0:
                n_local[k] = n / n_norm
        n_world = n_local @ R.T

        v_off = len(out_v)
        out_v.extend(v_world.tolist())
        out_n.extend(n_world.tolist())
        for face in base_f:
            out_f.append([int(face[0]) + v_off, int(face[1]) + v_off, int(face[2]) + v_off])
        # Dual skin (head→parent, tail→child); ridges blend by canonical Y so
        # the bone stretches between joints instead of going rigid with one.
        for k in range(base_v.shape[0]):
            y_canon = float(base_v[k, 1])
            w_parent = max(0.0, 1.0 - y_canon)
            w_child = max(0.0, y_canon)
            wsum = w_parent + w_child
            if wsum > 0:
                w_parent /= wsum
                w_child /= wsum
            out_j.append([int(parent_idx), int(child_idx), 0, 0])
            out_w.append([w_parent, w_child, 0.0, 0.0])
            child_per_vert.append(int(child_idx))

    if not out_v:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint32), np.zeros((0, 4), dtype=np.uint16),
                np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
    return (np.asarray(out_v, dtype=np.float32),
            np.asarray(out_n, dtype=np.float32),
            np.asarray(out_f, dtype=np.uint32),
            np.asarray(out_j, dtype=np.uint16),
            np.asarray(out_w, dtype=np.float32),
            np.asarray(child_per_vert, dtype=np.int64))


def build_glb_skeletal(
    pose_data: Dict[str, Any],
    model: Any = None,
    *,
    fps: float = 24.0,
    camera_translation: str = "off",
    track_index: int = -1,
    include_face_morphs: bool = True,
    shader: str = "default",
    rainbow_tilt_x_deg: float = 0.0,
    rainbow_tilt_z_deg: float = 0.0,
    person_palette_falloff: float = 0.6,
    bone_smooth_window: int = 0,
    use_stored_global_rots: bool = True,
    bone_vis: str = "off",
    bone_vis_radius_m: float = 0.04,
    bone_vis_color: str = "white",
    include_body_mesh: bool = True,
) -> bytes:
    """Build pose_data as a real Armature GLB with per-bone TRS keyframes. For
    MHR, facial expression is exposed as 72 morph targets when
    include_face_morphs=True.

    External skeletons (e.g. ComfyUI-Kimodo) can supply
    ``pose_data["_skeleton_override"]`` to bypass MHR rig extraction (``model``
    may be None then); per-frame state still reads ``pred_global_rots`` /
    ``pred_joint_coords``. See ``glb_shared._get_skeleton_override`` for the schema.
    """
    frames = pose_data["frames"]
    # Only `pred_cam_t` is camera-y-down; everything else is rig-native Y-up.
    faces_native = np.ascontiguousarray(pose_data["faces"], dtype=np.uint32)
    tracks = collect_tracks(pose_data, track_index)
    if not tracks:
        raise ValueError("build_glb_skeletal: no valid tracks in pose_data")

    rig = Rig.from_pose_data(pose_data, model)
    NJ = rig.num_joints
    NEXPR = rig.num_expr
    parents = rig.parents
    if not rig.can_rerun_fk:
        # External rigs have no PCA pose params to re-run; use stored globals.
        use_stored_global_rots = True
    joint_coords_y_down = rig.per_frame_y_down
    # Skin already compacted to ≤8 influences/vertex (some shoulder/hip verts
    # need >4, else per-bone rotation noise leaks into the mesh).
    joints_8 = rig.lbs_joints
    weights_8 = rig.lbs_weights
    actual_max_inf = rig.lbs_max_inf
    joints_set0 = np.ascontiguousarray(joints_8[:, :4])
    weights_set0 = np.ascontiguousarray(weights_8[:, :4])
    use_set1 = actual_max_inf > 4
    joints_set1 = np.ascontiguousarray(joints_8[:, 4:8]) if use_set1 else None
    weights_set1 = np.ascontiguousarray(weights_8[:, 4:8]) if use_set1 else None
    # Derive bone locals from bind globals so any `parents`/FK mismatch is
    # absorbed into the local TRS instead of producing wrong globals.
    bind_global_m = rig.bind_global_m
    bind_local = bone_locals_from_globals(bind_global_m[None], parents)[0]

    # IBP = inverse of bind global → skin_matrix at rest is identity.
    ibp_mat4 = ibp_from_bind_global(bind_global_m)

    w = GLBWriter()

    nodes: List[dict] = []
    meshes: List[dict] = []
    skins: List[dict] = []
    materials: List[dict] = []
    animations: List[dict] = []
    scene_root_indices: List[int] = []
    canonical_colors = pose_data.get("canonical_colors")

    indices_acc = w.add_indices_u32(faces_native)
    joints0_acc = w.add_joints_u16(joints_set0)
    weights0_acc = w.add_weights_f32(weights_set0)
    joints1_acc = w.add_joints_u16(joints_set1) if use_set1 else None
    weights1_acc = w.add_weights_f32(weights_set1) if use_set1 else None
    ibm_acc = w.add_mat4_f32(ibp_mat4)

    expr_morph_accs: List[int] = []
    if include_face_morphs and NEXPR > 0:
        eb = rig.expr_basis.astype(np.float32) * 0.01
        for e in range(NEXPR):
            expr_morph_accs.append(w.add_vec3_f32_no_minmax(eb[e]))

    samplers: List[dict] = []
    channels: List[dict] = []
    for track_i, (person_k, frame_indices) in enumerate(tracks):
        person_root = {"name": f"track{track_i:02d}", "children": []}
        nodes.append(person_root)
        person_root_idx = len(nodes) - 1
        scene_root_indices.append(person_root_idx)

        bone_node_indices: List[int] = []
        for j in range(NJ):
            bone = {
                "name": f"bone_{j:03d}",
                "translation": bind_local[j, :3].tolist(),
                "rotation": bind_local[j, 3:7].tolist(),
                "scale": [float(bind_local[j, 7])] * 3,
            }
            nodes.append(bone)
            bone_node_indices.append(len(nodes) - 1)

        bone_children: List[List[int]] = [[] for _ in range(NJ)]
        bone_root_indices: List[int] = []
        for j in range(NJ):
            p = int(parents[j])
            if 0 <= p < NJ and p != j:
                bone_children[p].append(bone_node_indices[j])
            else:
                bone_root_indices.append(bone_node_indices[j])
        for j in range(NJ):
            if bone_children[j]:
                nodes[bone_node_indices[j]]["children"] = bone_children[j]
        person_root["children"].extend(bone_root_indices)

        skin = {
            "joints": bone_node_indices,
            "inverseBindMatrices": ibm_acc,
            "skeleton": bone_root_indices[0] if bone_root_indices else bone_node_indices[0],
        }
        skins.append(skin)
        skin_idx = len(skins) - 1

        include_body = bool(include_body_mesh)
        include_bones = bone_vis == "octahedrons"
        body_mesh_node_idx: Optional[int] = None

        if include_body:
            # MHR rest verts depend on shape_params; external rigs ignore the arg.
            shape_params_arr = np.asarray(
                frames[frame_indices[0]][person_k].get("shape_params", []),
                dtype=np.float32,
            )
            rest_v = rig.rest_verts_m(shape_params_arr)
            normals = compute_normals(rest_v, faces_native)
            positions_acc = w.add_vec3_f32(rest_v)
            normals_acc = w.add_vec3_f32(normals)

            pastel_mix = compute_pastel_mix(track_i, person_palette_falloff)
            vcolor = bake_vertex_colors(
                canonical_colors, shader,
                rainbow_tilt_x_deg, rainbow_tilt_z_deg, pastel_mix,
            )
            color_acc = w.add_vec3_f32(vcolor) if vcolor is not None else None

            attributes = {
                "POSITION": positions_acc, "NORMAL": normals_acc,
                "JOINTS_0": joints0_acc, "WEIGHTS_0": weights0_acc,
            }
            if joints1_acc is not None:
                attributes["JOINTS_1"] = joints1_acc
                attributes["WEIGHTS_1"] = weights1_acc
            if color_acc is not None:
                attributes["COLOR_0"] = color_acc
            primitive = {
                "attributes": attributes,
                "indices": indices_acc,
                "mode": 4,
            }
            # See-through body when bones are shown, else opaque (only if a
            # shader baked COLOR_0; otherwise default material).
            if color_acc is not None or include_bones:
                materials.append(make_lit_material(opacity=0.35 if include_bones else 1.0))
                primitive["material"] = len(materials) - 1
            if expr_morph_accs:
                primitive["targets"] = [{"POSITION": a} for a in expr_morph_accs]

            mesh = {"primitives": [primitive]}
            if expr_morph_accs:
                mesh["weights"] = [0.0] * len(expr_morph_accs)
            meshes.append(mesh)
            mesh_idx = len(meshes) - 1

            mesh_node = {
                "name": f"track{track_i:02d}_mesh", "mesh": mesh_idx, "skin": skin_idx,
            }
            nodes.append(mesh_node)
            body_mesh_node_idx = len(nodes) - 1
            person_root["children"].append(body_mesh_node_idx)

        if include_bones:
            bone_palette = _bone_colors_rgb(bind_global_m[:, :3], bone_vis_color)

            # Color by child joint so every bone has its own color.
            color_idx_per_vert: Optional[np.ndarray] = None
            hw = float(bone_vis_radius_m)
            bv_v, bv_n, bv_f, bv_j, bv_w, child_per_vert = _build_bone_octahedrons_mesh(
                bind_global_m[:, :3], parents, half_width_m=hw,
            )
            if bv_v.shape[0] > 0:
                F = bv_f.shape[0]
                expanded_child = np.empty((F * 3,), dtype=np.int64)
                for k in range(3):
                    expanded_child[k::3] = child_per_vert[bv_f[:, k]]
                bv_v, bv_n, bv_f, bv_j, bv_w = flat_shade_mesh(bv_v, bv_f, bv_j, bv_w)
                color_idx_per_vert = expanded_child
            primitive_mode = 4
            bv_idx_flat = bv_f.reshape(-1)

            if bv_v.shape[0] > 0:
                bv_pos_acc = w.add_vec3_f32(bv_v)
                bv_idx_acc = w.add_indices_u32(bv_idx_flat)
                bv_j_acc = w.add_joints_u16(bv_j)
                bv_w_acc = w.add_weights_f32(bv_w)
                bv_attrs = {
                    "POSITION": bv_pos_acc,
                    "JOINTS_0": bv_j_acc, "WEIGHTS_0": bv_w_acc,
                }
                if bv_n is not None:
                    bv_attrs["NORMAL"] = w.add_vec3_f32(bv_n)
                if bone_palette is not None and color_idx_per_vert is not None:
                    bv_color = bone_palette[color_idx_per_vert].astype(np.float32)
                    bv_attrs["COLOR_0"] = w.add_vec3_f32(bv_color)
                bv_primitive = {
                    "attributes": bv_attrs,
                    "indices": bv_idx_acc,
                    "mode": primitive_mode,
                }
                if bone_palette is not None:
                    materials.append(make_lit_material())
                    bv_primitive["material"] = len(materials) - 1
                bv_mesh = {"primitives": [bv_primitive]}
                meshes.append(bv_mesh)
                bv_mesh_node = {
                    "name": f"track{track_i:02d}_bones",
                    "mesh": len(meshes) - 1,
                    "skin": skin_idx,
                }
                nodes.append(bv_mesh_node)
                person_root["children"].append(len(nodes) - 1)

        # Per-frame global skel state → bone locals via parent-inverse. Stored
        # output by default; fallback re-runs FK.
        if use_stored_global_rots:
            rig_global_m = global_skel_state_from_pose_data(
                pose_data, frame_indices, person_k, NJ,
                joint_coords_y_down=joint_coords_y_down,
            )
        else:
            mp_per_frame = np.stack([
                np.asarray(frames[t][person_k]["mhr_model_params"], dtype=np.float32)
                for t in frame_indices
            ], axis=0)
            rig_global_cm = global_skel_state_per_frame(model, mp_per_frame)
            rig_global_m = rig_global_cm.copy().astype(np.float32)
            rig_global_m[..., :3] *= 0.01
        # Sign-fix global quats BEFORE deriving locals: a parent's ±180° flip
        # would otherwise propagate into the child's local translation and cause
        # visible "axis resets" mid-animation.
        rig_global_m[..., 3:7] = quat_sign_fix_per_joint(rig_global_m[..., 3:7])
        bone_local_anim = bone_locals_from_globals(rig_global_m, parents)
        local_t = bone_local_anim[..., :3].astype(np.float32)
        local_q = bone_local_anim[..., 3:7].astype(np.float32)
        local_s = bone_local_anim[..., 7].astype(np.float32)
        # Second pass on locals catches residual drift from the parent-inverse.
        local_q = quat_sign_fix_per_joint(local_q)
        # Align frame 0 with the bind quat so pause/play takes the short path.
        bind_q = bind_local[:, 3:7].astype(np.float32)
        if local_q.shape[0] > 0:
            d0 = (bind_q * local_q[0]).sum(axis=-1)
            sign0 = np.where(d0 < 0, -1.0, 1.0).astype(np.float32)[:, None]
            local_q[0] = local_q[0] * sign0
            local_q = quat_sign_fix_per_joint(local_q)
        # Optional smoothing for multi-frame rig spikes (e.g. q.w at handstand).
        if bone_smooth_window and bone_smooth_window > 1:
            local_q = gaussian_smooth_quats(local_q, int(bone_smooth_window))
        # fp64 renormalize → fp32; viewers' nlerp amplifies non-unit drift.
        lq64 = local_q.astype(np.float64)
        lq64 = lq64 / np.maximum(np.linalg.norm(lq64, axis=-1, keepdims=True), 1e-12)
        local_q = lq64.astype(np.float32)

        n_frames = len(frame_indices)
        times = np.asarray(frame_indices, dtype=np.float32) / float(fps)
        time_acc = w.add_scalar_f32(times)

        for j in range(NJ):
            t_j = local_t[:, j, :]
            q_j = local_q[:, j, :]
            s_j = np.broadcast_to(local_s[:, j:j+1], (n_frames, 3)).astype(np.float32)

            t_const = (np.ptp(t_j, axis=0) < 1e-6).all()
            q_const = (np.ptp(q_j, axis=0) < 1e-6).all()
            s_const = (np.ptp(s_j, axis=0) < 1e-6).all()

            if t_const:
                nodes[bone_node_indices[j]]["translation"] = t_j[0].tolist()
            else:
                acc = w.add_vec3_f32_anim(t_j)
                samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": bone_node_indices[j], "path": "translation"},
                })

            if q_const:
                nodes[bone_node_indices[j]]["rotation"] = q_j[0].tolist()
            else:
                acc = w.add_vec4_f32(q_j)
                samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": bone_node_indices[j], "path": "rotation"},
                })

            if s_const:
                nodes[bone_node_indices[j]]["scale"] = s_j[0].tolist()
            else:
                acc = w.add_vec3_f32_anim(s_j)
                samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
                channels.append({
                    "sampler": len(samplers) - 1,
                    "target": {"node": bone_node_indices[j], "path": "scale"},
                })

        if camera_translation != "off":
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

        # Body mesh only — bone-vis primitives have no morph targets.
        if expr_morph_accs and body_mesh_node_idx is not None:
            expr_per_frame = np.stack([
                np.asarray(frames[t][person_k]["expr_params"], dtype=np.float32)
                for t in frame_indices
            ], axis=0).astype(np.float32)
            weights_acc_anim = w.add_scalar_f32_flat(expr_per_frame, count=n_frames * NEXPR)
            samplers.append({"input": time_acc, "output": weights_acc_anim, "interpolation": "LINEAR"})
            channels.append({
                "sampler": len(samplers) - 1,
                "target": {"node": body_mesh_node_idx, "path": "weights"},
            })

    if samplers:
        animations.append({
            "name": "all_tracks",
            "samplers": samplers, "channels": channels,
        })

    gltf = {
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
