"""BVH export: OFFSETs and channel rotations describe the bind pose.

The rig comes from an external `_skeleton_override`, so no model checkpoint (and
no SAM 3D Body download) is needed. The synthetic chain carries non-identity bind
rotations, which is what makes the OFFSET space observable: with identity bind
rotations the parent-local and world-axis offsets coincide.
"""

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import numpy as np  # noqa: E402

from comfy_extras.sam3d_body.export.bvh import build_bvh  # noqa: E402


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(0.5 * angle), [np.cos(0.5 * angle)]])


def _qmul(q1, q2):
    """xyzw quaternion product."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _qconj(q):
    return np.concatenate([-q[:3], q[3:]])


def _rotate(q, v):
    axis, w = np.asarray(q[:3]), q[3]
    av = np.cross(axis, v)
    return v + 2.0 * (av * w + np.cross(axis, av))


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _chain_rig():
    """4-joint chain; joints 1..3 carry non-identity bind rotations, like MHR's.
    Joint 0 is the static world anchor and joints 1..3 are its descendants."""
    parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    local_pos = [np.zeros(3), np.array([0.0, 1.0, 0.0]),
                 np.array([0.0, 0.4, 0.0]), np.array([0.3, 0.0, 0.0])]
    local_rot = [_quat_axis_angle(a, t) for a, t in [
        ([1.0, 0.0, 0.0], 0.7), ([0.0, 1.0, 0.0], -0.9),
        ([1.0, 0.0, 0.0], 1.1), ([0.0, 0.0, 1.0], 0.5)]]

    # FK the bind locals into globals: t_child = t_parent + R_parent * t_local.
    bind = np.zeros((4, 8))
    bind[:, 7] = 1.0
    for j in range(4):
        p = int(parents[j])
        bind[j, 3:7] = local_rot[j] if p < 0 else _qmul(bind[p, 3:7], local_rot[j])
        bind[j, :3] = local_pos[j] if p < 0 else bind[p, :3] + _rotate(bind[p, 3:7], local_pos[j])
    return parents, bind


def _pose_data(parents, bind):
    """Two frames of a known global pose on a `_skeleton_override` rig. Frame 0 is
    the bind pose; frame 1 swings every joint about world Y. The stored joint
    coords are FK'd from the same swung rotations the rig would report, so the
    two agree by construction."""
    NJ = len(parents)
    swing = _quat_axis_angle([0.0, 1.0, 0.0], 0.35)
    q_frames = [bind[:, 3:7], np.stack([_qmul(swing, bind[j, 3:7]) for j in range(NJ)])]

    frames = []
    for q in q_frames:
        coords = np.zeros((NJ, 3))
        for j in range(NJ):
            p = int(parents[j])
            # bind[j] - bind[p] is already rotated by the parent's bind rotation,
            # so undo that and apply the posed parent rotation instead.
            coords[j] = (bind[j, :3] if p < 0 else
                         coords[p] + _rotate(q[p], _rotate(_qconj(bind[p, 3:7]),
                                                           bind[j, :3] - bind[p, :3])))
        frames.append([{
            "pred_global_rots": np.stack([_quat_to_mat(q[j]) for j in range(NJ)]),
            "pred_joint_coords": (coords[None] * 0.01).astype(np.float32),
            "pred_cam_t": np.zeros(3, dtype=np.float32),
        }])
    return {
        "frames": frames,
        "_skeleton_override": {
            "parents": parents,
            "bind_global_m": bind.copy(),
            "rest_verts_m": np.zeros((1, 3), dtype=np.float32),
            "faces": np.zeros((0, 3), dtype=np.uint32),
            "lbs_compact_joints": np.zeros((1, 1), dtype=np.uint16),
            "lbs_compact_weights": np.ones((1, 1), dtype=np.float32),
            "lbs_compact_max_inf": 1,
        },
    }


def _parse(text):
    """-> (names, offsets, channel_counts, motion) for the single joint chain."""
    hierarchy, motion = text.split("HIERARCHY", 1)[1].split("MOTION")
    offsets, counts, names = [], [], []
    for line in hierarchy.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            names.append(stripped.split()[1])
        elif stripped.startswith("OFFSET"):
            offsets.append(np.array([float(v) for v in stripped.split()[1:4]]))
        elif stripped.startswith("CHANNELS"):
            counts.append(int(stripped.split()[1]))
    # The trailing End Site OFFSET belongs to no joint, so keep len(names).
    offsets = offsets[:len(names)]
    rows = np.array([[float(v) for v in line.split()]
                     for line in motion.split("Frame Time:", 1)[1].splitlines()[1:]
                     if line.strip()])
    return names, offsets, counts, rows


def _euler_to_mat_deg(zxy):
    """BVH `CHANNELS Zrotation Xrotation Yrotation` compose as Rz * Rx * Ry."""
    z, x, y = np.deg2rad(zxy)
    cz, sz, cx, sx, cy, sy = np.cos(z), np.sin(z), np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


def _fk(offsets, motion_row):
    """Standard BVH forward kinematics over the single joint chain: each joint's
    local matrix is T(OFFSET) * R(channels), and world = parent_world * local."""
    world_pos = []
    world = np.eye(4)
    for i, off in enumerate(offsets):
        local = np.eye(4)
        rot = np.eye(3) if i == 0 else _euler_to_mat_deg(motion_row[6 + 3 * (i - 1):9 + 3 * (i - 1)])
        local[:3, :3] = rot
        local[:3, 3] = motion_row[:3] if i == 0 else off
        world = local if i == 0 else world @ local
        world_pos.append(world[:3, 3].copy())
    return np.array(world_pos)


def _build(parents, bind, **kwargs):
    text = build_bvh(_pose_data(parents, bind), model=None, **kwargs).decode()
    return _parse(text)


def test_bvh_offsets_are_bind_world_deltas():
    """OFFSET is the bind-pose child-minus-parent vector in world axes. The rig's
    translation_offsets are already in the parent's bind frame, so writing them
    straight through garbles the rest skeleton whenever bind rotations are not
    identity."""
    parents, bind = _chain_rig()
    _, offsets, counts, _ = _build(parents, bind, units="m", camera_translation="off")

    assert counts == [6, 3, 3]
    # Joint 0 is the static anchor, so the BVH root is its first child (joint 1).
    for node, joint in enumerate([1, 2, 3]):
        parent = int(parents[joint])
        expected = bind[joint, :3] - bind[parent, :3]
        assert np.linalg.norm(offsets[node] - expected) < 1e-6


def test_bvh_zero_channels_reproduce_bind_pose():
    """BVH's rest pose is all-zero channels; it has to be the rig's bind pose,
    otherwise importers and retargeters start from a garbled skeleton. Frame 0 of
    `_pose_data` is the bind pose itself, so its channels must all be zero."""
    parents, bind = _chain_rig()
    _, offsets, _, rows = _build(parents, bind, units="m", camera_translation="off")

    assert np.abs(rows[0, 3:]).max() < 1e-4

    rest = _fk(offsets, np.concatenate([bind[1, :3], np.zeros(9)]))
    for node, joint in enumerate([1, 2, 3]):
        err = np.linalg.norm(rest[node] - bind[joint, :3])
        assert err < 1e-6, f"rest position of joint {joint} is {err} m from bind"


def test_bvh_posed_channels_encode_bind_relative_rotation():
    """Channels hold each joint's rotation relative to its bind rotation, made
    parent-local. `_pose_data` swings every joint by the same world rotation, so
    only the root carries it and the children stay identity. Together with the
    world-axis OFFSETs this is what puts the all-zero pose on the bind skeleton."""
    parents, bind = _chain_rig()
    _, offsets, _, rows = _build(parents, bind, units="m", camera_translation="off")

    swing = _quat_axis_angle([0.0, 1.0, 0.0], 0.35)
    rotated = [_qmul(swing, bind[j, 3:7]) for j in range(4)]      # posed globals
    rel = [_qmul(rotated[j], _qconj(bind[j, 3:7])) for j in range(4)]

    for joint in (1, 2, 3):
        # The BVH root (joint 1) drops the static anchor, so it keeps the global
        # bind-relative rotation; deeper joints are made parent-local.
        want = (rel[joint] if joint == 1
                else _qmul(rel[joint], _qconj(rel[int(parents[joint])])))
        channel = _euler_to_mat_deg(rows[1][3:6] if joint == 1
                                    else rows[1][6 + 3 * (joint - 2):9 + 3 * (joint - 2)])
        err = np.linalg.norm(channel - _quat_to_mat(want))
        assert err < 1e-5, f"joint {joint} channel is {err} off the bind-relative rotation"
