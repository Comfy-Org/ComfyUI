# MHR (Meta Human Rig) parameter packing/unpacking. The 6D-rotation helpers
# (batch6DFromXYZ, batchXYZfrom6D) are the continuity
# representation from Zhou et al., "On the Continuity of Rotation
# Representations in Neural Networks" (CVPR 2019, https://arxiv.org/abs/1812.07035),
# implementations from papagina/RotationContinuity:
# https://github.com/papagina/RotationContinuity/blob/758b0ce5/shapenet/code/tools.py
# The compact_cont_to_model_params_* functions are MHR-rig-specific glue.

import torch
import torch.nn.functional as F


def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """
    # Compute relative rotation matrix
    R_rel = torch.matmul(A, B.transpose(-2, -1))  # (B, 3, 3)
    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B,)
    # Compute angle using the trace formula
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    # Compute angle difference
    angle = torch.acos(cos_theta_clamped)
    return angle


def fix_wrist_euler(
    wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5), limits_y=(-1.2, 1.5)
):
    """
    wrist_xzy: B x 2 x 3 (X, Z, Y angles)
    Returns: Fixed angles within joint limits
    """
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]

    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))

    # Calculate L2 violation distance
    def calc_violation(val, limits):
        below = torch.clamp(limits[0] - val, min=0.0)
        above = torch.clamp(val - limits[1], min=0.0)
        return below**2 + above**2

    violation_orig = (
        calc_violation(x, limits_x)
        + calc_violation(z, limits_z)
        + calc_violation(y, limits_y)
    )

    violation_alt = (
        calc_violation(x_alt, limits_x)
        + calc_violation(z_alt, limits_z)
        + calc_violation(y_alt, limits_y)
    )

    # Use alternative where it has lower L2 violation
    use_alt = violation_alt < violation_orig

    # Stack alternative and apply mask
    wrist_xzy_alt = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    result = torch.where(use_alt.unsqueeze(-1), wrist_xzy_alt, wrist_xzy)

    return result


# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py
def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype, device=r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result


# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L82
def batchXYZfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)  # ... x 3 x 3

    # Now get it into euler
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L412
    sy = torch.sqrt(
        matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0]
    )
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(-matrix[..., 2, 0], sy)
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

    xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    ys = torch.atan2(-matrix[..., 2, 0], sy)
    zs = matrix[..., 1, 0] * 0

    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x * (1 - singular) + xs * singular
    out_euler[..., 1] = y * (1 - singular) + ys * singular
    out_euler[..., 2] = z * (1 - singular) + zs * singular

    return out_euler


_HAND_DOFS = (3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1)
_HAND_CONT_3DOF_MASK = [k == 3 for k in _HAND_DOFS for _ in range(2 * k)]
_HAND_CONT_1DOF_MASK = [k in (1, 2) for k in _HAND_DOFS for _ in range(2 * k)]
_HAND_MODEL_3DOF_MASK = [k == 3 for k in _HAND_DOFS for _ in range(k)]
_HAND_MODEL_1DOF_MASK = [k in (1, 2) for k in _HAND_DOFS for _ in range(k)]


def compact_cont_to_model_params_hand(hand_cont):
    # These are ordered by joint, not model params ^^
    # Convert hand_cont to eulers
    ## First for 3DoFs
    hand_cont_threedofs = hand_cont[..., _HAND_CONT_3DOF_MASK].unflatten(-1, (-1, 6))
    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_cont_onedofs = hand_cont[..., _HAND_CONT_1DOF_MASK].unflatten(
        -1, (-1, 2)
    )  # (sincos)
    hand_model_params_onedofs = torch.atan2(
        hand_cont_onedofs[..., -2], hand_cont_onedofs[..., -1]
    )

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27, dtype=hand_cont.dtype, device=hand_cont.device)
    hand_model_params[..., _HAND_MODEL_3DOF_MASK] = hand_model_params_threedofs
    hand_model_params[..., _HAND_MODEL_1DOF_MASK] = hand_model_params_onedofs

    return hand_model_params


# fmt: off
_BODY_3DOF_IDXS = ((0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132))
_BODY_1DOF_ROT_IDXS = (1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123)
_BODY_1DOF_TRANS_IDXS = (124, 125, 126, 127, 128, 129)
_BODY_3DOF_FLAT_IDXS = tuple(i for group in _BODY_3DOF_IDXS for i in group)
# fmt: on


def compact_cont_to_model_params_body(body_pose_cont):
    num_3dof_angles = len(_BODY_3DOF_IDXS) * 3
    num_1dof_angles = len(_BODY_1DOF_ROT_IDXS)
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_params_3dofs = batchXYZfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])
    ## Nothing to do for trans
    body_params_trans = body_cont_trans
    # Put them together
    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133, dtype=body_pose_cont.dtype, device=body_pose_cont.device)
    body_pose_params[..., list(_BODY_3DOF_FLAT_IDXS)] = body_params_3dofs
    body_pose_params[..., list(_BODY_1DOF_ROT_IDXS)] = body_params_1dofs
    body_pose_params[..., list(_BODY_1DOF_TRANS_IDXS)] = body_params_trans
    return body_pose_params


# Hand indices into the 133-dim body-pose vector.
mhr_param_hand_idxs = list(range(62, 116))
