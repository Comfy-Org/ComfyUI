# Adapted from facebookresearch/MHR (Apache 2.0):
#   https://github.com/facebookresearch/MHR/blob/main/mhr/mhr.py
# Skinning ops follow facebookincubator/momentum (Apache 2.0) — formulas
# verbatim from the upstream mhr_model.pt
# (pymomentum.{skel_state,quaternion,backend.skel_state_backend}).
# Original Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ops import cast_to_input

from .mhr_utils import batch6DFromXYZ

_LN2 = 0.6931471824645996

# Half-angle cos/sin are computed on the
# whole (..., 3) at once and concatenated to [cr, cp, cy, sr, sp, sy]; _EQ_I then
# picks the three factors of each term, reproducing:
#   x = sr*cp*cy - cr*sp*sy      z = cr*cp*sy - sr*sp*cy
#   y = cr*sp*cy + sr*cp*sy      w = cr*cp*cy + sr*sp*sy
def _euler_xyz_to_quat(angles):
    """(roll, pitch, yaw) -> quaternion (x, y, z, w). Matches pymomentum.quaternion.euler_xyz_to_quaternion."""
    half = angles * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    cr, cp, cy = c.unbind(-1)
    sr, sp, sy = s.unbind(-1)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


# Hamilton product as gather + 3 adds. Each output component is a 4-term sum;
# _QM_P1/_QM_P2 pick the operands and _QM_S the signs, reproducing:
#   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
#   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def _quat_rotate(q, v):
    """Rotate v by unit quaternion q (xyzw). v + 2 * (axis x v * w + axis x (axis x v))."""
    axis = q[..., :3]
    r = q[..., 3:4]
    av = torch.cross(axis, v, dim=-1)
    aav = torch.cross(axis, av, dim=-1)
    return v + 2.0 * (av * r + aav)


def _skel_multiply(s1, s2):
    """Compose two skel states (..., 8). Returns parent ∘ child.

    Mirrors pymomentum.skel_state.multiply: both quaternions are renormalized
    before composition. With many FK levels the previously-normalized quats
    drift in ULPs; upstream renormalizes defensively, so we do too to stay
    bit-close to its outputs.
    """
    t1, sc1 = s1[..., :3], s1[..., 7:8]
    t2, sc2 = s2[..., :3], s2[..., 7:8]
    q1 = F.normalize(s1[..., 3:7], p=2, dim=-1, eps=1e-12)
    q2 = F.normalize(s2[..., 3:7], p=2, dim=-1, eps=1e-12)
    t_res = t1 + sc1 * _quat_rotate(q1, t2)
    q_res = _quat_multiply(q1, q2)
    s_res = sc1 * sc2
    return torch.cat([t_res, q_res, s_res], dim=-1)


def _skel_transform_points(skel_state, points):
    """Apply skel_state (..., 8) to points (..., 3): t + q * (s * points).

    Assumes the quaternion in skel_state is already unit-norm. Callers that
    can't guarantee that should normalize first.
    """
    t = skel_state[..., :3]
    q = skel_state[..., 3:7]
    s = skel_state[..., 7:8]
    return t + _quat_rotate(q, s * points)


def _global_skel_state_from_local(local, pmi_levels):
    """FK walk in fp64 (matches upstream's use_double_precision=True path).

    `pmi_levels` is a precomputed list of (source_idx, target_idx) tensor pairs,
    one per BFS level. Avoids per-call torch.split + tolist() sync.
    """
    orig_dtype = local.dtype
    g = local.to(torch.float64).clone()
    for source, target in pmi_levels:
        parent = g.index_select(-2, target)
        child  = g.index_select(-2, source)
        g.index_copy_(-2, source, _skel_multiply(parent, child))
    return g.to(orig_dtype)


class MHRRig(nn.Module):
    """Plain-PyTorch reimplementation of Meta's MHR rig.

    All math runs in fp32 (FK upcast to fp64 internally, matching upstream's
    use_double_precision=True backend) regardless of the host model's dtype.
    """

    NUM_VERTS = 18439
    NUM_JOINTS = 127
    NUM_LBS_TRIPLETS = 51337
    NUM_IDENTITY = 45
    NUM_EXPR = 72
    PARAM_TRANSFORM_IN = 249    # = model_parameters(204) + identity_coeffs(45)
    PARAM_TRANSFORM_OUT = 889   # = NUM_JOINTS * 7
    POSE_CORR_IN = 750          # = (NUM_JOINTS - 2) * 6
    POSE_CORR_HIDDEN = 3000
    POSE_CORR_SPARSE_NNZ = 53136

    def __init__(self, device=None):
        super().__init__()

        # All buffers are populated by load_state_dict from the `mhr.*` keys
        def _p(*shape, dtype=torch.float32):
            return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)
        def _b(name, *shape, dtype):
            self.register_buffer(name, torch.empty(*shape, dtype=dtype, device=device))

        self.base_shape = _p(self.NUM_VERTS, 3)
        self.identity_basis = _p(self.NUM_IDENTITY, self.NUM_VERTS, 3)
        self.expr_basis = _p(self.NUM_EXPR, self.NUM_VERTS, 3)
        self.param_transform = _p(self.PARAM_TRANSFORM_OUT, self.PARAM_TRANSFORM_IN)

        self.skel_joint_translation_offsets = _p(self.NUM_JOINTS, 3)
        self.skel_joint_prerotations = _p(self.NUM_JOINTS, 4)
        _b("skel_joint_parents", self.NUM_JOINTS, dtype=torch.int32)
        _b("skel_pmi", 2, 266, dtype=torch.int64)
        _b("skel_pmi_buffer_sizes", 4, dtype=torch.int64)

        self.lbs_inverse_bind_pose = _p(self.NUM_JOINTS, 8)
        self.lbs_skin_weights = _p(self.NUM_LBS_TRIPLETS)
        _b("lbs_skin_indices", self.NUM_LBS_TRIPLETS, dtype=torch.int32)
        _b("lbs_vert_indices", self.NUM_LBS_TRIPLETS, dtype=torch.int64)

        _b("pose_corr_sparse_indices", 2, self.POSE_CORR_SPARSE_NNZ, dtype=torch.int64)
        self.pose_corr_sparse_weight = _p(self.POSE_CORR_SPARSE_NNZ)

        _b("pose_corr_sparse_shape", 2, dtype=torch.int64)
        self.pose_corr_weight = _p(self.NUM_VERTS * 3, self.POSE_CORR_HIDDEN)
        self.pose_corr_bias = None
        self._pmi_sizes = None
        self._pose_corr_shape = None
        self.register_load_state_dict_post_hook(self._set_pmi_sizes)

    def _set_pmi_sizes(self, module, incompatible_keys):
        self._pmi_sizes = tuple(self.skel_pmi_buffer_sizes.tolist())
        self._pose_corr_shape = tuple(self.pose_corr_sparse_shape.tolist())

    def forward(self, identity_coeffs, model_parameters, expr_coeffs, apply_correctives: bool = True):
        dtype = self.base_shape.dtype
        identity_coeffs = identity_coeffs.to(dtype)
        model_parameters = model_parameters.to(dtype)
        expr_coeffs = expr_coeffs.to(dtype)
        B = identity_coeffs.shape[0]

        base_shape = cast_to_input(self.base_shape, identity_coeffs, copy=False)
        identity_basis = cast_to_input(self.identity_basis, identity_coeffs, copy=False)
        identity_rest = base_shape + torch.einsum("nvd,bn->bvd", identity_basis, identity_coeffs)

        cat_in = torch.cat([model_parameters, torch.zeros_like(identity_coeffs)], dim=1)
        joint_parameters = torch.einsum("dn,bn->bd", cast_to_input(self.param_transform, cat_in, copy=False), cat_in)

        jp = joint_parameters.view(B, self.NUM_JOINTS, 7)
        local_t = jp[..., :3] + cast_to_input(self.skel_joint_translation_offsets, jp, copy=False).unsqueeze(0)
        local_q = _euler_xyz_to_quat(jp[..., 3:6])
        local_q = _quat_multiply(cast_to_input(self.skel_joint_prerotations, local_q, copy=False).unsqueeze(0), local_q)
        local_s = torch.exp(jp[..., 6:7] * _LN2)
        local_state = torch.cat([local_t, local_q, local_s], dim=-1)

        skel_state = _global_skel_state_from_local(local_state, self._pmi_levels(local_state.device))

        face_expr = torch.einsum("nvd,bn->bvd", cast_to_input(self.expr_basis, expr_coeffs, copy=False), expr_coeffs)
        unposed = identity_rest + face_expr
        if apply_correctives:
            unposed = unposed + self._pose_correctives(joint_parameters)

        verts = self._skin(skel_state, unposed)
        return verts, skel_state

    def _pose_correctives(self, joint_parameters):
        B = joint_parameters.shape[0]
        jp = joint_parameters.view(B, self.NUM_JOINTS, 7)
        # Joints [2:] only — root and one more skipped. Take Euler XYZ (cols 3:6).
        feat = batch6DFromXYZ(jp[:, 2:, 3:6], return_9D=False)  # (B, 125, 6)
        feat[..., 0] -= 1.0
        feat[..., 4] -= 1.0
        feat = feat.flatten(1, 2)  # (B, 750)

        h = (self._sparse_w(feat) @ feat.T).T                # (B, 3000)
        h = F.relu(h)
        out = F.linear(h, cast_to_input(self.pose_corr_weight, h, copy=False), self.pose_corr_bias)  # (B, 55317)
        return out.view(B, self.NUM_VERTS, 3)

    def _pmi_levels(self, device):
        if self._pmi_sizes is None:
            raise RuntimeError("MHR rig weights have not been loaded")
        pmi = self.skel_pmi.to(device=device)
        return [(part[0], part[1]) for part in torch.split(pmi, self._pmi_sizes, dim=1)]

    def _sparse_w(self, ref):
        if self._pose_corr_shape is None:
            raise RuntimeError("MHR rig weights have not been loaded")
        w = cast_to_input(self.pose_corr_sparse_weight, ref, copy=False)
        # PyTorch 2.12 warns unless invariant checking is explicitly scoped.
        with torch.sparse.check_sparse_tensor_invariants():
            return torch.sparse_coo_tensor(
                self.pose_corr_sparse_indices.to(device=ref.device),
                w,
                self._pose_corr_shape,
            ).coalesce()

    def _skin(self, skel_state, rest_verts):
        B = skel_state.shape[0]
        ibp = cast_to_input(self.lbs_inverse_bind_pose, skel_state, copy=False).unsqueeze(0).expand(B, self.NUM_JOINTS, 8)
        joint_xform = _skel_multiply(skel_state, ibp)

        norm_q = F.normalize(joint_xform[..., 3:7], p=2, dim=-1, eps=1e-12)
        joint_xform = torch.cat([joint_xform[..., :3], norm_q, joint_xform[..., 7:8]], dim=-1)

        sk_idx = self.lbs_skin_indices.to(device=rest_verts.device, dtype=torch.long)
        v_idx = self.lbs_vert_indices.to(device=rest_verts.device)
        w = cast_to_input(self.lbs_skin_weights, rest_verts, copy=False)

        per_triplet_xform = joint_xform.index_select(-2, sk_idx)   # (B, 51337, 8)
        per_triplet_rest = rest_verts.index_select(-2, v_idx)      # (B, 51337, 3)
        contrib = _skel_transform_points(per_triplet_xform, per_triplet_rest) * w.unsqueeze(0).unsqueeze(-1)

        out = torch.zeros(B, self.NUM_VERTS, 3, dtype=rest_verts.dtype, device=rest_verts.device)
        out.index_add_(-2, v_idx, contrib)
        return out
