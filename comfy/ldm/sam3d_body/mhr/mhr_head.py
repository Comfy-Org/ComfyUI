from typing import Optional

import torch
import torch.nn as nn
from comfy.ops import cast_to_input

from ..utils import euler_to_rotmat, rot6d_to_rotmat, rotmat_to_euler, unitquat_to_rotmat
from .mhr_utils import compact_cont_to_model_params_body, compact_cont_to_model_params_hand, mhr_param_hand_idxs

from ..model.transformer import MLP


class MHRHead(nn.Module):

    def __init__(self, input_dim: int, mhr_rig, mlp_depth: int = 1, mlp_channel_div_factor: int = 8, enable_hand_model=False,
        device=None, dtype=None, operations=None):
        super().__init__()
        # Store the shared MHRRig as a non-registered Python attribute
        object.__setattr__(self, "mhr", mhr_rig)

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model

        self.body_cont_dim = 260
        self.npose = (
            6  # Global Rotation
            + self.body_cont_dim  # then body
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = MLP(
            input_dim=input_dim,
            hidden_dim=input_dim // mlp_channel_div_factor,
            output_dim=self.npose,
            num_layers=mlp_depth,
            device=device, dtype=dtype, operations=operations,
        )

        # MHR Parameters
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps

        # Buffers populated by load_state_dict from the safetensors
        def _p(*shape, dtype=torch.float32):
            return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)
        self.joint_rotation = _p(127, 3, 3)
        self.scale_mean = _p(68)
        self.scale_comps = _p(28, 68)
        self.register_buffer("faces", torch.empty(36874, 3, dtype=torch.int64, device=device))
        self._faces_np = None
        self.hand_pose_mean = _p(54)
        self.hand_pose_comps = _p(54, 54)
        self.register_buffer("hand_joint_idxs_left", torch.empty(27, dtype=torch.int64, device=device))
        self.register_buffer("hand_joint_idxs_right", torch.empty(27, dtype=torch.int64, device=device))
        self.keypoint_mapping = _p(308, 18439 + 127)
        # Some special buffers for the hand-version
        self.right_wrist_coords = _p(3)
        self.root_coords = _p(3)
        self.local_to_world_wrist = _p(3, 3)
        self.register_buffer("nonhand_param_idxs", torch.empty(145, dtype=torch.int64, device=device))
        if not enable_hand_model:
            self.register_buffer("face_region_rgb", torch.empty(18439, 3, dtype=torch.float32, device=device))

    def canonical_vertices(self):
        """Return the T-pose vertices for the mean shape (scaled to meters).

        Runs MHR with zero pose / shape / scale / expression so the returned
        mesh is the canonical rest pose — fixed per-model
        """
        device = self.scale_mean.device
        dtype = self.scale_mean.dtype
        B = 1
        global_trans = torch.zeros(B, 3, device=device, dtype=dtype)
        global_rot   = torch.zeros(B, 3, device=device, dtype=dtype)
        body_pose    = torch.zeros(B, 130, device=device, dtype=dtype)
        hand_pose    = torch.zeros(B, self.num_hand_comps * 2, device=device, dtype=dtype)
        scale        = torch.zeros(B, self.num_scale_comps, device=device, dtype=dtype)
        shape        = torch.zeros(B, self.num_shape_comps, device=device, dtype=dtype)
        expr         = torch.zeros(B, self.num_face_comps, device=device, dtype=dtype)

        verts = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot,
            body_pose_params=body_pose,
            hand_pose_params=hand_pose,
            scale_params=scale,
            shape_params=shape,
            expr_params=expr,
        )  # single-tensor shape (1, N_v, 3) in meters
        return verts[0]

    def faces_np(self):
        """Static topology — cached so the per-layer pose_output doesn't force a D2H sync."""
        if self._faces_np is None:
            self._faces_np = self.faces.cpu().numpy()
        return self._faces_np

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136

        # This drops in the hand poses from hand_pose_params (PCA 6D) into full_pose_params.
        # Split into left and right hands
        left_hand_params, right_hand_params = torch.split(
            hand_pose_params,
            [self.num_hand_pose_comps, self.num_hand_pose_comps],
            dim=1,
        )

        # Change from cont to model params
        left_hand_params_model_params = compact_cont_to_model_params_hand(
            cast_to_input(self.hand_pose_mean, left_hand_params, copy=False)
            + torch.einsum("da,ab->db", left_hand_params, cast_to_input(self.hand_pose_comps, left_hand_params, copy=False))
        )
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            cast_to_input(self.hand_pose_mean, right_hand_params, copy=False)
            + torch.einsum("da,ab->db", right_hand_params, cast_to_input(self.hand_pose_comps, right_hand_params, copy=False))
        )

        # Drop it in
        full_pose_params[:, self.hand_joint_idxs_left.to(full_pose_params.device)] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right.to(full_pose_params.device)] = right_hand_params_model_params

        return full_pose_params  # B x 207

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
    ):
        # Align everything to the static buffers
        dt = self.scale_mean.dtype
        global_trans = global_trans.to(dt)
        global_rot = global_rot.to(dt)
        body_pose_params = body_pose_params.to(dt)
        if hand_pose_params is not None:
            hand_pose_params = hand_pose_params.to(dt)
        scale_params = scale_params.to(dt)
        shape_params = shape_params.to(dt)
        if expr_params is not None:
            expr_params = expr_params.to(dt)

        if self.enable_hand_model:
            # Transfer wrist-centric predictions to the body.
            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = rotmat_to_euler(
                "xyz",
                euler_to_rotmat("xyz", global_rot_ori) @ cast_to_input(self.local_to_world_wrist, global_rot_ori, copy=False),
            )
            right_wrist_coords = cast_to_input(self.right_wrist_coords, global_rot, copy=False)
            root_coords = cast_to_input(self.root_coords, global_rot, copy=False)
            global_trans = (
                -(
                    euler_to_rotmat("xyz", global_rot)
                    @ (right_wrist_coords - root_coords)
                    + root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]

        # Convert from scale and shape params to actual scales and vertices

        # Add singleton batches in case...
        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        # Convert scale...
        scale_mean = cast_to_input(self.scale_mean, scale_params, copy=False)
        scale_comps = cast_to_input(self.scale_comps, scale_params, copy=False)
        scales = scale_mean[None, :] + scale_params @ scale_comps

        # Now, figure out the pose.
        ## 10 here is because it's more stable to optimize global translation in meters.
        full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1)  # B x 127
        ## Put in hands
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )
        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:
            # Zero out non-hand parameters
            model_params[:, self.nonhand_param_idxs.to(model_params.device)] = 0

        curr_skinned_verts, curr_skel_state = self.mhr(
            shape_params, model_params, expr_params
        )
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100
        curr_joint_rots = unitquat_to_rotmat(curr_joint_quats)

        # Prepare returns
        to_return = [curr_skinned_verts]
        if return_keypoints:
            # Get sapiens 308 keypoints
            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )  # B x (num_verts + 127) x 3

            kp_map = cast_to_input(self.keypoint_mapping, model_vert_joints, copy=False)
            model_keypoints_pred = (
                (kp_map @ model_vert_joints.permute(1, 0, 2).flatten(1, 2))
                .reshape(-1, model_vert_joints.shape[0], 3)
                .permute(1, 0, 2)
            )

            if self.enable_hand_model:
                # Zero out everything except for the right hand
                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

    def forward(self, x: torch.Tensor, init_estimate: Optional[torch.Tensor] = None, intermediate: bool = False):
        """
        Args:
            x: pose token with shape [B, C], usually C=DECODER.DIM
            init_estimate: [B, self.npose]
            intermediate: when True, the caller only needs the keypoints/pose
                outputs needed by the per-layer keypoint-token update path —
                vertex output is suppressed so `camera_project` skips the
                18439-vertex perspective projection on intermediate decoder
                layers. The final layer must call with intermediate=False.
        """
        batch_size = x.shape[0]
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # From pred, we want to pull out individual predictions.

        ## First, get globals
        ### Global rotation is first 6.
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # B x 3 x 3
        global_rot_euler = rotmat_to_euler("ZYX", global_rot_rotmat)  # B x 3
        global_trans = torch.zeros_like(global_rot_euler)

        ## Next, get body pose.
        ### Hold onto raw, continuous version for iterative correction.
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        ### Convert to eulers (and trans)
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
        ### Zero-out hands
        pred_pose_euler[:, mhr_param_hand_idxs] = 0
        ### Zero-out jaw
        pred_pose_euler[:, -3:] = 0

        ## Get remaining parameters
        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps

        # Run everything through mhr
        output = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )

        # Some existing code to get joints and fix camera system
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
        j3d = j3d[:, :70]  # 308 --> 70 keypoints

        # Intermediate decoder layers only consume pred_keypoints_3d via the
        # keypoint-token update path; suppress verts so camera_project skips
        # the 18439-vertex perspective projection.
        if intermediate:
            verts = None
        if verts is not None:
            verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1

        # Head-MLP outputs are promoted to fp32 here so the external
        # pose_output["mhr"] contract has a stable dtype regardless of what
        # the head ran at (fp16/bf16 for speed). MHR-derived outputs are
        # already fp32 from MHR's math layers.
        output = {
            "pred_pose_raw": torch.cat([global_rot_6d, pred_pose_cont], dim=1).float(),
            "pred_pose_rotmat": None,
            "global_rot": global_rot_euler.float(),
            "body_pose": pred_pose_euler.float(),
            "shape": pred_shape.float(),
            "scale": pred_scale.float(),
            "hand": pred_hand.float(),
            "face": pred_face.float(),
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": verts.reshape(batch_size, -1, 3) if verts is not None else None,
            "pred_joint_coords": jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None,
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }

        return output
