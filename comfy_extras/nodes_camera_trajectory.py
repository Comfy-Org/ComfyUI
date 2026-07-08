import nodes
import torch
import numpy as np
from einops import rearrange
from typing_extensions import override
import comfy.model_management

from comfy_api.latest import ComfyExtension, io


CAMERA_DICT = {
    "base_T_norm": 1.5,
    "base_angle": np.pi/3,
    "Static": {     "angle":[0., 0., 0.],   "T":[0., 0., 0.]},
    "Pan Up": {     "angle":[0., 0., 0.],   "T":[0., -1., 0.]},
    "Pan Down": {   "angle":[0., 0., 0.],   "T":[0.,1.,0.]},
    "Pan Left": {   "angle":[0., 0., 0.],   "T":[-1.,0.,0.]},
    "Pan Right": {  "angle":[0., 0., 0.],   "T": [1.,0.,0.]},
    "Zoom In": {    "angle":[0., 0., 0.],   "T": [0.,0.,2.]},
    "Zoom Out": {   "angle":[0., 0., 0.],   "T": [0.,0.,-2.]},
    "Anti Clockwise (ACW)": {        "angle": [0., 0., -1.],  "T":[0., 0., 0.]},
    "ClockWise (CW)": {         "angle": [0., 0., 1.], "T":[0., 0., 0.]},
}


def process_pose_params(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu'):

    def get_relative_pose(cam_params):
        """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
        """
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    """Modified from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)

    K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
    c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
    c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
    return plucker_embedding

class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        c2w_mat = np.array(entry[7:]).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        indexing='ij'
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def get_camera_motion(angle, T, speed, n=81):
    def compute_R_form_rad_angle(angles):
        theta_x, theta_y, theta_z = angles
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
    RT = []
    for i in range(n):
        _angle = (i/n)*speed*(CAMERA_DICT["base_angle"])*angle
        R = compute_R_form_rad_angle(_angle)
        _T=(i/n)*speed*(CAMERA_DICT["base_T_norm"])*(T.reshape(3,1))
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT

class WanCameraEmbedding(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanCameraEmbedding",
            category="model/conditioning/wan/camera",
            inputs=[
                io.Combo.Input(
                    "camera_pose",
                    options=[
                        "Static",
                        "Pan Up",
                        "Pan Down",
                        "Pan Left",
                        "Pan Right",
                        "Zoom In",
                        "Zoom Out",
                        "Anti Clockwise (ACW)",
                        "ClockWise (CW)",
                    ],
                    default="Static",
                ),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Float.Input("speed", default=1.0, min=0, max=10.0, step=0.1, optional=True),
                io.Float.Input("fx", default=0.5, min=0, max=1, step=0.000000001, optional=True, advanced=True),
                io.Float.Input("fy", default=0.5, min=0, max=1, step=0.000000001, optional=True, advanced=True),
                io.Float.Input("cx", default=0.5, min=0, max=1, step=0.01, optional=True, advanced=True),
                io.Float.Input("cy", default=0.5, min=0, max=1, step=0.01, optional=True, advanced=True),
            ],
            outputs=[
                io.WanCameraEmbedding.Output(display_name="camera_embedding"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="length"),
            ],
        )

    @classmethod
    def execute(cls, camera_pose, width, height, length, speed=1.0, fx=0.5, fy=0.5, cx=0.5, cy=0.5) -> io.NodeOutput:
        """
        Use Camera trajectory as extrinsic parameters to calculate Plücker embeddings (Sitzmannet al., 2021)
        Adapted from https://github.com/aigc-apps/VideoX-Fun/blob/main/comfyui/comfyui_nodes.py
        """
        motion_list = [camera_pose]
        speed = speed
        angle = np.array(CAMERA_DICT[motion_list[0]]["angle"])
        T = np.array(CAMERA_DICT[motion_list[0]]["T"])
        RT = get_camera_motion(angle, T, speed, length)

        trajs=[]
        for cp in RT.tolist():
            traj=[fx,fy,cx,cy,0,0]
            traj.extend(cp[0])
            traj.extend(cp[1])
            traj.extend(cp[2])
            traj.extend([0,0,0,1])
            trajs.append(traj)

        cam_params = np.array([[float(x) for x in pose] for pose in trajs])
        cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
        control_camera_video = process_pose_params(cam_params, width=width, height=height)
        control_camera_video = control_camera_video.permute([3, 0, 1, 2]).unsqueeze(0).to(device=comfy.model_management.intermediate_device())

        control_camera_video = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)

        # Reshape, transpose, and view into desired shape
        b, f, c, h, w = control_camera_video.shape
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_video = control_camera_video.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)

        return io.NodeOutput(control_camera_video, width, height, length)


class CameraTrajectoryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanCameraEmbedding,
        ]

async def comfy_entrypoint() -> CameraTrajectoryExtension:
    return CameraTrajectoryExtension()
