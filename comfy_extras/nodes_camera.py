import math

import torch
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, IO
from comfy_extras.nodes_save_3d import get_mesh_batch_item


def _quat_to_mat(q):
    # q: (N, 4) wxyz, normalized -> (N, 3, 3)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _quat_mul(a, b):
    # Hamilton product a (x) b, wxyz.
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _mat_to_quat(m):
    # Rotation matrix (..., 3, 3) -> quaternion (..., 4) wxyz. Batched; builds the four candidate quaternions
    # and keeps the one with the largest component (numerically stable across all rotations).
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    m21, m12 = m[..., 2, 1], m[..., 1, 2]
    m02, m20 = m[..., 0, 2], m[..., 2, 0]
    m10, m01 = m[..., 1, 0], m[..., 0, 1]
    q2 = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                      1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)   # 4 * (w^2, x^2, y^2, z^2)
    cand = torch.stack([
        torch.stack([q2[..., 0], m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q2[..., 1], m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q2[..., 2], m12 + m21], -1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, q2[..., 3]], -1),
    ], -2)                                                            # (...,4,4) candidates, rows = wxyz
    sel = q2.argmax(-1)
    q = torch.gather(cand, -2, sel[..., None, None].expand(sel.shape + (1, 4)))[..., 0, :]
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _lookat_quat_wxyz(position, target, dev):
    # three.js lookAt in world frame: camera local +Z = (eye - target), up = world +Y. Returns wxyz.
    z = position - target
    z = z / z.norm().clamp_min(1e-8)
    up0 = torch.tensor([0.0, 1.0, 0.0], device=dev)
    if z.dot(up0).abs() > 0.999:                                # looking straight up/down
        up0 = torch.tensor([0.0, 0.0, 1.0], device=dev)
    x = torch.linalg.cross(up0, z)
    x = x / x.norm().clamp_min(1e-8)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=1)                           # columns = camera world axes
    return _mat_to_quat(R[None])[0]


def _lookat_camera_info(position, target, fov, dev, zoom=1.0, camera_type="perspective", roll=0.0):
    # Build a camera_info from a world-space (right-handed, Y-up) eye + look-at target; up = world +Y.
    pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)
    q = _lookat_quat_wxyz(pos, tgt, dev)
    if roll:                                                    # roll about the view axis (camera local Z)
        a = math.radians(roll)
        qz = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], device=dev)
        q = _quat_mul(q[None], qz[None])[0]
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {"position": xyz(pos), "target": xyz(tgt),
            "quaternion": {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])},
            **({"useCustomUp": True} if roll else {}),
            "fov": float(fov), "cameraType": str(camera_type), "zoom": float(zoom)}


def _quat_camera_info(position, quat_xyzw, fov, dev, zoom=1.0, camera_type="perspective"):
    # camera_info from an explicit world position + camera-rotation quaternion (three.js: looks down local -Z).
    pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
    qx, qy, qz, qw = (float(c) for c in quat_xyzw)
    qwxyz = torch.tensor([qw, qx, qy, qz], dtype=torch.float32, device=dev)
    qwxyz = qwxyz / qwxyz.norm().clamp_min(1e-8)
    R = _quat_to_mat(qwxyz[None])[0]
    tgt = pos - R[:, 2]                                         # look one unit down local -Z
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {"position": xyz(pos), "target": xyz(tgt),
            "quaternion": {"x": float(qwxyz[1]), "y": float(qwxyz[2]), "z": float(qwxyz[3]), "w": float(qwxyz[0])},
            "useCustomUp": True,
            "fov": float(fov), "cameraType": str(camera_type), "zoom": float(zoom)}


def _xyz(d):
    return [float(d.get(k, 0.0)) for k in "xyz"]


def _rotate_camera_info(camera_info, axis, angle_deg):
    # Rigidly rotate a camera_info about the unit world `axis` through its target: position, up and quaternion
    # turn together, so a turntable or a sweep over the top keeps the camera's orientation continuous.
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    kx, ky, kz = axis

    def rot(v):
        d = (kx * v[0] + ky * v[1] + kz * v[2]) * (1.0 - c)
        return [v[0] * c + (ky * v[2] - kz * v[1]) * s + kx * d,
                v[1] * c + (kz * v[0] - kx * v[2]) * s + ky * d,
                v[2] * c + (kx * v[1] - ky * v[0]) * s + kz * d]
    tgt = _xyz(camera_info.get("target", {}))
    rel = rot([p - t for p, t in zip(_xyz(camera_info.get("position", {})), tgt)])
    up = rot(_xyz(camera_info.get("up", {"y": 1.0})))
    out = {**camera_info, "position": dict(zip("xyz", (r + t for r, t in zip(rel, tgt)))), "up": dict(zip("xyz", up))}
    q = camera_info.get("quaternion")
    if q:                                         # world-frame rotation (cos a/2, axis sin a/2) applied first
        aw, ax, ay, az = math.cos(a / 2), kx * math.sin(a / 2), ky * math.sin(a / 2), kz * math.sin(a / 2)
        bw, bx, by, bz = (float(q.get(k, 1.0 if k == "w" else 0.0)) for k in "wxyz")
        out["quaternion"] = {"x": aw * bx + ax * bw + ay * bz - az * by, "y": aw * by - ax * bz + ay * bw + az * bx,
                             "z": aw * bz + ax * by - ay * bx + az * bw, "w": aw * bw - ax * bx - ay * by - az * bz}
    return out


def _mesh_frame_to_world(points, mesh):
    """Points in the mesh's frame -> world. The frame's origin is the mesh's bounding-box centre and its unit the
    diameter of the bounding sphere about that centre: one distance frames the mesh from every side of an orbit,
    poles included, and it is how the UltraTex ComfyUI wrapper normalizes meshes. Non-finite vertices are ignored."""
    vertices = get_mesh_batch_item(mesh, 0)[0].float()
    vertices = vertices[torch.isfinite(vertices).all(-1)]
    center = (vertices.amin(0) + vertices.amax(0)) * 0.5
    size = max(2.0 * float((vertices - center).norm(dim=-1).amax()), 1e-6)
    return [[c + size * x for c, x in zip(center.tolist(), point)] for point in points]


def camera_list(camera_info):
    """A camera_info is one camera or, from Create Camera Orbit / Batch Cameras, a list of them."""
    return camera_info if isinstance(camera_info, list) else [camera_info]


class CreateCameraInfo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreateCameraInfo",
            display_name="Create Camera Info",
            search_aliases=["camera position", "make camera info", "orbit camera", "look at camera"],
            category="3d",
            description="Build a camera_info. "
                        "Mode 'orbit' aims with yaw/pitch/distance around the target; "
                        "'look_at' places the camera at world position. Coordinates are the viewer's world space (right-handed,Y-up), "
                        "or with a mesh connected that mesh's frame, so the same settings frame any mesh.",
            inputs=[
                IO.DynamicCombo.Input("mode", options=[
                    IO.DynamicCombo.Option("orbit", [
                        IO.Float.Input("yaw", default=35.0, min=-360.0, max=360.0, step=1.0),
                        IO.Float.Input("pitch", default=30.0, min=-89.0, max=89.0, step=1.0),
                        IO.Float.Input("distance", default=4.0, min=0.01, max=1000.0, step=0.01,
                                       tooltip="Camera distance from the target."),
                    ]),
                    IO.DynamicCombo.Option("look_at", [
                        IO.Float.Input("position_x", default=4.0, min=-1000.0, max=1000.0, step=0.01,
                                       tooltip="Camera position in world space (right-handed, Y-up)."),
                        IO.Float.Input("position_y", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                    ]),
                    IO.DynamicCombo.Option("quaternion", [
                        IO.Float.Input("position_x", default=4.0, min=-1000.0, max=1000.0, step=0.01,
                                       tooltip="Camera position in world space (right-handed, Y-up)."),
                        IO.Float.Input("position_y", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=4.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("quat_x", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_y", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_z", default=0.0, min=-1.0, max=1.0, step=0.001),
                        IO.Float.Input("quat_w", default=1.0, min=-1.0, max=1.0, step=0.001,
                                       tooltip="Camera world-rotation quaternion (three.js: looks down local -Z). Normalized for you."),
                    ]),
                ], tooltip="How to define the camera: orbit angles, an explicit position, or a position + quaternion."),
                IO.Float.Input("target_x", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True,
                               tooltip="Look-at point (orbit pivot / aim). In orbit mode, move it to pan/translate the "
                                       "whole camera. Ignored in quaternion mode. Defaults to the origin."),
                IO.Float.Input("target_y", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True),
                IO.Float.Input("target_z", default=0.0, min=-1000.0, max=1000.0, step=0.01, advanced=True),
                IO.Float.Input("roll", default=0.0, min=-180.0, max=180.0, step=1.0,
                               tooltip="Camera roll about the view axis, degrees."),
                IO.Float.Input("fov", default=35.0, min=1.0, max=120.0, step=1.0,
                               tooltip="Vertical field of view in degrees."),
                IO.Float.Input("zoom", default=1.0, min=0.01, max=100.0, step=0.01,
                               tooltip="Digital zoom (focal-length multiplier). >1 zooms in without moving the camera."),
                IO.Combo.Input("camera_type", options=["perspective", "orthographic"],
                               tooltip="Projection used by Render Splat and Render Mesh: perspective (foreshortening) or orthographic (parallel)."),
                IO.CameraInfoState.Input("camera_info_state", optional=True),
                IO.Mesh.Input("mesh", optional=True,
                              tooltip="Frame this mesh: positions, target and distance count from its bounding-box centre in "
                                      "units of its bounding-sphere diameter, so the same settings frame any mesh."),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info")],
        )

    @classmethod
    def execute(cls, mode, target_x, target_y, target_z, roll, fov, zoom=1.0, camera_type="perspective", camera_info_state=None, mesh=None) -> IO.NodeOutput:
        dev = comfy.model_management.get_torch_device()
        kind = mode["mode"]
        if kind == "quaternion":  # explicit world position + camera rotation
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
            quat = [mode["quat_x"], mode["quat_y"], mode["quat_z"], mode["quat_w"]]
            camera = _quat_camera_info(position, quat, fov, dev, zoom=zoom, camera_type=camera_type)
            if mesh is not None:  # the target one unit ahead, which orbits pivot on, is in the mesh's units too
                position, target = _mesh_frame_to_world([_xyz(camera["position"]), _xyz(camera["target"])], mesh)
                camera = {**camera, "position": dict(zip("xyz", position)), "target": dict(zip("xyz", target))}
            return IO.NodeOutput(camera)
        target = [target_x, target_y, target_z]  # orbit pivot / aim; move it to pan the whole camera
        if kind == "orbit":  # yaw/pitch/distance about the target (world Y-up)
            y, p = math.radians(mode["yaw"]), math.radians(mode["pitch"])
            cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
            d = mode["distance"]
            position = [target_x + d * cp * sy, target_y + d * sp, target_z + d * cp * cy]
        else:  # look_at: explicit world-space camera position
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
        if mesh is not None:
            position, target = _mesh_frame_to_world([position, target], mesh)
        return IO.NodeOutput(_lookat_camera_info(position, target, fov, dev, zoom=zoom, camera_type=camera_type, roll=roll))


class CreateCameraOrbit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreateCameraOrbit",
            display_name="Create Camera Orbit",
            search_aliases=["camera rig", "multiview cameras", "turntable cameras", "spiral cameras", "camera views"],
            category="3d",
            description="Several cameras around the first camera's target: a turntable by default, tilted if wanted, plus "
                        "views straight down and up. Render Mesh renders one image per camera; Bake Texture From Views "
                        "projects them back.",
            inputs=[
                IO.Load3DCamera.Input("camera_info", tooltip="First camera of the orbit. A camera list orbits every camera in it."),
                IO.Int.Input("count", default=4, min=1, max=64, tooltip="Cameras evenly spaced along the sweep."),
                IO.Float.Input("yaw", default=360.0, min=-720.0, max=720.0, step=1.0,
                               tooltip="Total turn around the target, in Create Camera Info's yaw direction. Whole turns stop "
                                       "one step short of the first camera; other sweeps end on the last camera."),
                IO.Float.Input("pitch", default=0.0, min=-720.0, max=720.0, step=1.0,
                               tooltip="Total tilt over the target, positive rising like Create Camera Info's pitch; past 90 "
                                       "the cameras pass over the top. Whole turns loop like yaw."),
                IO.Combo.Input("poles", options=["none", "top", "bottom", "top and bottom"],
                               tooltip="Views straight down and/or up at the target, added after the orbit. Their image top "
                                       "points away from the first camera."),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info")],
        )

    @classmethod
    def execute(cls, camera_info, count, yaw, pitch, poles) -> IO.NodeOutput:
        step = lambda angle: angle / count if angle % 360.0 == 0.0 else angle / max(count - 1, 1)
        yaw_step, pitch_step = step(yaw), step(pitch)
        rig = []
        for base in camera_list(camera_info):
            pos, tgt = _xyz(base.get("position", {})), _xyz(base.get("target", {}))
            fx, fy, fz = (t - p for p, t in zip(pos, tgt))
            ux, uy, uz = _xyz(base.get("up", {"y": 1.0}))
            if abs(fx * ux + fy * uy + fz * uz) >= 0.99 * math.hypot(fx, fy, fz) * math.hypot(ux, uy, uz):
                # looking along up: Render Mesh's +Z up fallback, +Y when the view is along Z too
                ux, uy, uz = (0.0, 0.0, 1.0) if abs(fz) < 0.99 * math.hypot(fx, fy, fz) else (0.0, 1.0, 0.0)
            ax, ay, az = uy * fz - uz * fy, uz * fx - ux * fz, ux * fy - uy * fx   # up x forward: tilting about it rises
            n = math.hypot(ax, ay, az)
            if n == 0.0:
                raise ValueError("Create Camera Orbit: the camera's position equals its target, so it has no view direction.")
            for i in range(count):
                cam = _rotate_camera_info(base, (ax / n, ay / n, az / n), pitch_step * i)
                rig.append(_rotate_camera_info(cam, (0.0, 1.0, 0.0), yaw_step * i))
            dist = math.dist(pos, tgt)
            back_x, back_z = tgt[0] - pos[0], tgt[2] - pos[2]  # horizontal, away from the first camera
            if back_x == 0.0 and back_z == 0.0:                # looking straight down or up: its up is the image top
                back_x, back_z = ux, uz
            n = math.hypot(back_x, back_z)
            for sign, pole in ((1.0, "top"), (-1.0, "bottom")):
                if poles in (pole, "top and bottom"):
                    cam = {k: v for k, v in base.items() if k not in ("quaternion", "useCustomUp")}
                    cam["position"] = {"x": tgt[0], "y": tgt[1] + sign * dist, "z": tgt[2]}
                    cam["up"] = {"x": back_x / n, "y": 0.0, "z": back_z / n}
                    rig.append(cam)
        return IO.NodeOutput(rig)


class BatchCameras(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BatchCameras",
            display_name="Batch Cameras",
            search_aliases=["combine cameras", "merge cameras", "join camera rigs", "camera list"],
            category="3d",
            description="Join cameras and orbits into one camera list, in input order.",
            inputs=[
                IO.Autogrow.Input("cameras", template=IO.Autogrow.TemplatePrefix(
                    IO.Load3DCamera.Input("camera_info"), prefix="camera_info", min=1, max=50)),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info")],
        )

    @classmethod
    def execute(cls, cameras: IO.Autogrow.Type) -> IO.NodeOutput:
        return IO.NodeOutput([cam for c in cameras.values() for cam in camera_list(c)])


class CameraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [CreateCameraInfo, CreateCameraOrbit, BatchCameras]


async def comfy_entrypoint() -> CameraExtension:
    return CameraExtension()
