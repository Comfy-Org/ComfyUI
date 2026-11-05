import math

from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, IO
from comfy_extras.nodes_gaussian_splat import _lookat_camera_info, _quat_camera_info, _rotate_camera_info, _xyz


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
                        "'look_at' places the camera at world position. Coordinates are the viewer's world space (right-handed,Y-up).",
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
                               tooltip="Projection used by Render Splat: perspective (foreshortening) or orthographic (parallel)."),
                IO.CameraInfoState.Input("camera_info_state", optional=True),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info")],
        )

    @classmethod
    def execute(cls, mode, target_x, target_y, target_z, roll, fov, zoom=1.0, camera_type="perspective", camera_info_state=None) -> IO.NodeOutput:
        dev = comfy.model_management.get_torch_device()
        kind = mode["mode"]
        if kind == "quaternion":  # explicit world position + camera rotation
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
            quat = [mode["quat_x"], mode["quat_y"], mode["quat_z"], mode["quat_w"]]
            return IO.NodeOutput(_quat_camera_info(position, quat, fov, dev, zoom=zoom, camera_type=camera_type))
        target = [target_x, target_y, target_z]  # orbit pivot / aim; move it to pan the whole camera
        if kind == "orbit":  # yaw/pitch/distance about the target (world Y-up)
            y, p = math.radians(mode["yaw"]), math.radians(mode["pitch"])
            cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
            d = mode["distance"]
            position = [target_x + d * cp * sy, target_y + d * sp, target_z + d * cp * cy]
        else:  # look_at: explicit world-space camera position
            position = [mode["position_x"], mode["position_y"], mode["position_z"]]
        return IO.NodeOutput(_lookat_camera_info(position, target, fov, dev, zoom=zoom, camera_type=camera_type, roll=roll))


class CreateCameraOrbit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CreateCameraOrbit",
            display_name="Create Camera Orbit",
            search_aliases=["camera rig", "multiview cameras", "turntable cameras", "spiral cameras", "camera views"],
            category="3d",
            description="Expand a camera_info into several cameras around its target, swept in yaw and pitch from the camera "
                        "(by default a turntable), then optional views straight down and up. Render Mesh renders one view per "
                        "camera and Bake Texture From Views projects them back.",
            inputs=[
                IO.Load3DCamera.Input("camera_info", tooltip="First camera of the orbit. The cameras of another Create Camera Orbit "
                                                             "are each expanded."),
                IO.Int.Input("count", default=4, min=1, max=64, tooltip="Cameras evenly spaced along the sweep."),
                IO.Float.Input("yaw", default=360.0, min=-720.0, max=720.0, step=1.0,
                               tooltip="Total turn about world +Y around the camera target, in Create Camera Info's yaw "
                                       "direction. Whole turns end one step short so they loop; other sweeps end on the last camera."),
                IO.Float.Input("pitch", default=0.0, min=-720.0, max=720.0, step=1.0,
                               tooltip="Total tilt about the camera's horizontal axis through the target; positive rises like "
                                       "Create Camera Info's pitch and goes over the top past 90. Whole turns loop like yaw."),
                IO.Combo.Input("poles", options=["none", "top", "bottom", "top and bottom"],
                               tooltip="Views straight down and/or up onto the target after the orbit, with the first "
                                       "camera's back at the image top."),
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
            description="Join cameras and camera rigs into one camera_info, in input order",
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
