import math

from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, IO
from comfy_extras.nodes_gaussian_splat import _lookat_camera_info, _quat_camera_info


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


class CameraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [CreateCameraInfo]


async def comfy_entrypoint() -> CameraExtension:
    return CameraExtension()
