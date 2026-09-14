import math

from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

# Scene layout shared with the frontend preview (src/extensions/core/cameraAngle/types.ts):
# a unit subject cube sits at the world origin, where downstream 3D nodes centre their models.
SUBJECT_CENTER = (0.0, 0.0, 0.0)
CAMERA_FOV = 35.0
SUBJECT_DISTANCE = 6.0
MIN_ZOOM_FACTOR = 1.0
MAX_ZOOM_FACTOR = 1.875

HORIZONTAL_MIN, HORIZONTAL_MAX = 0, 360
VERTICAL_MIN, VERTICAL_MAX = -30, 60
ZOOM_MIN, ZOOM_MAX = 0.0, 10.0

HORIZONTAL_TERMS = (
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
)
VERTICAL_TERMS = ((-15, "low-angle shot"), (15, "eye-level shot"), (45, "elevated shot"), (None, "high-angle shot"))
DISTANCE_TERMS = ((2, "wide shot"), (6, "medium shot"), (None, "close-up"))


def _bucket(value: float, terms) -> str:
    for upper, term in terms:
        if upper is None or value < upper:
            return term
    return terms[-1][1]


def horizontal_term(angle: float) -> str:
    sector = int(((angle % 360) + 22.5) // 45) % len(HORIZONTAL_TERMS)
    return HORIZONTAL_TERMS[sector]


def vertical_term(angle: float) -> str:
    return _bucket(angle, VERTICAL_TERMS)


def distance_term(zoom: float) -> str:
    return _bucket(zoom, DISTANCE_TERMS)


def describe_camera_angle(horizontal_angle: float, vertical_angle: float, zoom: float) -> str:
    return f"{horizontal_term(horizontal_angle)} {vertical_term(vertical_angle)} {distance_term(zoom)}"


def zoom_to_factor(zoom: float) -> float:
    return MIN_ZOOM_FACTOR + (MAX_ZOOM_FACTOR - MIN_ZOOM_FACTOR) * (zoom / ZOOM_MAX)


def build_camera_info(horizontal_angle: float, vertical_angle: float, zoom: float) -> dict:
    yaw, pitch = math.radians(horizontal_angle), math.radians(vertical_angle)
    cx, cy, cz = SUBJECT_CENTER
    return {
        "position": {
            "x": cx + SUBJECT_DISTANCE * math.cos(pitch) * math.sin(yaw),
            "y": cy + SUBJECT_DISTANCE * math.sin(pitch),
            "z": cz + SUBJECT_DISTANCE * math.cos(pitch) * math.cos(yaw),
        },
        "target": {"x": cx, "y": cy, "z": cz},
        "zoom": zoom_to_factor(zoom),
        "cameraType": "perspective",
        "fov": CAMERA_FOV,
    }


class CameraAngle(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CameraAngle",
            display_name="Camera Angle",
            search_aliases=["multi angle", "camera view", "orbit camera", "shot angle", "camera prompt"],
            category="3d",
            description="Pick a camera angle around a subject with a 3D preview. Outputs a camera_info "
                        "for 3D nodes and a plain-English shot description for prompts.",
            inputs=[
                IO.Int.Input("horizontal_angle", default=0, min=HORIZONTAL_MIN, max=HORIZONTAL_MAX, step=1,
                             tooltip="Azimuth around the subject in degrees. 0 is the front, 90 the right side, 180 the back."),
                IO.Int.Input("vertical_angle", default=0, min=VERTICAL_MIN, max=VERTICAL_MAX, step=1,
                             tooltip="Elevation in degrees. Negative looks up from below, positive looks down from above."),
                IO.Float.Input("zoom", default=5.0, min=ZOOM_MIN, max=ZOOM_MAX, step=0.1,
                               tooltip="Lens zoom on the subject: 0 is a wide shot, 10 a close-up. Carried into camera_info.zoom."),
                IO.Image.Input("image", optional=True,
                               tooltip="Optional reference image shown on the front of the subject cube in the 3D preview."),
                IO.String.Input("view", default="", optional=True, socketless=True,
                                extra_dict={"widgetType": "CAMERA_ANGLE_VIEW"}),
            ],
            outputs=[
                IO.Load3DCamera.Output(display_name="camera_info"),
                IO.String.Output(display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, horizontal_angle, vertical_angle, zoom, image=None, view=None) -> IO.NodeOutput:
        horizontal_angle = max(HORIZONTAL_MIN, min(HORIZONTAL_MAX, int(horizontal_angle)))
        vertical_angle = max(VERTICAL_MIN, min(VERTICAL_MAX, int(vertical_angle)))
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, float(zoom)))
        return IO.NodeOutput(
            build_camera_info(horizontal_angle, vertical_angle, zoom),
            describe_camera_angle(horizontal_angle, vertical_angle, zoom),
        )


class CameraAngleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [CameraAngle]


async def comfy_entrypoint() -> CameraAngleExtension:
    return CameraAngleExtension()
