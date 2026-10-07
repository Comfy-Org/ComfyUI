import math

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy_extras.nodes_camera import CreateCameraInfo
from comfy_extras.nodes_camera_angle import (
    CAMERA_FOV,
    MAX_ZOOM_FACTOR,
    MIN_ZOOM_FACTOR,
    SUBJECT_CENTER,
    SUBJECT_DISTANCE,
    CameraAngle,
    build_camera_info,
    vertical_term,
)
from comfy_extras.nodes_gaussian_splat import _camera_basis, _lookat_camera_info, _quat_camera_info

DEVICE = torch.device("cpu")
EYE = [0.0, 0.0, 4.0]
ORIGIN = [0.0, 0.0, 0.0]
ROLLED = [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]


@pytest.mark.parametrize("node", [CameraAngle, CreateCameraInfo])
def test_camera_nodes_agree_on_the_camera_info_output_type(node):
    outputs = [o for o in node.define_schema().outputs if o.display_name == "camera_info"]
    assert [o.get_io_type() for o in outputs] == ["LOAD3D_CAMERA"]


def test_camera_angle_keeps_the_inputs_the_frontend_picker_binds_to():
    # The picker is selected by widgetType, writes back through these ids, and hand-duplicates the
    # limits, defaults, steps and scene constants in src/extensions/core/cameraAngle/types.ts, which
    # it uses to place the preview camera. Changing either side alone leaves the node and its 3D
    # preview disagreeing, with nothing negotiating the difference at runtime.
    inputs = {i.id: i for i in CameraAngle.define_schema().inputs}
    assert set(inputs) == {"horizontal_angle", "vertical_angle", "zoom", "image", "view"}
    assert inputs["view"].extra_dict["widgetType"] == "CAMERA_ANGLE_VIEW"
    assert [(inputs[i].default, inputs[i].min, inputs[i].max, inputs[i].step)
            for i in ("horizontal_angle", "vertical_angle", "zoom")] == [
        (0, 0, 360, 1), (0, -30, 60, 1), (5.0, 0.0, 10.0, 0.1),
    ]
    assert (SUBJECT_CENTER, CAMERA_FOV, SUBJECT_DISTANCE, MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR) == \
        ((0.0, 0.0, 0.0), 35.0, 6.0, 1.0, 1.875)


@pytest.mark.parametrize("horizontal, vertical", [(0, 0), (90, 0), (180, 0), (270, 0), (45, 30), (0, -30)])
def test_camera_angle_aims_the_splat_renderer_where_it_was_asked_to(horizontal, vertical):
    # CameraAngle emits no quaternion, so _camera_basis takes its look-at path. The view direction
    # it recovers there has to be the angle the node was given.
    yaw, pitch = math.radians(horizontal), math.radians(vertical)
    _, _, _, _, fwd = _camera_basis(build_camera_info(horizontal, vertical, 5.0), DEVICE)
    assert [float(v) for v in fwd] == pytest.approx(
        [-math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)], abs=1e-5)


def test_camera_angle_always_aims_at_the_subject_centre():
    _, target, _, _, _ = _camera_basis(build_camera_info(45, 30, 5.0), DEVICE)
    assert [float(v) for v in target] == pytest.approx(ORIGIN, abs=1e-6)


@pytest.mark.parametrize("vertical, term", [
    (-30, "low-angle shot"), (-15, "eye-level shot"), (0, "eye-level shot"),
    (15, "elevated shot"), (45, "high-angle shot"), (60, "high-angle shot"),
])
def test_camera_angle_elevation_wording_matches_the_rendered_view(vertical, term):
    # The splat frame is Y-down, so a camera above the subject looks along a positive forward Y.
    _, _, _, _, fwd = _camera_basis(build_camera_info(0, vertical, 5.0), DEVICE)
    assert vertical_term(vertical) == term
    assert (float(fwd[1]) > 0) == (vertical > 0)


def test_camera_info_flags_custom_up_when_world_up_would_be_wrong():
    # The viewer reads the quaternion's up only when useCustomUp is set, so a rolled camera and a
    # supplied rotation that tilts up both say so. A plain look-at leaves it off and keeps world up.
    assert "useCustomUp" not in _lookat_camera_info(EYE, ORIGIN, 35.0, DEVICE)
    assert _lookat_camera_info(EYE, ORIGIN, 35.0, DEVICE, roll=30.0)["useCustomUp"] is True
    assert _quat_camera_info(EYE, ROLLED, 35.0, DEVICE)["useCustomUp"] is True


def test_camera_angle_leaves_the_viewer_up_vector_alone():
    info = build_camera_info(90, 0, 5.0)
    assert "quaternion" not in info
    assert "useCustomUp" not in info
