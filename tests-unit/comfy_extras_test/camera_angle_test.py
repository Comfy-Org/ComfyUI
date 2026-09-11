import math

import pytest

from comfy_extras.nodes_camera_angle import (
    CameraAngle,
    MAX_DISTANCE,
    MIN_DISTANCE,
    SUBJECT_CENTER,
    build_camera_info,
    describe_camera_angle,
    horizontal_term,
    zoom_to_distance,
)


@pytest.mark.parametrize(
    "angle, term",
    [
        (0, "front view"),
        (22, "front view"),
        (23, "front-right quarter view"),
        (90, "right side view"),
        (135, "back-right quarter view"),
        (180, "back view"),
        (225, "back-left quarter view"),
        (270, "left side view"),
        (315, "front-left quarter view"),
        (338, "front view"),
        (360, "front view"),
    ],
)
def test_horizontal_term_buckets_by_45_degree_sector(angle, term):
    assert horizontal_term(angle) == term


@pytest.mark.parametrize(
    "vertical, zoom, expected",
    [
        (-30, 0, "front view low-angle shot wide shot"),
        (-15, 2, "front view eye-level shot medium shot"),
        (15, 6, "front view elevated shot close-up"),
        (45, 10, "front view high-angle shot close-up"),
    ],
)
def test_describe_camera_angle_joins_vertical_and_distance_terms(vertical, zoom, expected):
    assert describe_camera_angle(0, vertical, zoom) == expected


def test_zoom_maps_linearly_between_wide_and_close_distances():
    assert zoom_to_distance(0) == MAX_DISTANCE
    assert zoom_to_distance(10) == MIN_DISTANCE
    assert zoom_to_distance(5) == pytest.approx((MAX_DISTANCE + MIN_DISTANCE) / 2)


def test_front_view_places_camera_on_positive_z_looking_at_subject_centre():
    info = build_camera_info(0, 0, 0)
    assert info["position"] == pytest.approx({"x": 0.0, "y": SUBJECT_CENTER[1], "z": MAX_DISTANCE})
    assert info["target"] == {"x": 0.0, "y": SUBJECT_CENTER[1], "z": 0.0}
    assert info["cameraType"] == "perspective"
    assert info["zoom"] == 1.0


def test_orbit_position_follows_yaw_and_pitch():
    info = build_camera_info(90, 30, 10)
    position = info["position"]
    assert position["x"] == pytest.approx(MIN_DISTANCE * math.cos(math.radians(30)))
    assert position["y"] == pytest.approx(SUBJECT_CENTER[1] + MIN_DISTANCE * math.sin(math.radians(30)))
    assert position["z"] == pytest.approx(0.0, abs=1e-9)


def test_execute_clamps_inputs_and_returns_camera_info_and_prompt():
    result = CameraAngle.execute(horizontal_angle=400, vertical_angle=-90, zoom=99)
    camera_info, prompt = result.args
    assert prompt == "front view low-angle shot close-up"
    assert camera_info == build_camera_info(360, -30, 10)
