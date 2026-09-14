import math
import pytest
import torch
from unittest.mock import MagicMock, patch

with patch.dict("sys.modules", {"server": MagicMock()}):
    from comfy_extras.nodes_light_3d import (
        CreateLightInfo,
        RenderLight,
        _clean_light,
        _hex_to_rgb01,
        _linear_to_srgb,
        _normalize_light,
        _srgb_to_linear,
    )


def directional(**overrides):
    light = {
        "type": "directional",
        "color": "#ffffff",
        "intensity": 1.5,
        "position": {"x": 0.0, "y": 7.0, "z": 7.0},
        "target": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    light.update(overrides)
    return light


class TestHexToRgb01:
    def test_six_digit(self):
        assert _hex_to_rgb01("#ff8000") == pytest.approx([1.0, 128 / 255, 0.0])

    def test_three_digit(self):
        assert _hex_to_rgb01("#fff") == pytest.approx([1.0, 1.0, 1.0])

    def test_invalid_falls_back_to_white(self):
        assert _hex_to_rgb01("not-a-color") == [1.0, 1.0, 1.0]


class TestCleanLight:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported light type"):
            _clean_light({"type": "laser"})

    def test_directional_drops_range_and_cones(self):
        cleaned = _clean_light(directional(range=5.0, innerConeAngle=10.0))
        assert cleaned["type"] == "directional"
        assert "range" not in cleaned
        assert "innerConeAngle" not in cleaned
        assert cleaned["target"] == {"x": 0.0, "y": 0.0, "z": 0.0}

    def test_spot_inner_cone_is_clamped_to_outer(self):
        cleaned = _clean_light({"type": "spot", "innerConeAngle": 60.0, "outerConeAngle": 45.0})
        assert cleaned["innerConeAngle"] == 45.0

    def test_radius_and_cast_shadow_round_trip(self):
        cleaned = _clean_light(directional(radius=0.5, castShadow=False))
        assert cleaned["radius"] == 0.5
        assert cleaned["castShadow"] is False
        assert "radius" not in _clean_light(directional(radius=0))
        assert "castShadow" not in _clean_light(directional())

    def test_point_drops_target_and_keeps_positive_range(self):
        cleaned = _clean_light({
            "type": "point",
            "position": {"x": 1, "y": 2, "z": 3},
            "range": 8.0,
            "target": {"x": 9, "y": 9, "z": 9},
        })
        assert "target" not in cleaned
        assert cleaned["range"] == 8.0
        assert cleaned["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_point_zero_range_is_omitted(self):
        cleaned = _clean_light({"type": "point", "range": 0.0})
        assert "range" not in cleaned

    def test_spot_keeps_cone_angles(self):
        cleaned = _clean_light({
            "type": "spot",
            "innerConeAngle": 15,
            "outerConeAngle": 25,
        })
        assert cleaned["innerConeAngle"] == 15.0
        assert cleaned["outerConeAngle"] == 25.0

    def test_defaults_fill_missing_fields(self):
        cleaned = _clean_light({"type": "directional"})
        assert cleaned["color"] == "#ffffff"
        assert cleaned["intensity"] == 1.0
        assert cleaned["position"] == {"x": 0.0, "y": 1.8, "z": 1.8}

    def test_negative_intensity_clamps_to_zero(self):
        assert _clean_light(directional(intensity=-3))["intensity"] == 0.0


class TestCreateLightInfo:
    def test_outputs_cleaned_editor_lights(self):
        out = CreateLightInfo.execute(editor_state=[
            directional(),
            {"type": "spot", "position": {"x": 0, "y": 4, "z": 0}},
        ])
        result = out.result[0]
        assert [light["type"] for light in result] == ["directional", "spot"]
        assert result[1]["outerConeAngle"] == 45.0

    def test_empty_editor_outputs_empty_list(self):
        assert CreateLightInfo.execute(editor_state=None).result[0] == []

    def test_non_dict_entries_are_skipped(self):
        out = CreateLightInfo.execute(editor_state=[None, "junk", directional()])
        assert len(out.result[0]) == 1


class TestNormalizeLight:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported light type"):
            _normalize_light({"type": "laser"})

    def test_defaults_fill_missing_fields(self):
        light = _normalize_light({"type": "directional"})
        assert light["color"] == [1.0, 1.0, 1.0]
        assert light["intensity"] == 1.0
        assert light["target"] == [0.0, 0.0, 0.0]


class TestRenderLight:
    @staticmethod
    def _render(lights, **kwargs):
        defaults = dict(width=96, height=96, ambient=0.2)
        defaults.update(kwargs)
        return RenderLight.execute(light_info=lights, **defaults)

    def test_output_shape_and_range(self):
        lights = CreateLightInfo.execute(editor_state=[directional()]).result[0]
        image, _ = self._render(lights).result
        assert image.shape == (1, 96, 96, 3)
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

    def test_lambert_brdf_divides_by_pi(self):
        lights = CreateLightInfo.execute(editor_state=[directional(intensity=1.0)]).result[0]
        image, _ = self._render(lights, ambient=0.0).result
        expected = _linear_to_srgb(torch.tensor(_srgb_to_linear(0.8) / math.pi))
        assert abs(float(image.max()) - float(expected)) < 0.01

    def test_near_field_falloff_is_clamped(self):
        lights = CreateLightInfo.execute(editor_state=[
            {"type": "point", "color": "#ffffff", "intensity": 0.01,
             "position": {"x": 3.0, "y": -0.999, "z": 0.0}}]).result[0]
        image, _ = self._render(lights, ambient=0.0).result
        expected = _linear_to_srgb(torch.tensor(_srgb_to_linear(0.541) * 0.01 * 100.0 / math.pi))
        assert float(image.max()) < float(expected) + 0.01

    def test_light_direction_changes_the_image(self):
        def render_at(x):
            lights = [directional(position={"x": x, "y": 7.0, "z": 7.0})]
            return self._render(lights).result[0]

        assert not torch.allclose(render_at(-8.0), render_at(8.0))

    def test_multiple_lights_accumulate(self):
        one = self._render([directional(intensity=0.5)]).result[0]
        two = self._render([directional(intensity=0.5),
                            directional(intensity=0.5)]).result[0]
        assert float(two.sum()) > float(one.sum())

    def test_no_lights_renders_ambient_only(self):
        image, per_light = self._render([]).result
        assert float(image.std()) < 0.2
        assert float(image.max()) <= 0.55
        assert per_light.shape == image.shape

    @staticmethod
    def _shadow_rows(image):
        return image[0, -12:, :, 0]

    def test_light_radius_softens_the_shadow_edge(self):
        behind = dict(position={"x": 0.0, "y": 3.0, "z": -7.0}, intensity=1.0)
        hard, _ = self._render([directional(**behind)], ambient=0.0).result
        soft, _ = self._render([directional(radius=30.0, **behind)], ambient=0.0).result
        hard_levels = self._shadow_rows(hard).round(decimals=3).unique().numel()
        soft_levels = self._shadow_rows(soft).round(decimals=3).unique().numel()
        assert hard_levels <= 3
        assert soft_levels > hard_levels + 5

    def test_cast_shadow_false_disables_the_shadow(self):
        behind = dict(position={"x": 0.0, "y": 3.0, "z": -7.0}, intensity=1.0)
        shadowed, _ = self._render([directional(**behind)], ambient=0.0).result
        unshadowed, _ = self._render([directional(castShadow=False, **behind)], ambient=0.0).result
        assert float(self._shadow_rows(shadowed).min()) == 0.0
        assert float(self._shadow_rows(unshadowed).min()) > 0.1

    def test_per_light_isolates_each_light(self):
        left = directional(position={"x": -7.0, "y": 7.0, "z": 0.0}, intensity=1.0)
        right = directional(position={"x": 7.0, "y": 7.0, "z": 0.0}, intensity=1.0)
        _, per_light = self._render([left, right], ambient=0.2).result
        left_alone, _ = self._render([left], ambient=0.0).result
        assert per_light.shape == (2, 96, 96, 3)
        assert torch.allclose(per_light[0], left_alone[0])
        assert not torch.allclose(per_light[0], per_light[1])

    def test_hemisphere_fill_lights_up_facing_surfaces_from_the_sky(self):
        floor_lit, _ = self._render([], ambient=1.0, sky_color="#ffffff", ground_color="#000000").result
        floor_dark, _ = self._render([], ambient=1.0, sky_color="#000000", ground_color="#ffffff").result
        floor = (slice(None), slice(-8, None), slice(None), 0)
        assert float(floor_lit[floor].min()) > 0.3
        assert float(floor_dark[floor].max()) < 0.01

    def test_sky_matches_the_floor_color(self):
        image, _ = self._render([], camera_info={
            "position": {"x": 0, "y": 0.5, "z": 8}, "target": {"x": 0, "y": 0.5, "z": 0},
            "fov": 35, "zoom": 1.0, "cameraType": "perspective"}).result
        sky = image[0, :8, :, :]
        assert torch.allclose(sky, torch.full_like(sky, 0x8a / 255), atol=1e-3)

    def test_default_and_explicit_camera_differ(self):
        lights = [directional()]
        default_view = self._render(lights).result[0]
        side_view = self._render(lights, camera_info={
            "position": {"x": 8, "y": 1, "z": 0}, "target": {"x": 0, "y": 0, "z": 0},
            "fov": 35, "zoom": 1.0, "cameraType": "perspective"}).result[0]
        assert not torch.allclose(default_view, side_view)
