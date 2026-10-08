"""Blend-mode parity tests for the compositor.

The compositor blends in three places: this numpy module (server-side), the
``layerBlend.frag`` GLSL shader (the live preview the user actually sees), and
anything the frontend adds later. They have diverged before, silently, and the
divergences only show up as "the render does not look like the preview".

``compositor_blend_golden.json`` is the shared contract. This file pins the
numpy implementation to it and additionally spells out, by hand, the boundary
rules that the epsilon guards exist to enforce - so a future refactor of
``safe_div`` cannot quietly re-introduce the old behaviour by regenerating the
fixture.
"""

import json
import os

import numpy as np
import pytest

from comfy_extras.compositor_blend import (
    CHANNEL_BLEND,
    HSL_BLEND,
    EffectiveMode,
    blend_composite,
    blend_pixel,
    resolve_mode,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "compositor_blend_golden.json")

with open(GOLDEN_PATH) as _handle:
    GOLDEN = json.load(_handle)

TOLERANCE = GOLDEN["tolerance"]


def _blend(mode: str, i, l) -> np.ndarray:
    return np.asarray(
        blend_pixel(mode, np.float32(i), np.float32(l)), dtype=np.float64
    ).reshape(3)


def test_golden_covers_every_mode():
    """A new blend mode must arrive with golden values, not silently."""
    assert set(GOLDEN["channel"]) == set(CHANNEL_BLEND)
    assert set(GOLDEN["hsl"]) == set(HSL_BLEND)


@pytest.mark.parametrize("mode", sorted(CHANNEL_BLEND))
def test_channel_modes_match_golden(mode):
    for i, l, expected in GOLDEN["channel"][mode]:
        actual = _blend(mode, [i] * 3, [l] * 3)
        assert actual == pytest.approx([expected] * 3, abs=TOLERANCE), (
            f"{mode}(i={i}, l={l}) -> {actual.tolist()}, golden {expected}"
        )


@pytest.mark.parametrize("mode", sorted(HSL_BLEND))
def test_hsl_modes_match_golden(mode):
    for i, l, expected in GOLDEN["hsl"][mode]:
        actual = _blend(mode, i, l)
        assert actual == pytest.approx(expected, abs=TOLERANCE), (
            f"{mode}(i={i}, l={l}) -> {actual.tolist()}, golden {expected}"
        )


@pytest.mark.parametrize("mode", sorted(set(CHANNEL_BLEND) | set(HSL_BLEND)))
def test_no_mode_produces_nan_or_inf(mode):
    edges = [0.0, 1e-7, 1e-6, 0.5, 1.0 - 1e-7, 1.0]
    for i in edges:
        for l in edges:
            out = _blend(mode, [i, 0.0, 1.0], [l, 1.0, 0.0])
            assert np.all(np.isfinite(out)), f"{mode}(i={i}, l={l}) -> {out.tolist()}"


class TestBoundaryRules:
    """The rules the epsilon guards encode, written out independently of the fixture."""

    def test_color_dodge_full_layer_is_white_not_black(self):
        # Guarding the denominator returns 0 here, which reads as "the dodge
        # layer turned the image black" - the exact inversion CodeRabbit flagged.
        assert _blend("color-dodge", [0.5] * 3, [1.0] * 3) == pytest.approx([1.0] * 3)

    def test_color_dodge_black_backdrop_stays_black(self):
        assert _blend("color-dodge", [0.0] * 3, [1.0] * 3) == pytest.approx([0.0] * 3)

    def test_color_dodge_is_clamped(self):
        assert _blend("color-dodge", [0.6] * 3, [0.9] * 3) == pytest.approx([1.0] * 3)

    def test_color_burn_empty_layer_is_black_not_white(self):
        assert _blend("color-burn", [0.5] * 3, [0.0] * 3) == pytest.approx([0.0] * 3)

    def test_color_burn_white_backdrop_stays_white(self):
        assert _blend("color-burn", [1.0] * 3, [0.0] * 3) == pytest.approx([1.0] * 3)

    def test_vivid_light_boundaries(self):
        assert _blend("vivid-light", [0.5] * 3, [0.0] * 3) == pytest.approx([0.0] * 3)
        assert _blend("vivid-light", [0.5] * 3, [1.0] * 3) == pytest.approx([1.0] * 3)
        assert _blend("vivid-light", [1.0] * 3, [0.0] * 3) == pytest.approx([1.0] * 3)
        assert _blend("vivid-light", [0.0] * 3, [1.0] * 3) == pytest.approx([0.0] * 3)

    def test_divide_by_zero_is_clamped_to_one(self):
        assert _blend("divide", [0.5] * 3, [0.0] * 3) == pytest.approx([1.0] * 3)

    def test_luminosity_over_black_takes_the_layer_luminance(self):
        # A luminosity layer over a black backdrop must not vanish. There is no
        # hue or saturation in the backdrop to preserve, so the result is a
        # neutral grey at the layer's luminance.
        assert _blend("luminosity", [0.0] * 3, [1.0] * 3) == pytest.approx([1.0] * 3)
        assert _blend("luminosity", [0.0] * 3, [0.5] * 3) == pytest.approx([0.5] * 3)

    def test_luminosity_is_continuous_approaching_black(self):
        near = _blend("luminosity", [1e-7] * 3, [1.0] * 3)
        at = _blend("luminosity", [0.0] * 3, [1.0] * 3)
        assert near == pytest.approx(at, abs=TOLERANCE)

    def test_luminosity_preserves_backdrop_chroma(self):
        out = _blend("luminosity", [0.4, 0.2, 0.1], [0.5] * 3)
        assert out[0] > out[1] > out[2]


class TestCompositeAndModeTable:
    def test_unknown_blend_mode_falls_back_to_normal(self):
        unknown = resolve_mode("not-a-mode")
        assert (unknown.blend_space, unknown.composite) == (
            resolve_mode("normal").blend_space,
            resolve_mode("normal").composite,
        )
        assert _blend("not-a-mode", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]) == pytest.approx(
            _blend("normal", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        )

    def test_every_blend_mode_has_a_composite_entry(self):
        for mode in set(CHANNEL_BLEND) | set(HSL_BLEND):
            resolved = resolve_mode(mode)
            assert isinstance(resolved, EffectiveMode)
            assert resolved.blend == mode
            assert resolved.blend_space in ("linear", "perceptual")
            assert resolved.composite in (
                "union",
                "clip-to-backdrop",
                "clip-to-layer",
                "intersection",
            )

    def test_normal_over_transparent_backdrop_keeps_the_layer(self):
        backdrop = np.zeros((1, 1, 4), dtype=np.float32)
        layer = np.float32([[[0.25, 0.5, 0.75, 1.0]]])
        out = blend_composite(resolve_mode("normal"), backdrop, layer, 1.0)
        assert out[0, 0].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_zero_opacity_is_a_no_op(self):
        backdrop = np.float32([[[0.1, 0.2, 0.3, 1.0]]])
        layer = np.float32([[[1.0, 1.0, 1.0, 1.0]]])
        out = blend_composite(resolve_mode("multiply"), backdrop, layer, 0.0)
        assert out[0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3, 1.0])
