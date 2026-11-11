"""Regression tests for ImageCompositor's handling of untrusted layer state.

The compositor's `compositor` widget value is persisted into the saved workflow
and is accepted verbatim on `POST /prompt`, so every field in it is untrusted
input, not an internal invariant.
"""

import numpy as np
import pytest
import torch

from comfy_extras.nodes_compositor import (
    ImageCompositor,
    _layer_params,
    composite_from_state,
    expand_item_frames,
    state_from_items,
)


def _solid(color, w=4, h=4) -> torch.Tensor:
    frame = np.zeros((h, w, len(color)), dtype=np.float32)
    frame[:] = color
    return torch.from_numpy(frame).unsqueeze(0)


class TestLayerOpacity:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(-0.5, 0.0), (0.0, 0.0), (0.25, 0.25), (1.0, 1.0), (3.0, 1.0)],
    )
    def test_opacity_is_clamped(self, raw, expected):
        assert _layer_params({"opacity": raw}, 4, 4)["opacity"] == expected

    def test_opacity_defaults_to_opaque(self):
        assert _layer_params({}, 4, 4)["opacity"] == 1.0

    def test_out_of_range_opacity_does_not_leak_into_the_next_layer(self):
        # The canvas is only clamped once, after every layer has been composited,
        # so an out-of-range coverage multiplier on one layer changes the *blend*
        # of the layer above it. White at opacity 3.0 over black leaves the canvas
        # at 3.0; the multiply above it then reads 3.0 as its backdrop and the
        # result is visibly lighter than the same stack at opacity 1.0.
        def run(opacity):
            state = {
                "canvas": (2, 2),
                "layers": [{"opacity": opacity}, {"opacity": 1.0, "blend": "multiply"}],
                "inputs": None,
                "background": {"color": "#000000", "opacity": 1.0, "visible": True},
                "order": None,
            }
            tensors = [_solid([1.0, 1.0, 1.0], 2, 2), _solid([0.5, 0.5, 0.5], 2, 2)]
            return composite_from_state(tensors, state, [None, None])[0, 0, 0, :3]

        assert run(3.0).tolist() == pytest.approx(run(1.0).tolist(), abs=1e-6)


class TestGraphOnlyBackground:
    def test_default_layout_background_is_hidden(self):
        # A visible white background here would make every graph-only run emit a
        # white matte instead of transparency.
        frames = expand_item_frames([{"image": _solid([1.0, 0.0, 0.0])}])
        state = state_from_items(frames, (4, 4))
        assert state["background"]["visible"] is False

    def test_uncovered_canvas_stays_transparent(self):
        tensors = [_solid([1.0, 0.0, 0.0], w=2, h=2)]
        frames = expand_item_frames([{"image": tensors[0]}])
        state = state_from_items(frames, (4, 4))
        out = composite_from_state(tensors, state, [None])[0]
        assert out.shape[-1] == 4
        assert float(out[0, 0, 3]) == pytest.approx(1.0)
        assert float(out[3, 3, 3]) == pytest.approx(0.0)


class TestCanvasEmission:
    """execute must report the document canvas so the editor sizes itself to it
    rather than to the max natural size of cropped/placed layers."""

    def test_explicit_document_canvas_is_emitted(self):
        # A small layer placed on a large explicit canvas: the editor must learn
        # the 1280x1280 canvas, not the 200x150 layer size.
        doc = {
            "version": 1,
            "canvas": (1280, 1280),
            "layers": [
                {"image": _solid([1.0, 0.0, 0.0, 1.0], w=200, h=150),
                 "type": "raster", "x": 400, "y": 300, "z_index": 0}
            ],
        }
        ui = ImageCompositor.execute(layers=doc).ui
        assert ui["compositor_canvas"] == [{"w": 1280, "h": 1280}]

    def test_replay_emits_saved_canvas(self):
        tensor = _solid([0.0, 1.0, 0.0, 1.0], w=4, h=4)
        doc = {"version": 1, "layers": [{"image": tensor, "type": "raster"}]}
        fp = ImageCompositor.execute(layers=doc).ui["compositor_inputs"]
        saved = {
            "version": 1,
            "canvas": {"w": 640, "h": 480},
            "inputs": fp,
            "layers": [{
                "name": "a", "visible": True, "opacity": 1.0, "blend": "normal",
                "flipH": False, "flipV": False,
                "transform": {"x": 0, "y": 0, "w": 4, "h": 4, "rotation": 0.0},
            }],
        }
        ui = ImageCompositor.execute(layers=doc, compositor=saved).ui
        assert ui["compositor_canvas"] == [{"w": 640, "h": 480}]

    def test_no_layers_emits_no_canvas(self):
        ui = ImageCompositor.execute(layers={"version": 1, "layers": []}).ui
        assert "compositor_canvas" not in ui
