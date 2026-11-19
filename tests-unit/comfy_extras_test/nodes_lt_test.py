"""Unit tests for LTXVAddLatentGuide and the guide-attachment path it shares with LTXVAddGuide.

The RoPE arithmetic runs for real here: only ``nodes`` and ``server`` are stubbed, so
``append_keyframe``, ``dilate_latent`` and ``_append_guide_attention_entry`` are the
real implementations and the assertions are on actual keyframe coordinates.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Stub nodes/server for the import only, then restore exactly those keys. patch.dict is
# not used because it restores the whole of sys.modules on exit, which evicts everything
# imported inside the block and forces a re-import that trips duplicate TORCH_LIBRARY
# registration. Leaving the stubs installed is equally wrong: pytest imports every test
# module at collection time, so a lingering MagicMock "nodes" breaks later modules that
# use the real one. Same shape as tests-unit/comfy_extras_test/image_stitch_test.py.
_stubs = {"nodes": MagicMock(MAX_RESOLUTION=16384), "server": MagicMock()}
_saved = {name: sys.modules.get(name) for name in _stubs}
sys.modules.update(_stubs)
try:
    import comfy_extras.nodes_lt as nodes_lt
finally:
    for _name, _original in _saved.items():
        if _original is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _original

LATENT_CHANNELS = 128
SCALE_FACTORS = (8, 32, 32)
TIME, HEIGHT, WIDTH = 0, 1, 2
START, END = 0, 1


def _vae():
    return SimpleNamespace(downscale_index_formula=SCALE_FACTORS)


def _latent(frames, height, width):
    return {"samples": torch.zeros((1, LATENT_CHANNELS, frames, height, width))}


def _cond():
    return [({}, {})]


def _add_latent_guide(guide_hw, latent_hw=(4, 4), guide_frames=1, latent_frames=3, latent_idx=0):
    positive, negative, latent = nodes_lt.LTXVAddLatentGuide.execute(
        _cond(),
        _cond(),
        _vae(),
        _latent(latent_frames, *latent_hw),
        _latent(guide_frames, *guide_hw),
        latent_idx,
        1.0,
    )
    metadata = positive[0][1]
    return metadata["keyframe_idxs"], metadata["guide_attention_entries"], latent


def _axis(keyframe_idxs, axis, bound):
    return keyframe_idxs[0, axis, :, bound].tolist()


def test_same_size_guide_spans_one_patch_per_token():
    """A 1:1 guide gets no offset: each token's end is one scale factor past its start.

    The inverse of the case below, so a factor derived wrongly at 1:1 is caught too.
    """
    keyframe_idxs, entries, _ = _add_latent_guide(guide_hw=(4, 4))

    for axis in (HEIGHT, WIDTH):
        starts = _axis(keyframe_idxs, axis, START)
        assert _axis(keyframe_idxs, axis, END) == [s + SCALE_FACTORS[axis] for s in starts]

    assert entries[0]["latent_shape"] == [1, 4, 4]


def test_half_size_guide_expands_only_the_end_positions():
    """An x2 guide keeps its start positions and pushes each end out by one scale factor.

    That is what makes the dilated reference cover the whole canvas. Leaving the factor
    at 1 keeps same-size coordinates while each token encodes a larger patch, so the
    reference addresses only the top-left corner of the target.
    """
    same_size, _, _ = _add_latent_guide(guide_hw=(4, 4))
    downscaled, entries, _ = _add_latent_guide(guide_hw=(2, 2))

    # Dilation puts the small guide on the same sparse grid, so token count is unchanged.
    assert downscaled.shape == same_size.shape

    for axis in (HEIGHT, WIDTH):
        assert _axis(downscaled, axis, START) == _axis(same_size, axis, START)
        expected = [e + SCALE_FACTORS[axis] for e in _axis(same_size, axis, END)]
        assert _axis(downscaled, axis, END) == expected

    # Time is never touched by the spatial offset.
    assert _axis(downscaled, TIME, START) == _axis(same_size, TIME, START)
    assert _axis(downscaled, TIME, END) == _axis(same_size, TIME, END)

    assert entries[0]["latent_shape"] == [1, 2, 2]


def test_attention_entry_lets_context_windows_rederive_the_factor():
    """The entry keeps the pre-dilation shape while the token count is post-dilation.

    ``context_windows`` divides the post-dilation guide height by the entry's
    ``latent_shape`` height to recover the downscale factor, so these two must not drift
    apart or windowed and non-windowed sampling disagree on the guide's RoPE.
    """
    _, entries, latent = _add_latent_guide(guide_hw=(2, 2))

    entry = entries[0]
    assert entry["latent_shape"] == [1, 2, 2]  # pre-dilation
    assert entry["pre_filter_count"] == 1 * 4 * 4  # post-dilation
    assert latent["samples"].shape[3] // entry["latent_shape"][1] == 2


@pytest.mark.parametrize(
    "latent_idx, expected_start", [(-1, -8), (0, 0), (1, 1), (2, 9)]
)
def test_latent_idx_maps_onto_pixel_frames(latent_idx, expected_start):
    """latent_idx is in latent frames, and negatives sit before the start of the latent.

    The first latent frame covers a single pixel frame, so the mapping is 0 -> 0, 1 -> 1,
    then 8 apart. Negative values are not counted back from the end.
    """
    keyframe_idxs, _, _ = _add_latent_guide(
        guide_hw=(4, 4), latent_frames=8, latent_idx=latent_idx
    )

    assert set(_axis(keyframe_idxs, TIME, START)) == {expected_start}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(guide_hw=(2, 4)), "square"),
        (dict(guide_hw=(3, 3)), "whole number"),
        (dict(guide_hw=(4, 4), latent_idx=99), "runs past the end"),
        (dict(guide_hw=(4, 4), guide_frames=5), "runs past the end"),
    ],
)
def test_unusable_guides_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _add_latent_guide(**kwargs)


def test_non_5d_guiding_latent_is_rejected():
    """An image-model latent would otherwise fail with a bare IndexError on shape[4]."""
    with pytest.raises(ValueError, match="5D video latent"):
        nodes_lt.LTXVAddLatentGuide.execute(
            _cond(),
            _cond(),
            _vae(),
            _latent(3, 4, 4),
            {"samples": torch.zeros((1, LATENT_CHANNELS, 4, 4))},
            0,
            1.0,
        )


def test_attention_mask_reaches_the_guide_entry():
    """Only coverage that the optional input is forwarded to the guide entry."""
    mask = torch.full((1, 128, 128), 0.5)
    positive, negative, _ = nodes_lt.LTXVAddLatentGuide.execute(
        _cond(), _cond(), _vae(), _latent(3, 4, 4), _latent(1, 2, 2), 0, 1.0, attention_mask=mask
    )

    # Stored as (1, 1, F, H, W) for downstream self-attention masking.
    assert positive[0][1]["guide_attention_entries"][0]["pixel_mask"].shape == (1, 1, 1, 128, 128)
    # Positive and negative each get their own entry, so neither leaks into the other.
    assert len(negative[0][1]["guide_attention_entries"]) == 1


def test_node_is_registered_with_a_loadable_schema():
    """Both failure modes here are invisible to every other test in this file.

    A bad io.Schema keyword only surfaces when the node is registered, and a node left
    out of the extension list simply does not exist in ComfyUI. The strength cap is
    asserted here because raising it re-exposes the missing clamp in append_keyframe's
    guide_mask branch, where a dilated guide is dropped above 1.0.
    """
    schema = nodes_lt.LTXVAddLatentGuide.define_schema()
    inputs = {inp.id: inp for inp in schema.inputs}

    assert schema.node_id == "LTXVAddLatentGuide"
    assert inputs["strength"].max == 1.0
    assert inputs["attention_mask"].optional is True

    node_list = asyncio.run(nodes_lt.LtxvExtension().get_node_list())
    assert nodes_lt.LTXVAddLatentGuide in node_list


@pytest.mark.parametrize(
    "iclora_parameters, expected_shape, expected_extra_end",
    [
        (None, [1, 4, 4], 0),
        ({"reference_downscale_factor": 2}, [1, 2, 2], SCALE_FACTORS[HEIGHT]),
    ],
)
def test_add_guide_image_path_still_routes_through_the_shared_helper(
    iclora_parameters, expected_shape, expected_extra_end
):
    """LTXVAddGuide must be unchanged by sharing attach_guide_latent with the latent node."""

    class _Vae:
        downscale_index_formula = SCALE_FACTORS

        def encode(self, pixels):
            frames, height, width, _ = pixels.shape
            return torch.zeros(
                (1, LATENT_CHANNELS, (frames - 1) // SCALE_FACTORS[TIME] + 1, height // 32, width // 32)
            )

    positive, _, _ = nodes_lt.LTXVAddGuide.execute(
        _cond(),
        _cond(),
        _Vae(),
        _latent(3, 4, 4),
        torch.zeros((1, 4 * 32, 4 * 32, 3)),
        0,
        1.0,
        iclora_parameters=iclora_parameters,
    )

    metadata = positive[0][1]
    entry = metadata["guide_attention_entries"][0]
    assert entry["latent_shape"] == expected_shape
    assert entry["pre_filter_count"] == 1 * 4 * 4

    keyframe_idxs = metadata["keyframe_idxs"]
    starts = _axis(keyframe_idxs, HEIGHT, START)
    ends = _axis(keyframe_idxs, HEIGHT, END)
    assert ends == [s + SCALE_FACTORS[HEIGHT] + expected_extra_end for s in starts]
