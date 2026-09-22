import math
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_base  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.sampler_helpers  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402


def _model(latent_shapes):
    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    model.latent_shapes = latent_shapes
    return model


def _packed_scalar_shape(latent_shapes):
    return [1, 1, sum(math.prod(shape[1:]) for shape in latent_shapes)]


def test_minimax_h3_outer_sample_publishes_geometry_before_memory_admission(monkeypatch):
    shapes = [(1, 24, 57, 56, 74), (1, 32, 2, 320)]
    model = SimpleNamespace(latent_shapes=None)
    guider = comfy.samplers.CFGGuider.__new__(comfy.samplers.CFGGuider)
    guider.model_patcher = SimpleNamespace(model=model)
    guider.conds = {}
    guider.model_options = {}

    class ShapesSeen(Exception):
        pass

    def check_prepare_sampling(*args, **kwargs):
        assert model.latent_shapes == shapes
        raise ShapesSeen

    monkeypatch.setattr(comfy.sampler_helpers, "prepare_sampling", check_prepare_sampling)
    with pytest.raises(ShapesSeen):
        guider.outer_sample(
            noise=torch.zeros(1, 1, 1),
            latent_image=torch.zeros(1, 1, 1),
            sampler=None,
            sigmas=torch.ones(1),
            latent_shapes=shapes,
        )


def test_minimax_h3_inner_sample_rebinds_wrapper_changed_geometry():
    target_shapes = [(1, 24, 57, 56, 74), (1, 32, 2, 320)]
    source_shapes = [(1, 24, 57, 40, 52), (1, 32, 2, 320)]

    class ShapesSeen(Exception):
        pass

    class InnerModel:
        latent_shapes = target_shapes

        def process_latent_in(self, latent_image):
            assert self.latent_shapes == source_shapes
            raise ShapesSeen

    guider = comfy.samplers.CFGGuider.__new__(comfy.samplers.CFGGuider)
    guider.inner_model = InnerModel()
    packed_elements = sum(math.prod(shape[1:]) for shape in source_shapes)
    latent_image = torch.ones(1, 1, packed_elements)

    with pytest.raises(ShapesSeen):
        guider.inner_sample(
            noise=torch.zeros_like(latent_image),
            latent_image=latent_image,
            device=torch.device("cpu"),
            sampler=None,
            sigmas=torch.ones(1),
            denoise_mask=None,
            callback=None,
            disable_pbar=True,
            seed=0,
            latent_shapes=source_shapes,
        )


def test_minimax_h3_condition_row_count_matches_packed_layout():
    latent_shapes = [(1, 24, 3, 4, 6), (1, 32, 2, 5)]
    model = _model(latent_shapes)
    keyframes = [
        {"resolved_frame_index": 0, "latent": torch.empty(1, 24, 2, 4, 6)},
        {"resolved_frame_index": 1, "audio_latent": torch.empty(1, 32, 2, 4)},
    ]
    refs = [
        {"kind": "image", "latent_h": 4, "latent_w": 6},
        {"kind": "audio", "ref_audio_t": 3},
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6, "ref_audio_t": 0},
        {"kind": "video_audio", "latent_t": 2, "latent_h": 4, "latent_w": 6, "ref_audio_t": 3},
    ]
    cross_attn = torch.empty(1, 29, 8)

    cond_shape = model.extra_conds_shapes(
        cross_attn=cross_attn,
        minimax_keyframes=keyframes,
        minimax_refs=refs,
    )[model.PACKED_COND_ROWS_KEY]
    layout = PackedLayout(29, 3, 4, 6, 5, keyframes=keyframes, refs=refs)

    assert cond_shape == [1, 1, layout.seq_len - model._target_packed_rows()]


def test_minimax_h3_memory_estimate_scales_with_refs_and_cfg_batch():
    latent_shapes = [(1, 24, 37, 48, 84), (1, 32, 2, 207)]
    model = _model(latent_shapes)
    refs = [
        {"kind": "image", "latent_h": 56, "latent_w": 74},
        {"kind": "image", "latent_h": 64, "latent_w": 48},
    ]
    positive = {"cross_attn": torch.empty(1, 5120, 8), "minimax_refs": refs}
    negative = {"cross_attn": torch.empty(1, 1024, 8)}

    full, minimum = comfy.sampler_helpers.estimate_memory(
        SimpleNamespace(model=model),
        _packed_scalar_shape(latent_shapes),
        {"positive": [positive], "negative": [negative]},
    )

    target_rows = model._target_packed_rows()
    positive_rows = model._conditioning_packed_rows(**positive)
    negative_rows = model._conditioning_packed_rows(**negative)
    bytes_per_row = model.PACKED_ROW_MEMORY_BYTES

    assert full == (2 * target_rows + positive_rows + negative_rows) * bytes_per_row
    assert minimum == (target_rows + max(positive_rows, negative_rows)) * bytes_per_row
    assert model._conditioning_packed_rows(**{**positive, "minimax_refs": refs + [refs[0]]}) > positive_rows
