import math
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_base  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.sampler_helpers  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.supported_models  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402


def _model(monkeypatch, latent_shapes):
    monkeypatch.setattr(comfy.model_base.BaseModel, "__init__", lambda self, *args, **kwargs: None)
    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    model.__init__(None)
    model.memory_usage_factor = comfy.supported_models.MiniMaxH3.memory_usage_factor
    model.memory_usage_shape_process = {}
    model.latent_shapes = latent_shapes
    model.model_sampling = SimpleNamespace(audio_scale=1.0)
    return model


def _packed_shape(latent_shapes, batch=1):
    elements = sum(math.prod(shape[1:]) for shape in latent_shapes)
    return [batch, 1, elements]


def test_minimax_h3_without_latent_shapes_uses_generic_memory_fallback(monkeypatch):
    model = _model(monkeypatch, None)
    monkeypatch.setattr(comfy.model_management, "xformers_enabled", lambda: False)
    monkeypatch.setattr(comfy.model_management, "pytorch_attention_flash_attention", lambda: False)
    input_shape = (1, 24, 3, 4, 6)
    area = input_shape[0] * math.prod(input_shape[2:])

    assert model.memory_usage_factor == 0.24
    assert model.memory_required(input_shape) == area * 0.15 * 0.24 * 1024 * 1024


def test_sampler_exposes_nested_shapes_before_prepare_sampling(monkeypatch):
    video = torch.empty(1, 24, 3, 4, 6)
    audio = torch.empty(1, 32, 2, 5)
    model = SimpleNamespace(latent_shapes=None)
    guider = comfy.samplers.CFGGuider(SimpleNamespace(model=model, model_options={}))

    class ShapesSeen(Exception):
        pass

    def check_shapes(*args, **kwargs):
        assert model.latent_shapes == [video.shape, audio.shape]
        raise ShapesSeen

    monkeypatch.setattr(comfy.samplers, "detail", check_shapes)
    with pytest.raises(ShapesSeen):
        guider.sample(
            comfy.nested_tensor.NestedTensor([video, audio]),
            comfy.nested_tensor.NestedTensor([video, audio]),
            sampler=None,
            sigmas=torch.ones(1),
        )


def test_minimax_h3_estimates_real_packed_target_and_all_condition_rows(monkeypatch):
    latent_shapes = [(1, 24, 3, 4, 6), (1, 32, 2, 5)]
    model = _model(monkeypatch, latent_shapes)
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

    payload_shape = model.extra_conds_shapes(
        cross_attn=cross_attn,
        minimax_keyframes=keyframes,
        minimax_refs=refs,
    )["minimax_payload"]
    layout = PackedLayout(29, 3, 4, 6, 5, keyframes=keyframes, refs=refs)
    target_rows = 3 * 2 * 3 + 5 * 2

    assert payload_shape == [1, 1, layout.seq_len - target_rows]
    estimate = model.memory_required(
        _packed_shape(latent_shapes),
        cond_shapes={"minimax_payload": [payload_shape]},
    )
    assert estimate == layout.seq_len * model.MEMORY_BYTES_PER_PACKED_ROW

    monkeypatch.setattr(comfy.model_base.BaseModel, "extra_conds", lambda self, **kwargs: {})
    payload = model.extra_conds(minimax_keyframes=keyframes, minimax_refs=refs)["minimax_payload"]
    expected_shape = model.extra_conds_shapes(minimax_keyframes=keyframes, minimax_refs=refs)["minimax_payload"]
    assert payload.size() == expected_shape
    assert payload.process_cond(batch_size=1).size() == expected_shape


def test_minimax_h3_packed_scalar_count_does_not_become_row_count(monkeypatch):
    latent_shapes = [(1, 24, 37, 48, 84), (1, 32, 2, 207)]
    model = _model(monkeypatch, latent_shapes)
    packed_shape = _packed_shape(latent_shapes)
    cross_attn = torch.empty(1, 29, 8)

    assert packed_shape == [1, 1, 3593664]
    target_rows = 37 * 24 * 42 + 207 * 2
    full, minimum = comfy.sampler_helpers.estimate_memory(
        SimpleNamespace(model=model),
        packed_shape,
        {"positive": [{"cross_attn": cross_attn}]},
    )
    bytes_per_row = model.MEMORY_BYTES_PER_PACKED_ROW
    assert minimum == (target_rows + 29) * bytes_per_row
    assert full == (target_rows * 2 + 29) * bytes_per_row
    assert minimum < 6 * 1024 ** 3
    assert full < 12 * 1024 ** 3


def test_minimax_h3_measured_process_memory_row_delta_envelope(monkeypatch):
    # Windows process-dedicated allocation peaks from vanilla H3 runs stopped after
    # the first transformer block. Peak differences between sequence sizes cancel
    # fixed model/process allocation and isolate the row-dependent working-set slope.
    cases = {
        "t17": ((1, 24, 17, 30, 52), (1, 32, 2, 93), 1858949120),
        "t37": ((1, 24, 37, 30, 52), (1, 32, 2, 207), 3066916864),
        "t72": ((1, 24, 72, 30, 52), (1, 32, 2, 405), 5080199168),
    }
    estimates = {}
    for name, (video_shape, audio_shape, _) in cases.items():
        model = _model(monkeypatch, [video_shape, audio_shape])
        payload_shape = model.extra_conds_shapes(cross_attn=torch.empty(1, 29, 8))["minimax_payload"]
        estimates[name] = model.memory_required(
            _packed_shape([video_shape, audio_shape]),
            cond_shapes={"minimax_payload": [payload_shape]},
        )

    measured_deltas = (
        ("t17", "t37", cases["t37"][2] - cases["t17"][2]),
        ("t37", "t72", cases["t72"][2] - cases["t37"][2]),
    )
    for smaller, larger, measured in measured_deltas:
        estimated = estimates[larger] - estimates[smaller]
        assert measured <= estimated <= measured * 1.06

    model = _model(monkeypatch, [cases["t37"][0], cases["t37"][1]])
    ref = {"kind": "video", "latent_t": 37, "latent_h": 30, "latent_w": 52, "ref_audio_t": 0}
    target_payload = model.extra_conds_shapes(cross_attn=torch.empty(1, 29, 8))["minimax_payload"]
    ref_payload = model.extra_conds_shapes(cross_attn=torch.empty(1, 29, 8), minimax_refs=[ref])["minimax_payload"]
    estimated_ref_delta = model.memory_required(
        _packed_shape(model.latent_shapes), {"minimax_payload": [ref_payload]}
    ) - model.memory_required(
        _packed_shape(model.latent_shapes), {"minimax_payload": [target_payload]}
    )
    # The warm target/reference delta similarly isolates the reference-row working set.
    measured_ref_delta = 5247971328 - 3100471296
    assert measured_ref_delta <= estimated_ref_delta <= measured_ref_delta * 1.06
