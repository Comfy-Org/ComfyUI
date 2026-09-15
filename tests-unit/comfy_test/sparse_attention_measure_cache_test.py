from types import SimpleNamespace

import pytest
import torch

from comfy import attention_measure as m
from comfy_extras import sparse_attention_measure as sparse_measure


def _request():
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": 24,
        "kv_rows": 24,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": [2, 3],
        "prefix_grid": [3, 4],
        "segments": [
            {"start": 0, "stop": 6, "mass_num": 1, "mass_den": 1},
            {"start": 6, "stop": 18, "mass_num": 1, "mass_den": 2},
            {"start": 18, "stop": 24, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }


def _layout(video_start=6):
    return SimpleNamespace(
        seq_len=24,
        segments=((0, video_start, "text"), (video_start, 24, "video")),
    )


def _external(*, prefix_t=1):
    return {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 18,
        "sequence_rows": 24,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": prefix_t,
        "source_rows_per_frame": 6,
        "prefix_rows_per_frame": 12,
    }


def _provider(*args, key_bias=None, **kwargs):
    raise AssertionError("provider must not execute while binding a measure plan")


def _prepare(patch, options):
    return sparse_measure.prepare(
        patch,
        options,
        block_index=0,
        q_rows=24,
        kv_rows=24,
        device="cpu",
        dtype=torch.bfloat16,
        head_dim=128,
        mask_class="none",
        existing_sink=(0, 1),
        provider=_provider,
        profile="weighted_exact_blocks_v1",
        numerical_route="core_bsa_h3_chunked",
        preprocess_digest="pre",
    )


def test_cached_measure_plan_revalidates_live_layout_external_and_owner():
    patch = SimpleNamespace(
        vsa=False,
        measure_owner_generation="owner",
        measure_capability=None,
        measure_plans={},
    )
    options = {
        m.ATTENTION_MEASURE_KEY: _request(),
        "minimax_h3_layout": _layout(),
    }
    sparse_measure.register(patch, options)

    first = _prepare(patch, options)
    assert first is _prepare(patch, options)
    assert len(patch.measure_plans) == 1

    options["minimax_h3_layout"] = _layout(video_start=5)
    with pytest.raises(ValueError, match="video interval"):
        _prepare(patch, options)

    options["minimax_h3_layout"] = _layout()
    options[sparse_measure.VDN_EXTERNAL_SEQUENCE_KEY] = _external(prefix_t=2)
    with pytest.raises(ValueError, match="VDN external-sequence"):
        _prepare(patch, options)

    options.pop(sparse_measure.VDN_EXTERNAL_SEQUENCE_KEY)
    options[m.ATTENTION_MEASURE_CAPABILITIES_KEY] = {
        sparse_measure.PROVIDER_IDENTITY: object()
    }
    with pytest.raises(RuntimeError, match="capability owner changed"):
        _prepare(patch, options)
