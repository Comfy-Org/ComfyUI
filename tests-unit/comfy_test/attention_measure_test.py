from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from comfy import attention_measure as m
from comfy_extras import sparse_attention_measure as sparse_measure


def request(*, equal=False):
    source = [2, 3]
    prefix = source if equal else [3, 4]
    source_rows = 6
    prefix_rows = 6 if equal else 12
    rows = 6 + prefix_rows + source_rows
    ratio = (1, 1) if equal else (1, 2)
    return {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": rows,
        "kv_rows": rows,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": source,
        "prefix_grid": prefix,
        "segments": [
            {"start": 0, "stop": 6, "mass_num": 1, "mass_den": 1},
            {"start": 6, "stop": 6 + prefix_rows, "mass_num": ratio[0], "mass_den": ratio[1]},
            {"start": 6 + prefix_rows, "stop": rows, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }


def layout(rows):
    return SimpleNamespace(seq_len=rows, segments=((0, 6, "text"), (6, rows, "video")))


def context(r):
    return m.MeasureExecutionContext(
        provider_identity="test.provider",
        block_index=3,
        owner=object(),
        owner_generation="gen-1",
        layout=layout(r["q_rows"]),
        q_rows=r["q_rows"],
        kv_rows=r["kv_rows"],
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        mask_class="none",
        preprocess_digest="pre-1",
        numerical_route="test",
    )


def test_normalize_is_semantic_not_segment_spelling():
    r = request()
    canonical = m.normalize(r)
    split = {**r, "segments": [
        {"start": 0, "stop": 2, "mass_num": 4, "mass_den": 4},
        {"start": 2, "stop": 6, "mass_num": 1, "mass_den": 1},
        {"start": 6, "stop": 12, "mass_num": 2, "mass_den": 4},
        {"start": 12, "stop": 18, "mass_num": 1, "mass_den": 2},
        {"start": 18, "stop": 24, "mass_num": 3, "mass_den": 3},
    ]}
    assert m.normalize(split) == canonical
    assert m.semantic_digest(split) == m.semantic_digest(r)


def test_equal_grid_is_unit_measure_and_preserves_existing_sink():
    r = request(equal=True)
    normalized = m.normalize(r)
    assert normalized["segments"] == [{"start": 0, "stop": 18, "mass_num": 1, "mass_den": 1}]
    assert m.is_unit_measure(r)
    assert m.merge_exact_k_blocks(r, 64, existing=(0, 1)) == (0, 1)
    assert torch.equal(m.materialize(r, device="cpu"), torch.zeros(18))


def test_ragged_weighted_interval_expands_to_physical_k_blocks():
    r = {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "q_rows": 303,
        "kv_rows": 303,
        "video_start": 63,
        "temporal": 3,
        "prefix_t": 1,
        "source_grid": [8, 8],
        "prefix_grid": [8, 14],
        "segments": [
            {"start": 0, "stop": 63, "mass_num": 1, "mass_den": 1},
            {"start": 63, "stop": 175, "mass_num": 4, "mass_den": 7},
            {"start": 175, "stop": 303, "mass_num": 1, "mass_den": 1},
        ],
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
    }
    assert m.merge_exact_k_blocks(r, 64) == (0, 3)
    assert m.merge_exact_k_blocks(r, 64, existing=(0, 1)) == (0, 3)


def test_h3_validation_cross_checks_actual_layout_and_vdn_when_present():
    r = request()
    rows = r["q_rows"]
    external = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 18,
        "sequence_rows": rows,
        "video_start": 6,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 6,
        "prefix_rows_per_frame": 12,
    }
    assert m.validate_h3(r, layout=layout(rows), q_rows=rows, kv_rows=rows, external_sequence=external) == m.normalize(r)
    with pytest.raises(ValueError):
        m.validate_h3(r, layout=SimpleNamespace(seq_len=rows, segments=((0, rows, "video"),)), q_rows=rows, kv_rows=rows)
    with pytest.raises(ValueError):
        m.validate_h3(r, layout=layout(rows), q_rows=rows, kv_rows=rows, external_sequence={**external, "prefix_t": 2})


def test_bind_and_capability_validate_concrete_owner_identity():
    r = request()
    ctx = context(r)

    class Capability:
        def prepare(self, request, execution_context):
            return m.bind(
                request,
                context=execution_context,
                block_size=64,
                implementation_profile="weighted_exact_blocks_v1",
            )

    options = {}
    cap = Capability()
    m.register_capability(options, "test.provider", cap)
    plan = m.prepare_capability(options, r, ctx)
    assert plan.semantic_digest == m.semantic_digest(r)
    assert plan.exact_k_block_range == (0, 1)
    assert plan.provider_identity == "test.provider"
    with pytest.raises(RuntimeError):
        m.prepare_capability(options, r, replace(ctx, provider_identity="other"))


def test_bsa_pool_key_preserves_existing_unweighted_history_domain():
    assert sparse_measure.pool_key(7, 56029, ("cond", 2), None) == (
        7,
        56029,
        ("cond", 2),
    )

    plan = m.BoundMeasurePlan(
        semantic_digest="measure-a",
        implementation_profile="weighted_exact_blocks_v1",
        provider_identity="comfy.core.block_sparse_attention",
        owner_generation="owner-a",
        numerical_route="core_bsa_h3_chunked",
        preprocess_digest="pre-a",
        q_rows=56029,
        kv_rows=56029,
        exact_k_block_range=(0, 387),
        exact_range_digest="range-a",
        key_log_measure=torch.empty(0),
    )
    weighted = sparse_measure.pool_key(7, 56029, ("cond", 2), plan)
    assert weighted[:3] == (7, 56029, ("cond", 2))
    assert weighted[3] == (
        "attention_measure_v1",
        "measure-a",
        "comfy.core.block_sparse_attention",
        "owner-a",
        "core_bsa_h3_chunked",
        "pre-a",
        "weighted_exact_blocks_v1",
    )
    assert weighted != sparse_measure.pool_key(7, 56029, ("cond", 2), None)


def test_sparse_measure_rejects_provider_without_key_bias_before_binding():
    class Patch:
        vsa = False
        measure_owner_generation = "owner"
        measure_plans = {}

    def provider_without_bias(q, k, v):
        raise AssertionError("provider must not execute during capability check")

    assert not sparse_measure.supports_key_bias(provider_without_bias)
    with pytest.raises(RuntimeError, match="without key_bias support"):
        sparse_measure.prepare(
            Patch(),
            {m.ATTENTION_MEASURE_KEY: request()},
            block_index=0,
            q_rows=24,
            kv_rows=24,
            device="cpu",
            dtype=torch.bfloat16,
            head_dim=128,
            mask_class="none",
            existing_sink=(0, 1),
            provider=provider_without_bias,
            profile="weighted_exact_blocks_v1",
            numerical_route="core_bsa_h3_chunked",
            preprocess_digest="pre",
        )


def test_vdn_epilogue_resolution_is_optional_but_owner_bound_when_installed():
    class Bound:
        def apply(self, softmax_out, x):
            return softmax_out

        def receipt_fields(self):
            return (("completed", True),)

    class Capability:
        def __init__(self):
            self.calls = []

        def prepare(self, x, rope, options, block_index):
            self.calls.append((x, rope, options, block_index))
            return Bound()

    x = torch.zeros(24, 8)
    rope = torch.zeros(1, 24, 1, 1)
    options = {
        "vdn_h3_external_sequence_v1": {
            "api": 2,
            "mode": "dense_gate_no_linear",
            "topology": "mixed_grid_low_suffix",
        }
    }

    # Flow publishes API-2 mixed geometry independently of whether VDN is
    # installed. A native H3 owner therefore keeps its native projection.
    native = SimpleNamespace(forward=lambda *args, **kwargs: None)
    assert sparse_measure.prepare_vdn_epilogue(native, x, rope, options, 5) is None

    forward = lambda *args, **kwargs: None
    setattr(forward, sparse_measure.VDN_FORWARD_MARKER, True)
    setattr(forward, sparse_measure.VDN_EXTERNAL_SEQUENCE_API_ATTR, 2)
    capability = Capability()
    setattr(forward, sparse_measure.VDN_EPILOGUE_KEY, capability)
    attn = SimpleNamespace(forward=forward)
    bound = sparse_measure.prepare_vdn_epilogue(attn, x, rope, options, 5)
    assert isinstance(bound, Bound)
    assert capability.calls == [(x, rope, options, 5)]

    missing = lambda *args, **kwargs: None
    setattr(missing, sparse_measure.VDN_FORWARD_MARKER, True)
    setattr(missing, sparse_measure.VDN_EXTERNAL_SEQUENCE_API_ATTR, 2)
    with pytest.raises(RuntimeError, match="owner-bound"):
        sparse_measure.prepare_vdn_epilogue(
            SimpleNamespace(forward=missing), x, rope, options, 5
        )

    incompatible = lambda *args, **kwargs: None
    setattr(incompatible, sparse_measure.VDN_FORWARD_MARKER, True)
    setattr(incompatible, sparse_measure.VDN_EXTERNAL_SEQUENCE_API_ATTR, 1)
    with pytest.raises(RuntimeError, match="API 2"):
        sparse_measure.prepare_vdn_epilogue(
            SimpleNamespace(forward=incompatible), x, rope, options, 5
        )

    assert sparse_measure.prepare_vdn_epilogue(
        native,
        x,
        rope,
        {"vdn_h3_external_sequence_v1": {"api": 1}},
        5,
    ) is None


def test_weighted_dense_sdpa_and_streaming_match_and_preserve_masks():
    torch.manual_seed(12)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 24, 8)
    v = torch.randn(1, 2, 24, 8)
    bias = m.materialize(request(), device="cpu")
    mask = torch.ones(1, 1, 5, 24, dtype=torch.bool)
    mask[..., 1, :] = False
    mask[..., 3, 17:] = False
    got = m.weighted_dense(q, k, v, 2, key_bias=bias, mask=mask, skip_reshape=True, skip_output_reshape=True)
    streamed = m.weighted_dense_streaming(
        q, k, v, 2, key_bias=bias, mask=mask, skip_reshape=True, skip_output_reshape=True, key_chunk_size=7
    )
    torch.testing.assert_close(streamed, got, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(got[:, :, 1]) == 0


def test_malformed_values_and_additive_nan_rejected():
    r = request()
    with pytest.raises(TypeError):
        m.normalize({**r, "q_rows": True})
    bad = {**r, "segments": [dict(x) for x in r["segments"]]}
    bad["segments"][1]["mass_den"] = False
    with pytest.raises(TypeError):
        m.normalize(bad)
    q = torch.zeros(1, 1, 1, 8)
    k = torch.zeros(1, 1, 24, 8)
    v = torch.zeros_like(k)
    bias = m.materialize(r, device="cpu")
    mask = torch.zeros(1, 1, 1, 24)
    mask[..., 0] = float("nan")
    with pytest.raises(ValueError):
        m.weighted_dense(q, k, v, 1, key_bias=bias, mask=mask, skip_reshape=True)
