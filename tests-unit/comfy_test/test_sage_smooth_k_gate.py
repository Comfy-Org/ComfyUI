import pytest
import torch

from comfy.ldm.modules.attention import (
    SAGE_SMOOTH_K_MIN_MEAN_RATIO,
    SAGE_SMOOTH_K_MIN_SEQ,
    _sage_should_smooth_k,
)


def _keys(seq_len, layout, shared_component):
    """Random keys with a controllable common component.

    shared_component=0 gives zero-mean keys (mean norm << key norm);
    shared_component=30 makes one direction dominate every key (mean norm ~0.97 x key norm).
    """
    torch.manual_seed(0)
    heads, dim = 4, 64
    shape = (1, heads, seq_len, dim) if layout == "HND" else (1, seq_len, heads, dim)
    k = torch.randn(shape, dtype=torch.float16)
    if shared_component:
        direction = torch.zeros(dim, dtype=torch.float16)
        direction[0] = shared_component
        k = k + direction
    return k


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("seq_len", [SAGE_SMOOTH_K_MIN_SEQ - 1, SAGE_SMOOTH_K_MIN_SEQ, SAGE_SMOOTH_K_MIN_SEQ + 1])
def test_sequence_length_gate(layout, seq_len):
    # Keys with a dominant shared component: smoothing is worthwhile, but only at or above the minimum length.
    k = _keys(seq_len, layout, shared_component=30)
    expected = seq_len >= SAGE_SMOOTH_K_MIN_SEQ
    assert _sage_should_smooth_k(k, layout) is expected


@pytest.mark.parametrize("layout", ["HND", "NHD"])
def test_mean_ratio_gate(layout):
    seq_len = SAGE_SMOOTH_K_MIN_SEQ + 1
    assert _sage_should_smooth_k(_keys(seq_len, layout, shared_component=30), layout) is True
    assert _sage_should_smooth_k(_keys(seq_len, layout, shared_component=0), layout) is False


def test_gate_does_not_modify_keys():
    layout = "HND"
    k = _keys(SAGE_SMOOTH_K_MIN_SEQ + 1, layout, shared_component=30)
    before = k.clone()
    _sage_should_smooth_k(k, layout)
    assert torch.equal(k, before)


def test_ratio_threshold_is_sane():
    assert 0.0 < SAGE_SMOOTH_K_MIN_MEAN_RATIO < 1.0
