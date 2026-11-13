import pytest
import torch

import comfy.float


@pytest.fixture
def rng_numels(monkeypatch):
    """Stub the CK kernel and record the numel of every rng buffer it is
    handed, so the slicing arithmetic can be checked without comfy_kitchen
    or a GPU."""
    seen = []

    def fake_kernel(value, rng, dtype):
        assert rng.numel() == value.numel()
        seen.append(rng.numel())
        return torch.zeros_like(value, dtype=dtype)

    monkeypatch.setattr(comfy.float, "_CK_STOCHASTIC_ROUNDING_AVAILABLE", True)
    monkeypatch.setattr(comfy.float, "_ck_stochastic_rounding_fp8", fake_kernel)
    monkeypatch.setattr(comfy.float, "_CK_SLICE_NUMEL", 1024)  # Shrink the bound to keep the shapes below small
    return seen


@pytest.mark.parametrize("shape", [
    (32, 32),      # Within the bound, takes the zero-copy path
    (64, 64),      # Splits evenly across rows
    (100, 41),     # Splits across rows, not a round multiple
    (2, 5000),     # A single row is wider than the bound
    (1, 20000),    # One row only, far over the bound
    (20000, 1),    # Many rows, one element each
])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_ck_rng_buffer_never_exceeds_slice_bound(rng_numels, shape, dtype):
    value = torch.zeros(shape, dtype=torch.float16)

    out = comfy.float.stochastic_rounding(value, dtype)

    assert rng_numels, "the CK path was not exercised"
    assert max(rng_numels) <= comfy.float._CK_SLICE_NUMEL
    assert out.shape == value.shape
    assert out.dtype == dtype


def test_ck_single_slice_allocates_no_output_buffer(rng_numels, monkeypatch):
    monkeypatch.setattr(
        torch, "empty_like",
        lambda *a, **k: pytest.fail("allocated an output buffer for a single slice"),
    )

    value = torch.zeros((16, 16), dtype=torch.float16)
    comfy.float.stochastic_rounding(value, torch.float8_e4m3fn)

    assert rng_numels == [256]
