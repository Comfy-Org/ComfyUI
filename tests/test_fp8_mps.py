"""
Tests for FP8 quantization on MPS (Apple Silicon) devices.

MPS does not natively support float8_e4m3fn or float8_e5m2 dtypes.
These tests verify that:
  1. FP8 operations correctly fall back to CPU when on MPS.
  2. The round-trip (quantize on CPU -> result on original device) is numerically sound.
  3. No "Placeholder storage has not been allocated on MPS device!" errors occur.
"""
import pytest
import torch

import comfy.float
from comfy.quant_ops import TensorCoreFP8E4M3Layout, TensorCoreFP8E5M2Layout

# Skip the entire module if MPS is not available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_mps_tensor(shape=(256, 256), dtype=torch.float32):
    return torch.randn(shape, device="mps", dtype=dtype)


# ── Tests for comfy.float ────────────────────────────────────────────────────

class TestStochasticRoundingMPS:
    """Tests for comfy.float.stochastic_rounding on MPS device."""

    def test_stochastic_rounding_fp8_e4m3fn_on_mps(self):
        """stochastic_rounding must not crash when input is on MPS and target dtype is float8_e4m3fn."""
        x = _make_mps_tensor((64, 64), dtype=torch.float32)
        result = comfy.float.stochastic_rounding(x, dtype=torch.float8_e4m3fn, seed=42)

        assert result.dtype == torch.float8_e4m3fn
        assert result.shape == x.shape

    def test_stochastic_rounding_fp8_e5m2_on_mps(self):
        """stochastic_rounding must not crash when input is on MPS and target dtype is float8_e5m2."""
        x = _make_mps_tensor((64, 64), dtype=torch.float32)
        result = comfy.float.stochastic_rounding(x, dtype=torch.float8_e5m2, seed=42)

        assert result.dtype == torch.float8_e5m2
        assert result.shape == x.shape

    def test_stochastic_rounding_fp8_result_on_cpu(self):
        """Result of FP8 rounding from MPS input should be on CPU (since MPS can't hold FP8)."""
        x = _make_mps_tensor((32, 32), dtype=torch.float32)
        result = comfy.float.stochastic_rounding(x, dtype=torch.float8_e4m3fn, seed=42)

        # FP8 tensors cannot live on MPS, so result must be on CPU
        assert result.device.type == "cpu"

    def test_stochastic_rounding_non_fp8_still_works(self):
        """Non-FP8 dtypes on MPS must still work as before (no regression)."""
        x = _make_mps_tensor((32, 32), dtype=torch.float32)

        r16 = comfy.float.stochastic_rounding(x, dtype=torch.float16, seed=0)
        assert r16.dtype == torch.float16
        assert r16.device.type == "mps"

        rbf16 = comfy.float.stochastic_rounding(x, dtype=torch.bfloat16, seed=0)
        assert rbf16.dtype == torch.bfloat16
        assert rbf16.device.type == "mps"

    def test_stochastic_rounding_fp8_numerical_sanity(self):
        """FP8 round-trip (float32 -> fp8 -> float32) should have bounded error."""
        x = torch.randn(128, 128, device="mps", dtype=torch.float32)
        x_clamped = torch.clamp(x, min=-448, max=448)  # FP8 e4m3fn range

        fp8 = comfy.float.stochastic_rounding(x_clamped, dtype=torch.float8_e4m3fn, seed=123)
        # Convert back to float32 for comparison
        reconstructed = fp8.to(torch.float32)

        # Max relative error should be bounded (FP8 e4m3fn has ~0.125 relative precision)
        x_cpu = x_clamped.cpu()
        max_abs_err = (reconstructed - x_cpu).abs().max().item()
        # FP8 e4m3fn max value is 448, min subnormal ~0.001953
        # For random normal data, error should be well under 1.0
        assert max_abs_err < 2.0, f"FP8 round-trip error too large: {max_abs_err}"


class TestManualStochasticRoundMPS:
    """Tests for comfy.float.manual_stochastic_round_to_float8 on MPS device."""

    def test_manual_round_fp8_on_mps_tensor(self):
        """stochastic_rounding handles MPS generator internally without 'Placeholder storage' error."""
        x = _make_mps_tensor((16, 16), dtype=torch.float32)
        result = comfy.float.stochastic_rounding(x, dtype=torch.float8_e4m3fn, seed=42)
        assert result.dtype == torch.float8_e4m3fn


class TestNVFP4StochasticRoundMPS:
    """Tests for NVFP4 stochastic rounding on MPS - also creates FP8 tensors internally."""

    def test_nvfp4_stochastic_round_on_mps(self):
        """stochastic_round_quantize_nvfp4 creates FP8 block scales internally."""
        # NVFP4 requires 2D input with dimensions divisible by 16
        x = torch.randn(32, 32, device="mps", dtype=torch.float32)
        scale = torch.tensor(1.0, device="mps", dtype=torch.float32)

        # This should not crash - internally creates float8_e4m3fn block scales
        qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4(
            x, scale, pad_16x=False, seed=42
        )
        assert qdata.dtype == torch.uint8


# ── Tests for comfy.quant_ops (integration) ──────────────────────────────────

class TestQuantOpsMPS:
    """Tests for the quantization ops layer that calls into comfy.float."""

    def test_fp8_layout_quantize_on_mps(self):
        """TensorCoreFP8E4M3Layout.quantize must work with MPS tensors."""
        x = _make_mps_tensor((64, 64), dtype=torch.bfloat16)
        qdata, params = TensorCoreFP8E4M3Layout.quantize(
            x, scale="recalculate", stochastic_rounding=42
        )

        assert qdata.dtype == torch.float8_e4m3fn
        assert params.orig_dtype == torch.bfloat16

    def test_fp8_layout_quantize_without_stochastic_on_mps(self):
        """TensorCoreFP8E4M3Layout.quantize with stochastic_rounding=0 uses ck.quantize_per_tensor_fp8."""
        x = _make_mps_tensor((64, 64), dtype=torch.bfloat16)
        qdata, params = TensorCoreFP8E4M3Layout.quantize(
            x, scale="recalculate", stochastic_rounding=0
        )

        assert qdata.dtype == torch.float8_e4m3fn

    def test_fp8_e5m2_layout_quantize_on_mps(self):
        """TensorCoreFP8E5M2Layout.quantize must work with MPS tensors."""
        x = _make_mps_tensor((64, 64), dtype=torch.float32)
        qdata, params = TensorCoreFP8E5M2Layout.quantize(
            x, scale="recalculate", stochastic_rounding=42
        )

        assert qdata.dtype == torch.float8_e5m2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
