import pytest
from unittest.mock import patch, MagicMock
import torch

import comfy.model_management as mm


class FakeDeviceProps:
    """Minimal stand-in for torch.cuda.get_device_properties return value."""
    def __init__(self, major, minor, name="FakeGPU"):
        self.major = major
        self.minor = minor
        self.name = name


class TestSupportsFp8Compute:
    """Tests for per-device fp8 compute capability detection."""

    def test_cpu_device_returns_false(self):
        assert mm.supports_fp8_compute(torch.device("cpu")) is False

    @pytest.mark.skipif(not hasattr(torch.backends, "mps"), reason="MPS backend not available")
    def test_mps_device_returns_false(self):
        assert mm.supports_fp8_compute(torch.device("mps")) is False

    @patch("comfy.model_management.SUPPORT_FP8_OPS", True)
    def test_cli_override_returns_true(self):
        assert mm.supports_fp8_compute(torch.device("cpu")) is True

    @patch("comfy.model_management.get_torch_device", return_value=torch.device("cpu"))
    def test_none_device_defaults_to_get_torch_device(self, mock_get):
        result = mm.supports_fp8_compute(None)
        mock_get.assert_called_once()
        assert result is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_each_cuda_device_checked_independently(self):
        """On a multi-GPU system, each device should be queried for its own capabilities."""
        count = torch.cuda.device_count()
        if count < 2:
            pytest.skip("Need 2+ CUDA devices for multi-GPU test")
        results = {}
        for i in range(count):
            dev = torch.device(f"cuda:{i}")
            results[i] = mm.supports_fp8_compute(dev)
            props = torch.cuda.get_device_properties(dev)
            # Verify the result is consistent with the device's compute capability
            if props.major >= 9:
                assert results[i] is True, f"cuda:{i} ({props.name}) has SM {props.major}.{props.minor}, should support fp8"
            elif props.major < 8 or props.minor < 9:
                assert results[i] is False, f"cuda:{i} ({props.name}) has SM {props.major}.{props.minor}, should not support fp8"

    @patch("torch.version.cuda", None)
    @patch("comfy.model_management.SUPPORT_FP8_OPS", False)
    def test_rocm_build_returns_false(self):
        """On ROCm, devices appear as cuda:N via HIP but torch.version.cuda is None."""
        dev = MagicMock()
        dev.type = "cuda"
        assert mm.supports_fp8_compute(dev) is False

    @patch("torch.version.cuda", "12.4")
    @patch("comfy.model_management.SUPPORT_FP8_OPS", False)
    @patch("torch.cuda.get_device_properties")
    def test_sm89_supports_fp8(self, mock_props):
        """Ada Lovelace (SM 8.9, e.g. RTX 4080) should support fp8."""
        mock_props.return_value = FakeDeviceProps(major=8, minor=9)
        dev = torch.device("cuda:0")
        assert mm.supports_fp8_compute(dev) is True

    @patch("torch.version.cuda", "12.4")
    @patch("comfy.model_management.SUPPORT_FP8_OPS", False)
    @patch("torch.cuda.get_device_properties")
    def test_sm86_does_not_support_fp8(self, mock_props):
        """Ampere (SM 8.6, e.g. RTX 3090) should not support fp8."""
        mock_props.return_value = FakeDeviceProps(major=8, minor=6)
        dev = torch.device("cuda:0")
        assert mm.supports_fp8_compute(dev) is False

    @patch("torch.version.cuda", "12.4")
    @patch("comfy.model_management.SUPPORT_FP8_OPS", False)
    @patch("torch.cuda.get_device_properties")
    def test_sm90_supports_fp8(self, mock_props):
        """Hopper (SM 9.0) and above should support fp8."""
        mock_props.return_value = FakeDeviceProps(major=9, minor=0)
        dev = torch.device("cuda:0")
        assert mm.supports_fp8_compute(dev) is True


class TestSupportsNvfp4Compute:
    """Tests for per-device nvfp4 compute capability detection."""

    def test_cpu_device_returns_false(self):
        assert mm.supports_nvfp4_compute(torch.device("cpu")) is False

    @patch("torch.version.cuda", "12.4")
    @patch("torch.cuda.get_device_properties")
    def test_sm100_supports_nvfp4(self, mock_props):
        """Blackwell (SM 10.0) should support nvfp4."""
        mock_props.return_value = FakeDeviceProps(major=10, minor=0)
        dev = torch.device("cuda:0")
        assert mm.supports_nvfp4_compute(dev) is True

    @patch("torch.version.cuda", "12.4")
    @patch("torch.cuda.get_device_properties")
    def test_sm89_does_not_support_nvfp4(self, mock_props):
        """Ada Lovelace (SM 8.9) should not support nvfp4."""
        mock_props.return_value = FakeDeviceProps(major=8, minor=9)
        dev = torch.device("cuda:0")
        assert mm.supports_nvfp4_compute(dev) is False
