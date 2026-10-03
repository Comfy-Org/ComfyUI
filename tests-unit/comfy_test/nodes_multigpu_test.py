from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as model_management
from comfy_extras.nodes_multigpu import _force_supported_compute_dtype


@pytest.fixture
def supported_gpu(monkeypatch):
    monkeypatch.setattr(model_management, "should_use_fp16", lambda *args, **kwargs: True)
    monkeypatch.setattr(model_management, "should_use_bf16", lambda *args, **kwargs: True)


@pytest.mark.parametrize("prioritize_fp16", [False, True])
@pytest.mark.parametrize("supported_dtypes, expected_dtype", [
    ([torch.bfloat16, torch.float32], torch.bfloat16),
    ([torch.float32], torch.float32),
    ([torch.float16, torch.bfloat16, torch.float32], torch.float16),
])
def test_select_model_device_respects_supported_dtypes(monkeypatch, supported_gpu, prioritize_fp16, supported_dtypes, expected_dtype):
    monkeypatch.setattr(model_management, "PRIORITIZE_FP16", prioritize_fp16)
    patcher = Mock()
    patcher.model_dtype.return_value = torch.float8_e4m3fn
    patcher.model.model_config.supported_inference_dtypes = supported_dtypes

    _force_supported_compute_dtype(patcher, torch.device("cuda:0"))

    patcher.set_model_compute_dtype.assert_called_once_with(expected_dtype)


@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_select_model_device_keeps_native_dtype(supported_gpu, weight_dtype):
    patcher = Mock()
    patcher.model_dtype.return_value = weight_dtype
    patcher.model.model_config.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    _force_supported_compute_dtype(patcher, torch.device("cuda:0"))

    patcher.set_model_compute_dtype.assert_not_called()


def test_select_model_device_uses_float32_without_low_precision_support(monkeypatch):
    monkeypatch.setattr(model_management, "should_use_fp16", lambda *args, **kwargs: False)
    monkeypatch.setattr(model_management, "should_use_bf16", lambda *args, **kwargs: False)
    patcher = Mock()
    patcher.model_dtype.return_value = torch.float8_e4m3fn
    patcher.model.model_config.supported_inference_dtypes = [torch.bfloat16, torch.float32]

    _force_supported_compute_dtype(patcher, torch.device("cpu"))

    patcher.set_model_compute_dtype.assert_called_once_with(torch.float32)
