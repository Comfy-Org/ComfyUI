import sys

import pytest
import torch

from comfy_api.torch_helpers.torch_tweak import _DiffusionModelCall, _TunedDiffusionModel, set_torch_tweak_wrapper


class _Named(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.dtype = torch.float16

    def forward(self, *args, **kwargs):
        return self.name


class _Arguments(torch.nn.Module):
    def forward(self, x, timesteps, context=None, transformer_options=None, **kwargs):
        return transformer_options, kwargs


def _tuned_model():
    return _TunedDiffusionModel(_Named("original"), _Named("tuned"))


def test_plain_call_runs_the_tuned_model():
    model = _tuned_model()

    result = model(torch.zeros(1), torch.zeros(1), context=torch.zeros(1), y=torch.zeros(1))

    assert result == "tuned"


def test_call_with_control_runs_the_original_model():
    model = _tuned_model()

    result = model(torch.zeros(1), torch.zeros(1), control={"output": []})

    assert result == "original"


def test_call_with_patches_runs_the_original_model():
    model = _tuned_model()

    result = model(torch.zeros(1), torch.zeros(1), transformer_options={"patches": {"attn1_patch": [print]}})

    assert result == "original"


def test_call_with_a_non_tensor_input_runs_the_original_model():
    model = _tuned_model()

    result = model(torch.zeros(1), torch.zeros(1), num_tokens=77)

    assert result == "original"


def test_attributes_come_from_the_original_model():
    model = _tuned_model()

    dtype = model.dtype

    assert dtype == torch.float16


def test_diffusion_model_call_passes_empty_transformer_options():
    call = _DiffusionModelCall(_Arguments())

    result = call(torch.zeros(1), torch.zeros(1), torch.zeros(1))

    assert result == ({}, {})


def test_missing_torch_tweak_names_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_tweak.torch", None)

    with pytest.raises(RuntimeError, match="pip install torch-tweak"):
        set_torch_tweak_wrapper(object())
