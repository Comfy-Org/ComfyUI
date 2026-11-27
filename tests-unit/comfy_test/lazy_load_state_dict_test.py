from unittest import mock

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.memory_management  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402


@pytest.mark.parametrize("cls", [comfy.ops.disable_weight_init.Linear, comfy.ops.manual_cast.Linear])
def test_lazy_load_missing_keys_keeps_meta_dtype_and_shape(monkeypatch, cls):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    layer = cls(4, 8, bias=True, dtype=torch.bfloat16)

    missing, unexpected = layer.load_state_dict({}, strict=False, assign=True)

    assert unexpected == []
    assert missing == ["weight", "bias"]
    assert layer.weight.is_meta
    assert tuple(layer.weight.shape) == (8, 4)
    assert layer.weight.dtype == torch.bfloat16
    assert layer.bias.is_meta
    assert tuple(layer.bias.shape) == (8,)
    assert layer.bias.dtype == torch.bfloat16


def test_lazy_load_no_bias_skips_bias_placeholder(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    layer = comfy.ops.disable_weight_init.Linear(4, 8, bias=False, dtype=torch.float16)

    missing, unexpected = layer.load_state_dict({}, strict=False, assign=True)

    assert unexpected == []
    assert missing == ["weight"]
    assert layer.bias is None
    assert layer.weight.is_meta
    assert tuple(layer.weight.shape) == (8, 4)
    assert layer.weight.dtype == torch.float16


def test_lazy_load_two_stage_partial_preserves_existing_param(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    layer = comfy.ops.disable_weight_init.Linear(4, 8, bias=True, dtype=torch.bfloat16)

    weight = torch.full((8, 4), 2.5, dtype=torch.bfloat16)
    missing_first, unexpected_first = layer.load_state_dict({"weight": weight}, strict=False, assign=True)
    assert unexpected_first == []
    assert missing_first == ["bias"]
    assert layer.weight.data.data_ptr() == weight.data_ptr()
    assert layer.bias.is_meta

    bias = torch.ones(8, dtype=torch.bfloat16)
    missing_second, unexpected_second = layer.load_state_dict({"bias": bias}, strict=False, assign=True)
    assert unexpected_second == []
    assert missing_second == ["weight"]
    assert not layer.weight.is_meta
    assert not layer.bias.is_meta
    assert torch.equal(layer.weight, weight)
    assert torch.equal(layer.bias, bias)


def test_lazy_load_embedding_missing_key_keeps_meta_and_materializes_on_use(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    emb = comfy.ops.disable_weight_init.Embedding(16, 8, dtype=torch.bfloat16)

    missing, unexpected = emb.load_state_dict({}, strict=False, assign=True)
    assert unexpected == []
    assert missing == ["weight"]
    assert emb.weight.is_meta
    assert tuple(emb.weight.shape) == (16, 8)
    assert emb.weight.dtype == torch.bfloat16

    out = emb(torch.tensor([1, 3]))
    assert not emb.weight.is_meta
    assert emb.weight.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros((2, 8), dtype=torch.bfloat16))

    emb_host = comfy.ops.disable_weight_init.Embedding(16, 8, dtype=torch.bfloat16)
    emb_host.load_state_dict({}, strict=False, assign=True)
    host_out = emb_host.host_rows(torch.tensor([0, 2]), out_dtype=torch.float32)
    assert not emb_host.weight.is_meta
    assert torch.equal(host_out, torch.zeros((2, 8), dtype=torch.float32))


@pytest.mark.parametrize("cls", [comfy.ops.disable_weight_init.Linear, comfy.ops.manual_cast.Linear])
def test_forward_materializes_missing_meta_params_on_demand(monkeypatch, cls):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    layer = cls(4, 8, bias=True, dtype=torch.bfloat16)
    layer.load_state_dict({}, strict=False, assign=True)

    out = layer(torch.ones((2, 4), dtype=torch.bfloat16))

    assert not layer.weight.is_meta
    assert not layer.bias.is_meta
    assert tuple(layer.weight.shape) == (8, 4)
    assert layer.weight.dtype == torch.bfloat16
    assert tuple(layer.bias.shape) == (8,)
    assert layer.bias.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros((2, 8), dtype=torch.bfloat16))


def test_forward_on_meta_input_keeps_meta_weights(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    layer = comfy.ops.disable_weight_init.Linear(4, 8, bias=True, dtype=torch.bfloat16)
    layer.load_state_dict({}, strict=False, assign=True)

    out = layer(torch.empty((2, 4), device="meta", dtype=torch.bfloat16))

    assert out.is_meta
    assert tuple(out.shape) == (2, 8)
    assert layer.weight.is_meta
    assert layer.bias.is_meta


def test_model_patcher_dynamic_force_loads_tiny_missing_linear(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(comfy.model_patcher.comfy_aimdo.host_buffer, "HostBuffer", mock.MagicMock())
    cpu = torch.device("cpu")

    model = nn.Module()
    model.model_loaded_weight_memory = 0
    layer = comfy.ops.manual_cast.Linear(4, 8, bias=True, dtype=torch.bfloat16)
    model.proj = layer
    model.load_state_dict({}, strict=False, assign=True)

    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcher.__init__(patcher, model, cpu, cpu)
    comfy.model_patcher.ModelPatcherDynamic.__init__(patcher, model, cpu, cpu)
    patcher.load(device_to=cpu)

    assert not layer.weight.is_meta
    assert not layer.bias.is_meta
    assert tuple(layer.weight.shape) == (8, 4)
    assert layer.weight.dtype == torch.bfloat16
    assert torch.equal(layer(torch.ones((1, 4), dtype=torch.bfloat16)), torch.zeros((1, 8), dtype=torch.bfloat16))
