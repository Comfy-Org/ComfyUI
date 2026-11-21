"""Regression tests for text encoder placement on Apple Silicon (MPS)."""

import json

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as mm  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


class DummyTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        pass


class DummyTEModel(torch.nn.Module):
    def __init__(self, device=None, dtype=None, model_options=None):
        super().__init__()
        self.linear = comfy.ops.manual_cast.Linear(8, 8, dtype=dtype, device=device)
        self.dtypes = set([dtype])
        self.built_on = device

    def load_sd(self, sd):
        return ([], [])


class Target:
    params = {}
    clip = DummyTEModel
    tokenizer = DummyTokenizer


def quantized_state_dict(quant_format):
    return [{
        "linear.weight": torch.zeros((8, 8), dtype=torch.int8),
        "linear.comfy_quant": torch.tensor(list(json.dumps({"format": quant_format}).encode()), dtype=torch.uint8),
        "spiece_model": b"",
    }]


def load_clip(dtype, state_dict=None, parameters=0):
    return comfy.sd.CLIP(target=Target(), parameters=parameters, model_options={"dtype": dtype}, state_dict=state_dict or [], disable_dynamic=True)


@pytest.fixture(autouse=True)
def apple_silicon(monkeypatch):
    monkeypatch.setattr(mm, "cpu_state", mm.CPUState.MPS)
    monkeypatch.setattr(mm, "vram_state", mm.VRAMState.SHARED)
    yield
    monkeypatch.undo()
    mm.unload_all_models()


def test_text_encoder_device_is_mps():
    assert mm.text_encoder_device() == torch.device("mps")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_text_encoder_loads_on_mps(dtype):
    assert load_clip(dtype).patcher.load_device == torch.device("mps")


@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=str)
def test_fp8_text_encoder_is_built_on_cpu(dtype):
    # Big enough that it would otherwise be built on mps, then shifted back.
    assert load_clip(dtype, parameters=2 * 1024 ** 3).cond_stage_model.built_on == torch.device("cpu")


@pytest.mark.parametrize("state_dict", [
    [{"linear.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn)}],
    {"linear.weight": torch.zeros((8, 8), dtype=torch.float8_e5m2)},
    *[quantized_state_dict(f) for f in ("float8_e4m3fn", "mxfp8", "nvfp4", "asym_w4a8_int8", "w6a8_int8")],
], ids=["fp8_weights", "full_model_fp8_weights", "float8_e4m3fn", "mxfp8", "nvfp4", "asym_w4a8_int8", "w6a8_int8"])
def test_quantized_text_encoder_stays_on_cpu(state_dict):
    assert load_clip(torch.float16, state_dict).patcher.load_device == torch.device("cpu")


@pytest.mark.parametrize("quant_format", ["int8_tensorwise", "convrot_w4a4"])
def test_int8_quantized_text_encoder_loads_on_mps(quant_format):
    assert load_clip(torch.float16, quantized_state_dict(quant_format)).patcher.load_device == torch.device("mps")


def test_quantized_text_encoder_stays_on_cuda(monkeypatch):
    # Devices that can cast fp8 skip the state dict scan.
    monkeypatch.setattr(mm, "text_encoder_device", lambda: torch.device("cuda"))
    state_dict = [{"linear.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn)}]
    assert load_clip(torch.float16, state_dict).patcher.load_device == torch.device("cuda")
