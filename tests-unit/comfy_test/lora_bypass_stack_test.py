import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.lora  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
from comfy.model_patcher import ModelPatcher  # noqa: E402

DIM, RANK = 8, 2


@pytest.fixture(autouse=True)
def cpu_adapters(monkeypatch):
    # the bypass hook moves adapter weights to get_torch_device(); keep them next to the CPU test modules on GPU machines
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.o = torch.nn.Linear(DIM, DIM)


def _lora():
    return {"diffusion_model.o.lora_down.weight": torch.randn(RANK, DIM),
            "diffusion_model.o.lora_up.weight": torch.randn(DIM, RANK),
            "diffusion_model.o.alpha": torch.tensor(float(RANK))}


def _delta(lora, strength):
    return strength * lora["diffusion_model.o.lora_up.weight"] @ lora["diffusion_model.o.lora_down.weight"]


def test_stacked_bypass_loras_all_apply(monkeypatch):
    torch.manual_seed(0)
    model = _Model()
    monkeypatch.setattr(comfy.lora, "model_lora_keys_unet", lambda model, key_map: {**key_map, "diffusion_model.o": "diffusion_model.o.weight"})
    o = model.diffusion_model.o
    x = torch.randn(3, DIM)
    base = o(x)
    first, second = _lora(), _lora()
    expected = torch.nn.functional.linear(x, o.weight + _delta(first, 0.5) + _delta(second, 0.8), o.bias)

    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher = comfy.sd.load_bypass_lora_for_models(patcher, None, first, 0.5, 0)[0]
    patcher = comfy.sd.load_bypass_lora_for_models(patcher, None, second, 0.8, 0)[0]
    for _ in range(2):  # both LoRAs wrap the same module; eject must unwind them for the next inject
        patcher.inject_model()
        try:
            torch.testing.assert_close(o(x), expected)
        finally:
            patcher.eject_model()
        assert torch.equal(o(x), base)


class _Clip:
    """The parts of comfy.sd.CLIP that load_bypass_lora_for_models uses."""

    def __init__(self, patcher):
        self.patcher = patcher
        self.cond_stage_model = patcher.model

    def clone(self):
        return _Clip(self.patcher.clone())


def test_stacked_bypass_loras_all_apply_to_clip(monkeypatch):
    torch.manual_seed(0)
    model = _Model()
    monkeypatch.setattr(comfy.lora, "model_lora_keys_clip", lambda model, key_map: {**key_map, "diffusion_model.o": "diffusion_model.o.weight"})
    o = model.diffusion_model.o
    x = torch.randn(3, DIM)
    base = o(x)
    first, second = _lora(), _lora()
    expected = torch.nn.functional.linear(x, o.weight + _delta(first, 0.5) + _delta(second, 0.8), o.bias)

    clip = _Clip(ModelPatcher(model, torch.device("cpu"), torch.device("cpu")))
    clip = comfy.sd.load_bypass_lora_for_models(None, clip, first, 0, 0.5)[1]
    clip = comfy.sd.load_bypass_lora_for_models(None, clip, second, 0, 0.8)[1]
    for _ in range(2):
        clip.patcher.inject_model()
        try:
            torch.testing.assert_close(o(x), expected)
        finally:
            clip.patcher.eject_model()
        assert torch.equal(o(x), base)
