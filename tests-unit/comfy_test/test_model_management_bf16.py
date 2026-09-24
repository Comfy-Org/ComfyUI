import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as mm


class FakeDeviceProperties:
    def __init__(self, gcn_arch_name):
        self.gcnArchName = gcn_arch_name


def _patch_amd(monkeypatch, gcn_arch_name):
    monkeypatch.setattr(mm, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(mm, "mps_mode", lambda: False)
    monkeypatch.setattr(mm, "cpu_mode", lambda: False)
    monkeypatch.setattr(mm, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(mm, "is_ascend_npu", lambda: False)
    monkeypatch.setattr(mm, "is_ixuca", lambda: False)
    monkeypatch.setattr(mm, "is_amd", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: FakeDeviceProperties(gcn_arch_name)
    )


def test_should_use_bf16_false_for_gfx1032(monkeypatch):
    _patch_amd(monkeypatch, "gfx1032:sramecc+:xnack-")

    assert mm.should_use_bf16(device=object()) is False


def test_should_use_bf16_still_false_for_gfx1030(monkeypatch):
    _patch_amd(monkeypatch, "gfx1030:sramecc+:xnack-")

    assert mm.should_use_bf16(device=object()) is False
