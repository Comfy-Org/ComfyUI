"""Regression tests for the Stable Audio 3 VAE dtype on MPS."""

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.audio.vae_sa3  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402


class StubSA3AudioVAE(nn.Module):
    # The dtype is picked in comfy.sd.VAE, so skip building the real decoder.
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(2, 2)


@pytest.fixture
def load_sa3_vae(monkeypatch):
    monkeypatch.setattr(comfy.ldm.audio.vae_sa3, "SA3AudioVAE", StubSA3AudioVAE)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)

    def load(cpu_state, device):
        monkeypatch.setattr(mm, "cpu_state", cpu_state)
        monkeypatch.setattr(mm, "vae_device", lambda: device)
        return comfy.sd.VAE(sd={"decoder.layers.3.transformers.0.pre_norm.alpha": torch.zeros(1)})
    return load


def test_sa3_vae_uses_fp16_on_mps(monkeypatch, load_sa3_vae):
    # bf16 is allowed on macOS >= 14 but decodes this VAE to noise.
    monkeypatch.setattr(mm, "mac_version", lambda: (15, 0))
    vae = load_sa3_vae(mm.CPUState.MPS, torch.device("mps"))
    assert vae.vae_dtype == torch.float16


def test_sa3_vae_keeps_bf16_on_cuda(monkeypatch, load_sa3_vae):
    monkeypatch.setattr(mm, "should_use_fp16", lambda device=None, **kwargs: True)
    monkeypatch.setattr(mm, "should_use_bf16", lambda device=None, **kwargs: True)
    vae = load_sa3_vae(mm.CPUState.GPU, torch.device("cuda"))
    assert vae.vae_dtype == torch.bfloat16
