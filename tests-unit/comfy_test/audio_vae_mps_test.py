"""Stable Audio 3 VAE dtype selection on Apple Silicon (MPS).

The SA3 audio decoder cannot run in bf16 on backends whose attention
softmax runs in the input dtype (mps, cpu): its DyT qk norms emit values
around +-30..50, attention logits land where bf16's 8-bit mantissa
quantizes in steps of ~0.25-0.5, and the decoded audio is broadband
noise, fully decorrelated from the fp32 decode (corr ~0). fp16 matches
fp32 (corr 0.9997). The VAE must therefore not default to bf16 on mps,
while CUDA keeps its bf16 default.
"""

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
    cli_args.cpu = True

import comfy.ldm.audio.vae_sa3  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402

mps_only = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and mm.is_device_mps(mm.get_torch_device())),
    reason="requires an Apple Silicon MPS device",
)


def sa3_state_dict():
    # Routes VAE() into the Stable Audio 3 branch (small config).
    return {"decoder.layers.3.transformers.0.pre_norm.alpha": torch.zeros(1)}


class StubSA3AudioVAE(nn.Module):
    # Stands in for SA3AudioVAE so tests don't build the full 100M+ param
    # module; dtype selection under test happens outside the model class.
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(2, 2)


@pytest.fixture
def sa3_vae_factory(monkeypatch):
    def build(device):
        monkeypatch.setattr(comfy.ldm.audio.vae_sa3, "SA3AudioVAE", StubSA3AudioVAE)
        monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
        monkeypatch.setattr(mm, "vae_device", lambda: device)
        return comfy.sd.VAE(sd=sa3_state_dict())

    return build


def test_sa3_audio_vae_does_not_default_to_bf16_on_mps(monkeypatch, sa3_vae_factory):
    # On macOS >= 14 should_use_bf16() approves bf16 for mps, which is the
    # dtype the audio decode corrupts under; pin the version so the test
    # exercises that path on any host.
    monkeypatch.setattr(mm, "mac_version", lambda: (15, 0))
    vae = sa3_vae_factory(torch.device("mps"))
    assert vae.vae_dtype != torch.bfloat16
    assert vae.vae_dtype == torch.float16


def test_sa3_audio_vae_keeps_bf16_default_on_cuda(monkeypatch, sa3_vae_factory):
    # The mps exclusion must not leak: CUDA-like devices (bf16 approved)
    # keep preferring bf16.
    monkeypatch.setattr(mm, "should_use_fp16", lambda device=None, **kwargs: True)
    monkeypatch.setattr(mm, "should_use_bf16", lambda device=None, **kwargs: True)
    vae = sa3_vae_factory(torch.device("cuda"))
    assert vae.vae_dtype == torch.bfloat16


def test_sa3_audio_vae_explicit_dtype_still_wins(sa3_vae_factory, monkeypatch):
    # --bf16-vae style overrides bypass working_dtypes entirely.
    monkeypatch.setattr(comfy.ldm.audio.vae_sa3, "SA3AudioVAE", StubSA3AudioVAE)
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    vae = comfy.sd.VAE(sd=sa3_state_dict(), device=torch.device("cpu"), dtype=torch.bfloat16)
    assert vae.vae_dtype == torch.bfloat16


@mps_only
def test_sa3_audio_vae_picks_fp16_on_mps_hardware(monkeypatch, sa3_vae_factory):
    vae = sa3_vae_factory(mm.get_torch_device())
    assert vae.vae_dtype == torch.float16
