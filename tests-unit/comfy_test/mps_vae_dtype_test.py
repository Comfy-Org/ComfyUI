"""VAE dtype on Apple GPUs: fp16 preferred when the VAE runs there, flags and --force-fp32 still win."""
import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management as mm  # noqa: E402

MPS, CPU = torch.device("mps"), torch.device("cpu")
WORKING = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture
def flags(monkeypatch):
    for name in ("fp16_vae", "bf16_vae", "fp32_vae"):
        monkeypatch.setattr(mm.args, name, False)
    monkeypatch.setattr(mm, "FORCE_FP32", False)
    return monkeypatch


def test_vae_on_mps_prefers_fp16(flags):
    # bf16 is usable and listed first, as on macOS 14+; fp16 must still win.
    flags.setattr(mm, "should_use_bf16", lambda device=None: True)
    assert mm.vae_dtype(MPS, WORKING) == torch.float16


def test_vae_without_device_resolves_mps_and_prefers_fp16(flags):
    flags.setattr(mm, "vae_device", lambda: MPS)
    flags.setattr(mm, "should_use_bf16", lambda device=None: True)
    assert mm.vae_dtype(None, WORKING) == torch.float16


def test_vae_on_mps_without_fp16_keeps_the_usual_choice(flags):
    flags.setattr(mm, "should_use_bf16", lambda device=None: True)
    assert mm.vae_dtype(MPS, [torch.bfloat16, torch.float32]) == torch.bfloat16


def test_cpu_vae_on_an_mps_host_is_not_forced_to_fp16(flags):
    """--cpu-vae puts the VAE on the CPU; the Apple GPU preference must not follow it there."""
    flags.setattr(mm, "mps_mode", lambda: True)
    flags.setattr(mm, "should_use_fp16", lambda device=None: False)
    flags.setattr(mm, "should_use_bf16", lambda device=None: False)
    assert mm.vae_dtype(CPU, WORKING) == torch.float32


def test_force_fp32_wins_on_mps(flags):
    flags.setattr(mm, "FORCE_FP32", True)
    flags.setattr(mm, "should_use_fp16", lambda device=None: False)
    flags.setattr(mm, "should_use_bf16", lambda device=None: False)
    assert mm.vae_dtype(MPS, WORKING) == torch.float32


@pytest.mark.parametrize("flag, expected", [("fp32_vae", torch.float32), ("bf16_vae", torch.bfloat16), ("fp16_vae", torch.float16)])
def test_explicit_vae_flags_win_on_mps(flags, flag, expected):
    flags.setattr(mm.args, flag, True)
    assert mm.vae_dtype(MPS, WORKING) == expected
