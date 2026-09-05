"""Regression test for init_kv_cache crashing when the text encoder runs on CPU."""

from __future__ import annotations

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops as ops  # noqa: E402
import comfy.text_encoders.llama as llama  # noqa: E402


def _make_model():
    config = llama.Llama2Config(
        vocab_size=16, hidden_size=8, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )
    config.head_dim = 4
    config.fixed_kv = True
    return llama.Llama2_(config, device="cpu", dtype=torch.float32, ops=ops.manual_cast)


def test_init_kv_cache_on_cpu_does_not_probe_cuda_capability(monkeypatch):
    def _raise_if_not_cuda(device):
        if torch.device(device).type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        return True

    monkeypatch.setattr(
        llama.comfy_kitchen, "flash_attention_decode_is_available", _raise_if_not_cuda
    )

    model = _make_model()
    past = model.init_kv_cache(1, 4, torch.device("cpu"), torch.float32)

    assert not isinstance(past[0], llama.FixedKV)
