import importlib
from types import SimpleNamespace

import torch

from comfy.cli_args import args

args.cpu = True

llama = importlib.import_module("comfy.text_encoders.llama")


def make_model():
    model = llama.Llama2_.__new__(llama.Llama2_)
    model.fixed_kv = True
    model.config = SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=2,
        head_dim=8,
    )
    return model


def test_init_kv_cache_uses_regular_cache_for_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(llama.comfy_kitchen, "flash_attention_decode_is_available", lambda device: True)

    cache = make_model().init_kv_cache(1, 4, torch.device("cpu"), torch.float32)[0]

    assert isinstance(cache, tuple)
    assert not isinstance(cache, llama.FixedKV)


def test_init_kv_cache_uses_flash_cache_for_supported_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(llama.comfy_kitchen, "flash_attention_decode_is_available", lambda device: True)

    cache = make_model().init_kv_cache(1, 4, torch.device("cpu"), torch.float32)[0]

    assert isinstance(cache, llama.FixedKV)
