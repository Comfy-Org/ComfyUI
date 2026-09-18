import logging
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

# comfy.model_management probes the CUDA device at import time, so the CPU
# override must be in place before the import below runs (a per-test
# monkeypatch fixture would apply too late). Scope it to this module instead
# of leaving it mutated for the rest of the pytest process.
_original_cpu_flag = args.cpu
if not torch.cuda.is_available():
    args.cpu = True

from comfy import model_base, model_management


@pytest.fixture(scope="module", autouse=True)
def _restore_cli_cpu_flag():
    yield
    args.cpu = _original_cpu_flag


def _model_config():
    return SimpleNamespace(
        unet_config={"disable_unet_model_creation": True},
        latent_format=None,
        manual_cast_dtype=None,
        custom_operations=None,
        optimizations={},
        memory_usage_factor=1.0,
        sampling_settings={},
    )


def test_trellis2_warns_when_ck_attention_is_enabled(monkeypatch, caplog):
    monkeypatch.setattr(model_management, "comfy_kitchen_attention_enabled", lambda: True)

    with caplog.at_level(logging.WARNING):
        model_base.Trellis2(_model_config())

    assert any("use-ck-attention" in record.message for record in caplog.records)


def test_trellis2_does_not_warn_when_ck_attention_is_disabled(monkeypatch, caplog):
    monkeypatch.setattr(model_management, "comfy_kitchen_attention_enabled", lambda: False)

    with caplog.at_level(logging.WARNING):
        model_base.Trellis2(_model_config())

    assert not any("use-ck-attention" in record.message for record in caplog.records)
