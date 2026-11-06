import logging

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.lora  # noqa: E402


def _missing_key_warnings(caplog):
    return [r for r in caplog.records if "lora keys not loaded" in r.getMessage()]


def test_load_lora_aggregates_missing_key_warning(caplog):
    lora = {"unrelated.{}.lora_up.weight".format(i): None for i in range(20)}

    with caplog.at_level(logging.WARNING):
        comfy.lora.load_lora(lora, {})

    warnings = _missing_key_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "20 of 20" in message
    assert "unrelated.0.lora_up.weight" in message
    assert "..." in message


def test_load_lora_lists_all_missing_keys_when_few(caplog):
    lora = {"a.weight": None, "b.weight": None}

    with caplog.at_level(logging.WARNING):
        comfy.lora.load_lora(lora, {})

    warnings = _missing_key_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2 of 2: a.weight, b.weight" in message
    assert "..." not in message


def test_load_lora_no_warning_when_all_keys_loaded(caplog):
    lora = {"a.config": 1}

    with caplog.at_level(logging.WARNING):
        comfy.lora.load_lora(lora, {})

    assert _missing_key_warnings(caplog) == []
