from types import SimpleNamespace
from unittest.mock import Mock

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_patcher
from comfy.hooks import EnumHookMode


def test_patch_hook_weight_backup_uses_transfer_result(monkeypatch):
    weight = torch.arange(4, dtype=torch.bfloat16)
    backup = torch.empty_like(weight)
    patcher = SimpleNamespace(
        model=object(),
        hook_backup={},
        hook_mode=EnumHookMode.MinVram,
        offload_device=torch.device("cpu"),
    )

    monkeypatch.setattr(comfy.model_patcher, "get_key_weight", lambda model, key: (weight, None, None))
    transfer = Mock(return_value=backup)
    monkeypatch.setattr(comfy.model_patcher.comfy.model_management, "cast_to", transfer)
    monkeypatch.setattr(comfy.model_patcher.comfy.model_management, "cast_to_device", lambda *args, **kwargs: weight.float())
    monkeypatch.setattr(comfy.model_patcher.comfy.lora, "calculate_weight", lambda *args, **kwargs: weight.float())
    monkeypatch.setattr(comfy.model_patcher.comfy.float, "stochastic_rounding", lambda tensor, dtype, seed: tensor.to(dtype))
    monkeypatch.setattr(comfy.model_patcher.comfy.utils, "copy_to_param", lambda *args, **kwargs: None)

    original_weights = {"layer.weight": weight}
    comfy.model_patcher.ModelPatcher.patch_hook_weight_to_device(
        patcher,
        hooks=None,
        combined_patches={"layer.weight": []},
        key="layer.weight",
        original_weights=original_weights,
        memory_counter=None,
    )

    transfer.assert_called_once_with(weight, dtype=weight.dtype, device=torch.device("cpu"), copy=True)
    assert patcher.hook_backup["layer.weight"][0] is backup
    assert patcher.hook_backup["layer.weight"][1] == weight.device
    assert "layer.weight" not in original_weights
