from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.hooks as hooks  # noqa: E402
import comfy.lora as lora  # noqa: E402
import comfy.sd as sd  # noqa: E402


def _contract(**overrides):
    values = {
        "api": 1,
        "architecture": "h3_keyless_core50_v1",
        "core_blocks": 50,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": 2,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "projection_attr": "qv_proj",
        "qv_order": "q_effective;v",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _KeylessBase:
    def __init__(self, contract=None):
        self.diffusion_model = SimpleNamespace(
            **{lora.KEYLESS_H3_CONTRACT_KEY: contract or _contract()}
        )
        self.model_config = SimpleNamespace(unet_config={})

    def state_dict(self):
        return {}


class _Patcher:
    def __init__(self, base=None):
        self.model = base or _KeylessBase()
        self.injections = {}

    def clone(self):
        return _Patcher(self.model)

    def add_patches(self, loaded, strength):
        return tuple(loaded)

    def add_hook_patches(self, hook, patches, strength_patch):
        return tuple(patches)

    def set_injections(self, key, injections):
        self.injections[key] = injections


def _stub_lora_mapping(monkeypatch):
    monkeypatch.setattr(lora, "model_lora_keys_unet", lambda model, key_map={}: key_map)
    monkeypatch.setattr(sd.comfy.lora_convert, "convert_lora", lambda value: value)
    monkeypatch.setattr(lora, "load_lora", lambda value, key_map, log_missing=True: {})


def test_keyless_adapter_guard_rejects_canonical_contract():
    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        lora._reject_unsupported_keyless_h3_adapter(_KeylessBase())


def test_keyless_adapter_guard_rejects_malformed_contract():
    model = _KeylessBase(_contract(routing_source="key"))

    with pytest.raises(
        ValueError,
        match="malformed minimax_h3_keyless_contract_v1.*routing_source",
    ):
        lora._reject_unsupported_keyless_h3_adapter(model)


def test_keyless_adapter_guard_ignores_models_without_marker():
    model = SimpleNamespace(diffusion_model=SimpleNamespace())

    assert lora._reject_unsupported_keyless_h3_adapter(model) is None


def test_standard_lora_rejects_nonzero_keyless_model_strength_before_mapping(monkeypatch):
    patcher = _Patcher()

    def mapping_must_not_run(model, key_map={}):
        raise AssertionError("Keyless guard must run before generic model LoRA mapping")

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping_must_not_run)

    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        sd.load_lora_for_models(patcher, None, {}, 1.0, 0.0)


def test_standard_lora_allows_zero_model_strength_to_reach_model_mapping(monkeypatch):
    patcher = _Patcher()
    calls = []

    def mapping(model, key_map={}):
        calls.append(model)
        return key_map

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping)
    monkeypatch.setattr(sd.comfy.lora_convert, "convert_lora", lambda value: value)
    monkeypatch.setattr(lora, "load_lora", lambda value, key_map, log_missing=True: {})

    model_out, clip_out = sd.load_lora_for_models(patcher, None, {}, 0.0, 0.0)

    assert calls == [patcher.model]
    assert model_out is not patcher
    assert clip_out is None


def test_bypass_lora_rejects_nonzero_keyless_model_strength_before_mapping(monkeypatch):
    patcher = _Patcher()

    def mapping_must_not_run(model, key_map={}):
        raise AssertionError("Keyless guard must run before bypass LoRA mapping")

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping_must_not_run)

    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        sd.load_bypass_lora_for_models(patcher, None, {}, 1.0, 0.0)


def test_hook_lora_rejects_nonzero_keyless_model_strength_before_mapping(monkeypatch):
    patcher = _Patcher()

    def mapping_must_not_run(model, key_map={}):
        raise AssertionError("Keyless guard must run before hook LoRA mapping")

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping_must_not_run)

    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        hooks.load_hook_lora_for_models(patcher, None, {}, 1.0, 0.0)


def test_lazy_weight_hook_rejects_nonzero_keyless_model_strength_before_mapping(monkeypatch):
    patcher = _Patcher()
    hook = hooks.WeightHook(strength_model=1.0, strength_clip=0.0)
    hook.weights = {}
    monkeypatch.setattr(hook, "should_register", lambda *args, **kwargs: True)

    def mapping_must_not_run(model, key_map={}):
        raise AssertionError("Keyless guard must run before lazy hook LoRA mapping")

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping_must_not_run)

    with pytest.raises(ValueError, match="Generic LoRA/DoRA loading is not supported"):
        hook.add_hook_patches(
            patcher,
            {},
            {"target": hooks.EnumWeightTarget.Model},
            set(),
        )


def test_lazy_weight_hook_allows_zero_keyless_model_strength(monkeypatch):
    patcher = _Patcher()
    hook = hooks.WeightHook(strength_model=0.0, strength_clip=0.0)
    hook.weights = {}
    monkeypatch.setattr(hook, "should_register", lambda *args, **kwargs: True)
    calls = []

    def mapping(model, key_map={}):
        calls.append(model)
        return key_map

    monkeypatch.setattr(lora, "model_lora_keys_unet", mapping)
    monkeypatch.setattr(lora, "load_lora", lambda value, key_map, log_missing=True: {})

    assert hook.add_hook_patches(
        patcher,
        {},
        {"target": hooks.EnumWeightTarget.Model},
        set(),
    )
    assert calls == [patcher.model]
