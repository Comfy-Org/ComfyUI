import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.model_management as model_management


class _Patcher:
    def __init__(self, device, dynamic, loaded_size=100, partial_unload_limit=None):
        self.model = object()
        self.load_device = device
        self.offload_device = torch.device("cpu")
        self.dynamic = dynamic
        self.loaded_bytes = loaded_size
        self.partial_unload_limit = partial_unload_limit
        self.detached = False
        self.partial_unloads = []

    def is_dynamic(self):
        return self.dynamic

    def model_patches_models(self):
        return []

    def is_clone(self, other):
        return False

    def loaded_size(self):
        return self.loaded_bytes

    def model_size(self):
        return 100

    def partially_unload(self, device, memory_to_free):
        self.partial_unloads.append((device, memory_to_free))
        freed = min(self.loaded_bytes, memory_to_free)
        if self.partial_unload_limit is not None:
            freed = min(freed, self.partial_unload_limit)
        self.loaded_bytes -= freed
        return freed


class _LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.currently_used = True

    def __eq__(self, other):
        return self.model is other.model

    def is_dead(self):
        return False

    def model_memory(self):
        return 100

    def model_memory_required(self, device):
        return 100

    def model_loaded_memory(self):
        return 0

    def model_offloaded_memory(self):
        return 0

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        pass

    def model_unload(self, memory_to_free):
        self.model.detached = True
        return True


def _mock_model_management(monkeypatch, loaded_models):
    monkeypatch.setattr(model_management, "LoadedModel", _LoadedModel)
    monkeypatch.setattr(model_management, "current_loaded_models", loaded_models)
    monkeypatch.setattr(model_management, "cleanup_models_gc", lambda: None)
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(model_management, "minimum_inference_memory", lambda: 1)
    monkeypatch.setattr(model_management, "extra_reserved_memory", lambda: 0)
    monkeypatch.setattr(model_management, "get_free_memory", lambda device, torch_free_too=False: (0, 0) if torch_free_too else 0)
    monkeypatch.setattr(model_management, "ensure_pin_budget", lambda size: True)
    monkeypatch.setattr(model_management, "ensure_pin_registerable", lambda size: True)
    monkeypatch.setattr(model_management, "lowvram_available", False)


def test_free_memory_keeps_dynamic_ram_cache_after_sufficient_partial_unload(monkeypatch):
    device = torch.device("cuda:0")
    dynamic = _Patcher(device, dynamic=True)
    dynamic_loaded = _LoadedModel(dynamic)
    _mock_model_management(monkeypatch, [dynamic_loaded])

    unloaded = model_management.free_memory(80, device, retain_ram_cache=True)

    assert unloaded == []
    assert dynamic_loaded in model_management.current_loaded_models
    assert dynamic.detached is False
    assert dynamic.loaded_size() == 20


def test_loading_static_model_keeps_dynamic_model_ram_cache(monkeypatch):
    device = torch.device("cuda:0")
    dynamic = _Patcher(device, dynamic=True)
    static = _Patcher(device, dynamic=False)
    dynamic_loaded = _LoadedModel(dynamic)
    _mock_model_management(monkeypatch, [dynamic_loaded])

    model_management.load_models_gpu([static])

    assert dynamic_loaded in model_management.current_loaded_models
    assert dynamic.detached is False
    assert dynamic.partial_unloads
    assert dynamic.partial_unloads[0][1] > 100
    assert dynamic.loaded_size() == 0


def test_free_memory_detaches_dynamic_model_after_insufficient_partial_unload(monkeypatch):
    device = torch.device("cuda:0")
    dynamic = _Patcher(device, dynamic=True, partial_unload_limit=40)
    dynamic_loaded = _LoadedModel(dynamic)
    _mock_model_management(monkeypatch, [dynamic_loaded])

    unloaded = model_management.free_memory(80, device, retain_ram_cache=True)

    assert unloaded == [dynamic_loaded]
    assert dynamic_loaded not in model_management.current_loaded_models
    assert dynamic.detached is True
    assert dynamic.loaded_size() == 60
