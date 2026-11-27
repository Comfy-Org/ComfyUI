import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_ram_pressure_keeps_current_static_model(monkeypatch):
    class ModelOutput:
        def is_dynamic(self):
            return False

    class CpuTensor:
        device = SimpleNamespace(type="cpu")

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: 1, nbytes=lambda: 1024)

    stubs = {
        "torch": {"Tensor": CpuTensor},
        "nodes": {},
        "comfy.model_patcher": {"is_model_patcher_output": lambda output: isinstance(output, ModelOutput)},
        "comfy.system_memory": {"virtual_memory_available": lambda: 0},
        "comfy_execution.graph": {"DynamicPrompt": type("DynamicPrompt", (), {})},
        "comfy_execution.graph_utils": {"is_link": lambda value: False},
    }
    with monkeypatch.context() as patch:
        for name, attributes in stubs.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            patch.setitem(sys.modules, name, module)
        path = Path(__file__).parents[2] / "comfy_execution" / "caching.py"
        spec = importlib.util.spec_from_file_location("ram_pressure_cache_under_test", path)
        caching = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(caching)

    cache = caching.RAMPressureCache(None)
    cache.generation = 2
    cache.cache = {
        "model": SimpleNamespace(outputs=[ModelOutput()]),
        "tensor": SimpleNamespace(outputs=[CpuTensor()]),
    }
    cache.used_generation = {"model": 2, "tensor": 2}
    cache.timestamps = {"model": 1, "tensor": 2}

    assert cache.ram_release(1, free_active=True) > 0
    assert "model" in cache.cache
    assert "tensor" not in cache.cache

    cache.generation = 3
    assert cache.ram_release(1) > 0
    assert "model" not in cache.cache
