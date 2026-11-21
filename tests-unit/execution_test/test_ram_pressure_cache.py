from types import SimpleNamespace

from comfy_execution.caching import RAMPressureCache


def test_ram_release_tied_entries_with_different_key_types(monkeypatch):
    # Same oom score and same timestamp must not fall through to comparing the cache keys.
    cache = object.__new__(RAMPressureCache)
    keys = [("1", "CheckpointLoaderSimple"), frozenset({(0, "CLIPLoader")})]
    cache.cache = {key: SimpleNamespace(outputs=[object()]) for key in keys}
    cache.used_generation = {key: 0 for key in keys}
    cache.timestamps = {key: 1.0 for key in keys}
    cache.children = {}
    cache.generation = 1
    cache.active_evictions = False
    cache.full_evictions = False
    monkeypatch.setattr("comfy_execution.caching.virtual_memory_available", lambda: 0)

    assert cache.ram_release(target=1) > 0
    assert cache.cache == {}
