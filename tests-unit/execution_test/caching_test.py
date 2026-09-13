from types import SimpleNamespace

import torch

from comfy_execution.caching import RAMPressureCache, ScoreCache


def make_cache(cache_type, entries, execution_times=None):
    cache = cache_type(None)
    cache.cache = entries
    cache.generation = 1
    cache.used_generation = {key: 1 for key in entries}
    cache.timestamps = {key: index for index, key in enumerate(entries)}
    if execution_times is not None:
        cache.execution_times = execution_times
    return cache


def release_one_entry(monkeypatch, cache):
    availability = iter((0, 0, 1))
    monkeypatch.setattr("comfy_execution.caching.virtual_memory_available", lambda: next(availability))
    cache.ram_release(1, free_active=True)


def test_score_cache_evicts_fast_large_output_first(monkeypatch):
    entries = {
        "fast-large": SimpleNamespace(outputs=[torch.empty(1024, dtype=torch.uint8)]),
        "slow-small": SimpleNamespace(outputs=[torch.empty(128, dtype=torch.uint8)]),
    }
    cache = make_cache(ScoreCache, entries, {"fast-large": 0.1, "slow-small": 10.0})

    release_one_entry(monkeypatch, cache)

    assert "fast-large" not in cache.cache
    assert "fast-large" not in cache.execution_times
    assert "slow-small" in cache.cache


def test_score_cache_handles_missing_execution_time(monkeypatch):
    entries = {
        "unknown": SimpleNamespace(outputs=[torch.empty(128, dtype=torch.uint8)]),
        "measured": SimpleNamespace(outputs=[torch.empty(128, dtype=torch.uint8)]),
    }
    cache = make_cache(ScoreCache, entries, {"measured": 1.0})

    release_one_entry(monkeypatch, cache)

    assert "unknown" not in cache.cache
    assert "measured" in cache.cache


def test_ram_pressure_cache_still_evicts_largest_output_first(monkeypatch):
    entries = {
        "large": SimpleNamespace(outputs=[torch.empty(1024, dtype=torch.uint8)]),
        "small": SimpleNamespace(outputs=[torch.empty(128, dtype=torch.uint8)]),
    }
    cache = make_cache(RAMPressureCache, entries)

    release_one_entry(monkeypatch, cache)

    assert "large" not in cache.cache
    assert "small" in cache.cache
