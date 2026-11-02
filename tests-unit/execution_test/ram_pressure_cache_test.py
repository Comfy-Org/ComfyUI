from types import SimpleNamespace

import pytest
import torch

import comfy_execution.caching as caching

ENTRY_SIZE = 1024 * 1024
GB = 1024 ** 3


@pytest.mark.parametrize(
    "total_ram_gb, cache_ram, expected",
    [
        (16, [], (2.0, 16.0)),
        (64, [], (6.4, 64.0)),
        (96, [], (9.6, 24.0)),
        (128, [], (10.0, 32.0)),
        (512, [], (10.0, 128.0)),
        (16, [8], (8, 16.0)),
        (128, [20], (20, 32.0)),
        (128, [40], (40, 40.0)),
        (16, [8, 3], (8, 3)),
    ],
)
def test_ram_cache_thresholds(total_ram_gb, cache_ram, expected):
    assert caching.ram_cache_thresholds(total_ram_gb, cache_ram) == pytest.approx(expected)


def ram_cache(monkeypatch, free_ram):
    cache = caching.RAMPressureCache(caching.CacheKeySetInputSignature)
    cache.generation = 2
    # Two entries from the previous prompt, one from the current one
    for key, generation in (("old_a", 1), ("old_b", 1), ("current", 2)):
        cache.cache[key] = SimpleNamespace(outputs=[[torch.zeros(ENTRY_SIZE, dtype=torch.uint8)]])
        cache.used_generation[key] = generation
        cache.timestamps[key] = 0.0

    # Evicting an entry gives its RAM back
    monkeypatch.setattr(caching, "virtual_memory_available", lambda: free_ram + (3 - len(cache.cache)) * ENTRY_SIZE)
    return cache


def test_unreachable_inactive_target_drops_other_prompt_and_counts_it(monkeypatch):
    cache = ram_cache(monkeypatch, free_ram=45 * GB)

    cache.ram_release(int(caching.ram_cache_thresholds(64, [])[1] * GB))

    assert list(cache.cache) == ["current"]
    assert cache.inactive_evicted == 2
    assert cache.inactive_freed >= 2 * ENTRY_SIZE
    assert not cache.active_evictions


def test_default_keeps_other_prompt_when_ram_is_free(monkeypatch):
    # 128GB machine with 45GB free
    cache = ram_cache(monkeypatch, free_ram=45 * GB)

    cache.ram_release(int(caching.ram_cache_thresholds(128, [])[1] * GB))

    assert len(cache.cache) == 3
    assert cache.inactive_evicted == 0


def test_active_release_is_not_counted_as_inactive(monkeypatch):
    cache = ram_cache(monkeypatch, free_ram=0)

    cache.ram_release(100 * ENTRY_SIZE, free_active=True)

    assert len(cache.cache) == 0
    assert cache.active_evictions
    assert cache.inactive_evicted == 0
