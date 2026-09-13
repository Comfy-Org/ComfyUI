from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import patch

import torch

from comfy_execution.caching import CacheKeySetID, RAM_CACHE_DEFAULT_RAM_USAGE, RAMPressureCache


class LazyMapping(Mapping):
    def __getitem__(self, key):
        raise AssertionError("lazy mapping was evaluated")

    def __iter__(self):
        raise AssertionError("lazy mapping was evaluated")

    def __len__(self):
        return 1


def test_ram_release_does_not_evaluate_lazy_mappings_when_accounting_for_memory():
    cache = RAMPressureCache(CacheKeySetID)
    key = "entry"
    tensor = torch.zeros(4)
    cache.cache[key] = SimpleNamespace(outputs=[LazyMapping(), {"nested": [tensor, tensor]}])
    cache.used_generation[key] = cache.generation - 1
    cache.timestamps[key] = 0

    with patch("comfy_execution.caching.virtual_memory_available", return_value=0):
        freed = cache.ram_release(1)

    assert freed == RAM_CACHE_DEFAULT_RAM_USAGE + tensor.untyped_storage().nbytes()
    assert key not in cache.cache
