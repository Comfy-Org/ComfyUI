from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import patch

import torch

from comfy.cli_args import args

args.cpu = True

from comfy_execution.caching import CacheKeySetID, RAM_CACHE_DEFAULT_RAM_USAGE, RAMPressureCache


class LazyMapping(Mapping):
    def __getitem__(self, key):
        raise AssertionError("lazy mapping was evaluated")

    def __iter__(self):
        raise AssertionError("lazy mapping was evaluated")

    def __len__(self):
        return 1


class ExplicitCacheTensors:
    def __init__(self, tensor):
        self.tensor = tensor

    def _comfy_cache_tensors(self):
        return {"nested": [self.tensor, (self.tensor, LazyMapping())]}


def test_ram_release_does_not_evaluate_lazy_mappings_when_accounting_for_memory():
    cache = RAMPressureCache(CacheKeySetID)
    key = "entry"
    tensor = torch.zeros(4, dtype=torch.float32)
    cache.cache[key] = SimpleNamespace(outputs=[LazyMapping(), ExplicitCacheTensors(tensor)])
    cache.used_generation[key] = cache.generation - 1
    cache.timestamps[key] = 0

    with patch("comfy_execution.caching.virtual_memory_available", return_value=0):
        freed = cache.ram_release(1)

    assert freed == RAM_CACHE_DEFAULT_RAM_USAGE + 16
    assert key not in cache.cache
