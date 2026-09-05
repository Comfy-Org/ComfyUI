import asyncio
import bisect
import heapq
import itertools
import time
import torch
from typing import Sequence, Mapping, Dict
from comfy.model_patcher import is_model_patcher_output
from comfy.system_memory import virtual_memory_available
from comfy_execution.graph import DynamicPrompt
from abc import ABC, abstractmethod

import nodes

from comfy_execution.graph_utils import is_link

NODE_CLASS_CONTAINS_UNIQUE_ID: Dict[str, bool] = {}


def include_unique_id_in_input(class_type: str) -> bool:
    if class_type in NODE_CLASS_CONTAINS_UNIQUE_ID:
        return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    NODE_CLASS_CONTAINS_UNIQUE_ID[class_type] = "UNIQUE_ID" in class_def.INPUT_TYPES().get("hidden", {}).values()
    return NODE_CLASS_CONTAINS_UNIQUE_ID[class_type]

class CacheKeySet(ABC):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        self.keys = {}
        self.subcache_keys = {}

    @abstractmethod
    async def add_keys(self, node_ids):
        raise NotImplementedError()

    def all_node_ids(self):
        return set(self.keys.keys())

    def get_used_keys(self):
        return self.keys.values()

    def get_used_subcache_keys(self):
        return self.subcache_keys.values()

    def get_data_key(self, node_id):
        return self.keys.get(node_id, None)

    def get_subcache_key(self, node_id):
        return self.subcache_keys.get(node_id, None)

class Unhashable:
    def __init__(self):
        self.value = float("NaN")

def to_hashable(obj):
    # So that we don't infinitely recurse since frozenset and tuples
    # are Sequences.
    if isinstance(obj, (int, float, str, bool, bytes, type(None))):
        return obj
    elif isinstance(obj, Mapping):
        return frozenset([(to_hashable(k), to_hashable(v)) for k, v in sorted(obj.items())])
    elif isinstance(obj, Sequence):
        return frozenset(zip(itertools.count(), [to_hashable(i) for i in obj]))
    else:
        # TODO - Support other objects like tensors?
        return Unhashable()

class CacheKeySetID(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = (node_id, node["class_type"])
            self.subcache_keys[node_id] = (node_id, node["class_type"])

class CacheKeySetInputSignature(CacheKeySet):
    def __init__(self, dynprompt, node_ids, is_changed_cache):
        super().__init__(dynprompt, node_ids, is_changed_cache)
        self.dynprompt = dynprompt
        self.is_changed_cache = is_changed_cache

    def include_node_id_in_input(self) -> bool:
        return False

    async def add_keys(self, node_ids):
        for node_id in node_ids:
            if node_id in self.keys:
                continue
            if not self.dynprompt.has_node(node_id):
                continue
            node = self.dynprompt.get_node(node_id)
            self.keys[node_id] = await self.get_node_signature(self.dynprompt, node_id)
            self.subcache_keys[node_id] = (node_id, node["class_type"])

    async def get_node_signature(self, dynprompt, node_id):
        signature = []
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)
        signature.append(await self.get_immediate_node_signature(dynprompt, node_id, order_mapping))
        for ancestor_id in ancestors:
            signature.append(await self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))
        return to_hashable(signature)

    async def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
        if not dynprompt.has_node(node_id):
            # This node doesn't exist -- we can't cache it.
            return [float("NaN")]
        node = dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        signature = [class_type, await self.is_changed_cache.get(node_id)]
        if self.include_node_id_in_input() or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT) or include_unique_id_in_input(class_type):
            signature.append(node_id)
        inputs = node["inputs"]
        for key in sorted(inputs.keys()):
            if is_link(inputs[key]):
                (ancestor_id, ancestor_socket) = inputs[key]
                ancestor_index = ancestor_order_mapping[ancestor_id]
                signature.append((key,("ANCESTOR", ancestor_index, ancestor_socket)))
            else:
                signature.append((key, inputs[key]))
        return signature

    # This function returns a list of all ancestors of the given node. The order of the list is
    # deterministic based on which specific inputs the ancestor is connected by.
    def get_ordered_ancestry(self, dynprompt, node_id):
        ancestors = []
        order_mapping = {}
        self.get_ordered_ancestry_internal(dynprompt, node_id, ancestors, order_mapping)
        return ancestors, order_mapping

    def get_ordered_ancestry_internal(self, dynprompt, node_id, ancestors, order_mapping):
        if not dynprompt.has_node(node_id):
            return
        inputs = dynprompt.get_node(node_id)["inputs"]
        input_keys = sorted(inputs.keys())
        for key in input_keys:
            if is_link(inputs[key]):
                ancestor_id = inputs[key][0]
                if ancestor_id not in order_mapping:
                    ancestors.append(ancestor_id)
                    order_mapping[ancestor_id] = len(ancestors) - 1
                    self.get_ordered_ancestry_internal(dynprompt, ancestor_id, ancestors, order_mapping)

class BasicCache:
    def __init__(self, key_class, enable_providers=False):
        self.key_class = key_class
        self.initialized = False
        self.enable_providers = enable_providers
        self.dynprompt: DynamicPrompt
        self.cache_key_set: CacheKeySet
        self.cache = {}
        self.subcaches = {}
        self._pending_store_tasks: set = set()

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.dynprompt = dynprompt
        self.cache_key_set = self.key_class(dynprompt, node_ids, is_changed_cache)
        await self.cache_key_set.add_keys(node_ids)
        self.is_changed_cache = is_changed_cache
        self.initialized = True

    def all_node_ids(self):
        assert self.initialized
        node_ids = self.cache_key_set.all_node_ids()
        for subcache in self.subcaches.values():
            node_ids = node_ids.union(subcache.all_node_ids())
        return node_ids

    def _clean_cache(self):
        preserve_keys = set(self.cache_key_set.get_used_keys())
        to_remove = []
        for key in self.cache:
            if key not in preserve_keys:
                to_remove.append(key)
        for key in to_remove:
            del self.cache[key]

    def _clean_subcaches(self):
        preserve_subcaches = set(self.cache_key_set.get_used_subcache_keys())

        to_remove = []
        for key in self.subcaches:
            if key not in preserve_subcaches:
                to_remove.append(key)
        for key in to_remove:
            del self.subcaches[key]

    def clean_unused(self):
        assert self.initialized
        self._clean_cache()
        self._clean_subcaches()

    def poll(self, **kwargs):
        pass

    def get_local(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key in self.cache:
            return self.cache[cache_key]
        return None

    def set_local(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

    async def _set_immediate(self, node_id, value):
        assert self.initialized
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

        await self._notify_providers_store(node_id, cache_key, value)

    async def _get_immediate(self, node_id):
        if not self.initialized:
            return None
        cache_key = self.cache_key_set.get_data_key(node_id)

        if cache_key in self.cache:
            return self.cache[cache_key]

        external_result = await self._check_providers_lookup(node_id, cache_key)
        if external_result is not None:
            self.cache[cache_key] = external_result
            return external_result

        return None

    async def _notify_providers_store(self, node_id, cache_key, value):
        from comfy_execution.cache_provider import (
            _has_cache_providers, _get_cache_providers,
            CacheValue, _contains_self_unequal, _logger
        )

        if not self.enable_providers:
            return
        if not _has_cache_providers():
            return
        if not self._is_external_cacheable_value(value):
            return
        if _contains_self_unequal(cache_key):
            return

        context = self._build_context(node_id, cache_key)
        if context is None:
            return
        cache_value = CacheValue(outputs=value.outputs, ui=value.ui)

        for provider in _get_cache_providers():
            try:
                if provider.should_cache(context, cache_value):
                    task = asyncio.create_task(self._safe_provider_store(provider, context, cache_value))
                    self._pending_store_tasks.add(task)
                    task.add_done_callback(self._pending_store_tasks.discard)
            except Exception as e:
                _logger.warning(f"Cache provider {provider.__class__.__name__} error on store: {e}")

    @staticmethod
    async def _safe_provider_store(provider, context, cache_value):
        from comfy_execution.cache_provider import _logger
        try:
            await provider.on_store(context, cache_value)
        except Exception as e:
            _logger.warning(f"Cache provider {provider.__class__.__name__} async store error: {e}")

    async def _check_providers_lookup(self, node_id, cache_key):
        from comfy_execution.cache_provider import (
            _has_cache_providers, _get_cache_providers,
            CacheValue, _contains_self_unequal, _logger
        )

        if not self.enable_providers:
            return None
        if not _has_cache_providers():
            return None
        if _contains_self_unequal(cache_key):
            return None

        context = self._build_context(node_id, cache_key)
        if context is None:
            return None

        for provider in _get_cache_providers():
            try:
                if not provider.should_cache(context):
                    continue
                result = await provider.on_lookup(context)
                if result is not None:
                    if not isinstance(result, CacheValue):
                        _logger.warning(f"Provider {provider.__class__.__name__} returned invalid type")
                        continue
                    if not isinstance(result.outputs, (list, tuple)):
                        _logger.warning(f"Provider {provider.__class__.__name__} returned invalid outputs")
                        continue
                    from execution import CacheEntry
                    return CacheEntry(ui=result.ui, outputs=list(result.outputs))
            except Exception as e:
                _logger.warning(f"Cache provider {provider.__class__.__name__} error on lookup: {e}")

        return None

    def _is_external_cacheable_value(self, value):
        return hasattr(value, 'outputs') and hasattr(value, 'ui')

    def _get_class_type(self, node_id):
        if not self.initialized or not self.dynprompt:
            return ''
        try:
            return self.dynprompt.get_node(node_id).get('class_type', '')
        except Exception:
            return ''

    def _build_context(self, node_id, cache_key):
        from comfy_execution.cache_provider import CacheContext, _serialize_cache_key, _logger
        try:
            cache_key_hash = _serialize_cache_key(cache_key)
            if cache_key_hash is None:
                return None
            return CacheContext(
                node_id=node_id,
                class_type=self._get_class_type(node_id),
                cache_key_hash=cache_key_hash,
            )
        except Exception as e:
            _logger.warning(f"Failed to build cache context for node {node_id}: {e}")
            return None

    async def _ensure_subcache(self, node_id, children_ids):
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        subcache = self.subcaches.get(subcache_key, None)
        if subcache is None:
            subcache = BasicCache(self.key_class)
            self.subcaches[subcache_key] = subcache
        await subcache.set_prompt(self.dynprompt, children_ids, self.is_changed_cache)
        return subcache

    def _get_subcache(self, node_id):
        assert self.initialized
        subcache_key = self.cache_key_set.get_subcache_key(node_id)
        if subcache_key in self.subcaches:
            return self.subcaches[subcache_key]
        else:
            return None

    def recursive_debug_dump(self):
        result = []
        for key in self.cache:
            result.append({"key": key, "value": self.cache[key]})
        for key in self.subcaches:
            result.append({"subcache_key": key, "subcache": self.subcaches[key].recursive_debug_dump()})
        return result

class HierarchicalCache(BasicCache):
    def __init__(self, key_class, enable_providers=False):
        super().__init__(key_class, enable_providers=enable_providers)

    def _get_cache_for(self, node_id):
        assert self.dynprompt is not None
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is None:
            return self

        hierarchy = []
        while parent_id is not None:
            hierarchy.append(parent_id)
            parent_id = self.dynprompt.get_parent_node_id(parent_id)

        cache = self
        for parent_id in reversed(hierarchy):
            cache = cache._get_subcache(parent_id)
            if cache is None:
                return None
        return cache

    async def get(self, node_id):
        cache = self._get_cache_for(node_id)
        if cache is None:
            return None
        return await cache._get_immediate(node_id)

    def get_local(self, node_id):
        cache = self._get_cache_for(node_id)
        if cache is None:
            return None
        return BasicCache.get_local(cache, node_id)

    async def set(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        await cache._set_immediate(node_id, value)

    def set_local(self, node_id, value):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        BasicCache.set_local(cache, node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        cache = self._get_cache_for(node_id)
        assert cache is not None
        return await cache._ensure_subcache(node_id, children_ids)

class NullCache:

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        pass

    def all_node_ids(self):
        return []

    def clean_unused(self):
        pass

    def poll(self, **kwargs):
        pass

    async def get(self, node_id):
        return None

    def get_local(self, node_id):
        return None

    async def set(self, node_id, value):
        pass

    def set_local(self, node_id, value):
        pass

    async def ensure_subcache_for(self, node_id, children_ids):
        return self

class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100, enable_providers=False):
        super().__init__(key_class, enable_providers=enable_providers)
        self.max_size = max_size
        self.min_generation = 0
        self.generation = 0
        self.used_generation = {}
        self.children = {}

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)
        self.generation += 1
        for node_id in node_ids:
            self._mark_used(node_id)

    def clean_unused(self):
        while len(self.cache) > self.max_size and self.min_generation < self.generation:
            self.min_generation += 1
            to_remove = [key for key in self.cache if self.used_generation[key] < self.min_generation]
            for key in to_remove:
                del self.cache[key]
                del self.used_generation[key]
                if key in self.children:
                    del self.children[key]
        self._clean_subcaches()

    async def get(self, node_id):
        self._mark_used(node_id)
        return await self._get_immediate(node_id)

    def _mark_used(self, node_id):
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.used_generation[cache_key] = self.generation

    async def set(self, node_id, value):
        self._mark_used(node_id)
        return await self._set_immediate(node_id, value)

    def set_local(self, node_id, value):
        self._mark_used(node_id)
        BasicCache.set_local(self, node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        # Just uses subcaches for tracking 'live' nodes
        await super()._ensure_subcache(node_id, children_ids)

        await self.cache_key_set.add_keys(children_ids)
        self._mark_used(node_id)
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.children[cache_key] = []
        for child_id in children_ids:
            self._mark_used(child_id)
            self.children[cache_key].append(self.cache_key_set.get_data_key(child_id))
        return self


#Small baseline weight used when a cache entry has no measurable CPU tensors.
#Keeps unknown-sized entries in eviction scoring without dominating tensor-backed entries.

RAM_CACHE_DEFAULT_RAM_USAGE = 0.05

#Exponential bias towards evicting older workflows so garbage will be taken out
#in constantly changing setups.

RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER = 1.3

RAM_CACHE_LARGE_INTERMEDIATE = 512 * 1024 ** 2

SCORE_CACHE_MIN_EXECUTION_TIME = 0.001


def all_outputs_dynamic(outputs):
    if outputs is None:
        return False

    for output in outputs:
        if isinstance(output, (list, tuple)):
            if not all_outputs_dynamic(output):
                return False
        elif not hasattr(output, "is_dynamic") or not output.is_dynamic():
            return False

    return True

class RAMPressureCache(LRUCache):

    def __init__(self, key_class, enable_providers=False):
        super().__init__(key_class, 0, enable_providers=enable_providers)
        self.timestamps = {}
        self.active_evictions = False
        self.full_evictions = False

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        self.active_evictions = False
        self.full_evictions = False
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)

    def clean_unused(self):
        self._clean_subcaches()

    async def set(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        await super().set(node_id, value)

    async def get(self, node_id):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        return await super().get(node_id)

    def set_local(self, node_id, value):
        self.timestamps[self.cache_key_set.get_data_key(node_id)] = time.time()
        super().set_local(node_id, value)

    def _ram_eviction_candidates(self, free_active, min_entry_size):
        for key, cache_entry in self.cache.items():
            if not free_active and self.used_generation[key] == self.generation:
                continue

            if all_outputs_dynamic(cache_entry.outputs) and self.used_generation[key] == self.generation:
                continue

            ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
            oom_ram_usage = ram_usage
            seen_storages = set()
            def scan_list_for_ram_usage(outputs):
                nonlocal ram_usage, oom_ram_usage
                if outputs is None:
                    return
                if isinstance(outputs, Mapping):
                    outputs = outputs.values()
                elif not isinstance(outputs, (list, tuple)):
                    outputs = (outputs,)
                for output in outputs:
                    if isinstance(output, (list, tuple, Mapping)):
                        scan_list_for_ram_usage(output)
                    elif isinstance(output, torch.Tensor) and output.device.type == 'cpu':
                        storage = output.untyped_storage()
                        storage_key = (storage.data_ptr(), storage.nbytes())
                        if storage_key not in seen_storages:
                            seen_storages.add(storage_key)
                            ram_usage += storage.nbytes()
                            oom_ram_usage += storage.nbytes()
                    elif is_model_patcher_output(output) and self.used_generation[key] != self.generation:
                        #old ModelPatchers are the first to go
                        oom_ram_usage = 1e30
                    elif hasattr(output, "_comfy_cache_tensors"):
                        scan_list_for_ram_usage(output._comfy_cache_tensors())
            scan_list_for_ram_usage(cache_entry.outputs)

            if ram_usage < min_entry_size:
                continue

            yield key, cache_entry, ram_usage, oom_ram_usage

    def _remove_entry(self, key):
        del self.cache[key]
        self.used_generation.pop(key, None)
        self.timestamps.pop(key, None)
        self.children.pop(key, None)

    def ram_release(self, target, free_active=False, min_entry_size=0):
        if virtual_memory_available() >= target:
            return 0

        clean_list = []

        for key, _, ram_usage, oom_ram_usage in self._ram_eviction_candidates(free_active, min_entry_size):
            oom_score = RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER ** (self.generation - self.used_generation[key])

            oom_score *= oom_ram_usage
            #In the case where we have no information on the node ram usage at all,
            #break OOM score ties on the last touch timestamp (pure LRU)
            bisect.insort(clean_list, (oom_score, self.timestamps[key], key, ram_usage))

        freed = 0
        while virtual_memory_available() < target and clean_list:
            _, _, key, ram_usage = clean_list.pop()
            self._remove_entry(key)
            freed += ram_usage
        if freed and free_active:
            self.active_evictions = True
            if min_entry_size == 0:
                self.full_evictions = True
        return freed


class ScoreCache(RAMPressureCache):
    def __init__(self, key_class, enable_providers=False):
        super().__init__(key_class, enable_providers=enable_providers)
        self.execution_times = {}

    def set_execution_time(self, node_id, execution_time):
        self.execution_times[self.cache_key_set.get_data_key(node_id)] = execution_time

    def _remove_entry(self, key):
        super()._remove_entry(key)
        self.execution_times.pop(key, None)

    def ram_release(self, target, free_active=False, min_entry_size=0):
        available = virtual_memory_available()
        if available >= target:
            return 0

        eviction_heap = []
        tie_breaker = itertools.count()
        for key, _, ram_usage, oom_ram_usage in self._ram_eviction_candidates(free_active, min_entry_size):
            relative_size = oom_ram_usage / max(available, 1)
            execution_time = max(self.execution_times.get(key, 0.0), SCORE_CACHE_MIN_EXECUTION_TIME)
            score = relative_size / execution_time
            heapq.heappush(eviction_heap, (-score, self.timestamps[key], next(tie_breaker), key, ram_usage))

        freed = 0
        while virtual_memory_available() < target and eviction_heap:
            _, _, _, key, ram_usage = heapq.heappop(eviction_heap)
            self._remove_entry(key)
            freed += ram_usage
        if freed and free_active:
            self.active_evictions = True
            if min_entry_size == 0:
                self.full_evictions = True
        return freed
