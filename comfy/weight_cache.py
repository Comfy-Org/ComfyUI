"""Optional weight preparation scoped to one fully resident sampling operation."""

import contextlib
import logging

import comfy.model_management


def cache_budget(max_bytes, free_bytes, inference_bytes, reserve_bytes):
    # Do not evict model weights to make room for an optimization cache.
    return min(max_bytes, max(0, int(free_bytes - 2 * inference_bytes - reserve_bytes)))


@contextlib.contextmanager
def sampling_weight_cache(patcher, inference_bytes, max_bytes):
    if (max_bytes <= 0 or patcher.load_device.type != "npu"
            or patcher.is_dynamic() or patcher.model.model_lowvram
            or patcher.patches or patcher.hook_patches):
        yield None
        return

    budget = cache_budget(max_bytes, patcher.get_free_memory(patcher.load_device),
                          inference_bytes, comfy.model_management.minimum_inference_memory())
    if budget == 0:
        yield None
        return

    # Optional NPU-only dependency, needed only when explicitly enabled.
    from comfy_kitchen.backends.ascend import W4A4WeightCache

    with W4A4WeightCache(budget) as cache, contextlib.ExitStack() as stack:
        for module in patcher.model.modules():
            use_cache = getattr(module, "use_weight_cache", None)
            if use_cache is not None:
                stack.enter_context(use_cache(cache))
        try:
            yield cache
        finally:
            logging.debug("W4A4 cache: %d hits, %d misses, %.1f MiB", cache.hits,
                          cache.misses, cache.bytes_used / 1024 ** 2)
