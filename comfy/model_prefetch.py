import comfy_aimdo.model_vbar
import comfy.memory_management
import comfy.model_management
import comfy.ops

PREFETCH_QUEUES = []

def cleanup_prefetched_modules(comfy_modules):
    for s in comfy_modules:
        prefetch = getattr(s, "_prefetch", None)
        if prefetch is None:
            continue
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        if prefetch["signature"] is not None:
            comfy_aimdo.model_vbar.vbar_unpin(s._v)
        delattr(s, "_prefetch")

def cleanup_prefetch_queues():
    global PREFETCH_QUEUES

    for queue in PREFETCH_QUEUES:
        for entry in queue:
            if entry is None or not isinstance(entry, tuple):
                continue
            _, prefetch_state = entry
            comfy_modules = prefetch_state[1]
            if comfy_modules is not None:
                cleanup_prefetched_modules(comfy_modules)
    PREFETCH_QUEUES = []

def prefetch_queue_pop(queue, device, module):
    if queue is None:
        return

    consumed = queue.pop(0)
    if consumed is not None:
        offload_stream, prefetch_state = consumed
        if offload_stream is not None:
            offload_stream.wait_stream(comfy.model_management.current_stream(device))
        _, comfy_modules = prefetch_state
        if comfy_modules is not None:
            cleanup_prefetched_modules(comfy_modules)

    prefetch = queue[0]
    if prefetch is not None:
        comfy_modules = []
        for s in prefetch.modules():
            if hasattr(s, "_v"):
                comfy_modules.append(s)

        registerable_size = 0
        for s in comfy_modules:
            registerable_size += comfy.memory_management.vram_aligned_size([s.weight, s.bias])
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    registerable_size += lowvram_fn.memory_required()

        offload_stream = comfy.ops.cast_modules_with_vbar(comfy_modules, None, device, None, True)
        if not comfy.model_management.args.fast_disk:
            comfy.model_management.ensure_pin_registerable(registerable_size)
        comfy.model_management.sync_stream(device, offload_stream)
        queue[0] = (offload_stream, (prefetch, comfy_modules))

def make_prefetch_queue(queue, device, transformer_options):
    if (not transformer_options.get("prefetch_dynamic_vbars", False)
        or comfy.model_management.NUM_STREAMS == 0
        or comfy.model_management.is_device_cpu(device)
        or not comfy.model_management.device_supports_non_blocking(device)):
        return None

    queue = [None] + queue + [None]
    PREFETCH_QUEUES.append(queue)
    return queue
