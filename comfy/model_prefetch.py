import torch
import warnings
import weakref

import comfy_aimdo.model_vbar
from comfy.cli_args import args
import comfy.memory_management
import comfy.model_management
import comfy.ops

PREFETCH_QUEUES = []
GRAPH_MODULES = weakref.WeakSet()
GRAPH_WARMED_MODULES = weakref.WeakSet()
GRAPH_CAPTURE_STREAMS = {}

def cleanup_prefetched_modules(module, comfy_modules):
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
    if getattr(module, "_v_block_faulted", False):
        comfy_aimdo.model_vbar.vbar_unpin(module._v_block)
        del module._v_block_faulted

def _drop_graph(module):
    graph = getattr(module, "_comfy_graph", None)
    if graph is None:
        return
    # reset() through the bound method surfaces the allocator's benign
    # "uncaptured free of a captured allocation" as catchable Python warnings;
    # a plain del frees from the C++ dealloc path and spams stderr instead
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph["graph"].reset()
    del module._comfy_graph

def cleanup_prefetch_queues():
    global PREFETCH_QUEUES, GRAPH_CAPTURE_STREAMS

    for queue in PREFETCH_QUEUES:
        for entry in queue:
            if entry is None or not isinstance(entry, tuple):
                continue
            _, prefetch_state = entry
            prefetched_module, comfy_modules = prefetch_state
            if comfy_modules is not None:
                cleanup_prefetched_modules(prefetched_module, comfy_modules)
    PREFETCH_QUEUES = []
    for module in GRAPH_MODULES:
        _drop_graph(module)
    GRAPH_MODULES.clear()
    GRAPH_WARMED_MODULES.clear()
    GRAPH_CAPTURE_STREAMS = {}

def prefetch_queue_pop(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None):
    enable_graph = enable_graph and not args.disable_cuda_graphs and comfy.model_management.is_device_cuda(device) and getattr(module, "_v_block", None) is not None
    if queue is None:
        if core is not None:
            core()
        return

    capture_stream = None
    if enable_graph:
        capture_stream = GRAPH_CAPTURE_STREAMS.get(device)
        if capture_stream is None:
            capture_stream = torch.cuda.Stream(device=device)
            GRAPH_CAPTURE_STREAMS[device] = capture_stream

    signature = None
    graph_hit = False
    graph = getattr(module, "_comfy_graph", None) if enable_graph else None
    if graph is not None:
        signature = comfy_aimdo.model_vbar.vbar_fault(module._v_block)
        if signature is not None:
            module._v_block_faulted = True
            graph_hit = comfy_aimdo.model_vbar.vbar_signature_compare(signature, graph["signature"])

    consumed = queue.pop(0)
    if consumed is not None:
        offload_stream, prefetch_state = consumed
        if offload_stream is not None:
            offload_stream.wait_stream(comfy.model_management.current_stream(device))
        prefetched_module, comfy_modules = prefetch_state
        if comfy_modules is not None:
            cleanup_prefetched_modules(prefetched_module, comfy_modules)

    if graph_hit:
        queue[0] = (None, (module, []))
        graph["graph"].replay()
        return

    fully_faulted = False
    prefetch = queue[0]
    if prefetch is not None:
        comfy_modules = []
        prefetch_modules = prefetch if isinstance(prefetch, (list, tuple)) else (prefetch,)
        for root in prefetch_modules:
            for s in root.modules():
                if hasattr(s, "_v"):
                    comfy_modules.append(s)

        registerable_size = 0
        for s in comfy_modules:
            registerable_size += comfy.memory_management.vram_aligned_size([s.weight, s.bias])
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    registerable_size += lowvram_fn.memory_required()

        offload_stream, fully_faulted = comfy.ops.cast_modules_with_vbar(comfy_modules, None, device, None, True, return_faulted=True)
        if not comfy.model_management.args.fast_disk:
            comfy.model_management.ensure_pin_registerable(registerable_size)
        comfy.model_management.sync_stream(device, offload_stream)
        if fully_faulted and dtype is not None:
            for comfy_module in comfy_modules:
                comfy.ops.resolve_cast_module_with_vbar(comfy_module, dtype, device, dtype, None, False, return_weights=False)
        queue[0] = (offload_stream, (module, comfy_modules))

    if core is not None:
        if enable_graph and fully_faulted and module in GRAPH_WARMED_MODULES:
            if signature is None:
                signature = comfy_aimdo.model_vbar.vbar_fault(module._v_block)
                if signature is not None:
                    module._v_block_faulted = True
            if signature is not None:
                _drop_graph(module)
                graph = torch.cuda.CUDAGraph()
                if generator is not None:
                    graph.register_generator_state(generator)
                capture_stream.wait_stream(comfy.model_management.current_stream(device))
                with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                    core()
                comfy.model_management.current_stream(device).wait_stream(capture_stream)
                graph.replay()
                module._comfy_graph = {"graph": graph, "signature": signature}
                GRAPH_MODULES.add(module)
                return
        if capture_stream is None:
            core()
        else:
            capture_stream.wait_stream(comfy.model_management.current_stream(device))
            with torch.cuda.stream(capture_stream):
                core()
            comfy.model_management.current_stream(device).wait_stream(capture_stream)
            GRAPH_WARMED_MODULES.add(module)

def make_prefetch_queue(queue, device, transformer_options):
    if (not transformer_options.get("prefetch_dynamic_vbars", False)
        or comfy.model_management.NUM_STREAMS == 0
        or comfy.model_management.is_device_cpu(device)
        or not comfy.model_management.device_supports_non_blocking(device)):
        return None

    queue = [None] + queue + [None]
    PREFETCH_QUEUES.append(queue)
    return queue
