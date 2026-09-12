import bisect

import comfy.model_management
import comfy.memory_management
import comfy.utils
import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import torch

from comfy.cli_args import args

def _add_to_bucket(module, module_pin, buckets, size, priority):
    bucket = buckets.setdefault(size, [])
    entry = [-priority, 0, module]
    entry[1] = id(entry)
    bisect.insort(bucket, entry)
    module_pin["balancer_entry"] = entry

def _steal_pin(module, stack, buckets, size, priority, subset):
    bucket = buckets.get(size)
    if bucket is None:
        return False

    while bucket and bucket[-1][-1] is None:
        bucket.pop()
    if not bucket:
        del buckets[size]
        return False

    if priority <= -bucket[-1][0]:
        return False

    *_, victim = bucket.pop()
    module_pin = module._pins[subset]
    victim_pin = victim._pins[subset]
    module_pin["pin"] = victim_pin["pin"]
    module_pin["registered"] = victim_pin["registered"]
    module_pin["stack_index"] = victim_pin["stack_index"]
    stack_index = module_pin["stack_index"]
    stack[stack_index] = (module, stack[stack_index][1])

    victim_pin["registered"] = False
    del victim_pin["pin"]
    del victim_pin["stack_index"]
    del victim_pin["balancer_entry"]

    _add_to_bucket(module, module_pin, buckets, size, priority)
    return True

def get_pin(module, subset="weights"):
    pins = module.__dict__.get("_pins")
    module_pin = None if pins is None else pins.get(subset)
    pin = None if module_pin is None else module_pin.get("pin")
    if pin is None or module_pin["registered"] or args.disable_pinned_memory:
        return pin

    _, _, stack_split, pinned_size, *_ = module._pin_state[subset]
    size = pin.nbytes
    comfy.model_management.ensure_pin_registerable(size)

    if torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1) != 0:
        comfy.model_management.discard_cuda_async_error()
        comfy.model_management.free_registrations(size)
        if torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1) != 0:
            comfy.model_management.discard_cuda_async_error()
            return pin

    module_pin["registered"] = True
    stack_split[0] = max(stack_split[0], module_pin["stack_index"])
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    pinned_size[0] += size
    return pin

def pin_memory(module, subset="weights", size=None):
    pin_state = module._pin_state
    if args.disable_pinned_memory:
        return

    pin = get_pin(module, subset)
    if pin is not None:
        return

    pins = module.__dict__.setdefault("_pins", {})
    module_pin = pins.setdefault(subset, {})
    hostbuf, stack, stack_split, pinned_size, counter, buckets = pin_state[subset]
    if size is None:
        size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    registerable_size = size
    loaded = subset.endswith("-loaded")
    priority = module_pin.get("balancer_priority")

    if priority is None:
        priority = comfy.utils.bit_reverse_range(counter[0], 16)
        counter[0] += 1
        module_pin["balancer_priority"] = priority

    comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
    if (not comfy.model_management.ensure_pin_budget(size, loaded=loaded) or
        not comfy.model_management.ensure_pin_registerable(registerable_size)):
        return _steal_pin(module, stack, buckets, size, priority, subset)

    offset = hostbuf.size
    extended = False
    try:
        hostbuf.extend(size=size, register=False)
        extended = True
        pin = comfy_aimdo.torch.hostbuf_to_tensor(hostbuf)[offset:offset + size]
        pin.untyped_storage()._comfy_hostbuf = hostbuf
        if torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1) != 0:
            comfy.model_management.discard_cuda_async_error()
            comfy.model_management.free_registrations(size)
            if torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1) != 0:
                comfy.model_management.discard_cuda_async_error()
                del pin
                hostbuf.truncate(offset, do_unregister=False)
                return _steal_pin(module, stack, buckets, size, priority, subset)
    except RuntimeError:
        if extended:
            hostbuf.truncate(offset, do_unregister=False)
        return _steal_pin(module, stack, buckets, size, priority, subset)

    module_pin["pin"] = pin
    stack.append((module, offset))
    module_pin["registered"] = True
    module_pin["stack_index"] = len(stack) - 1
    stack_split[0] = max(stack_split[0], module_pin["stack_index"])
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    pinned_size[0] += size
    _add_to_bucket(module, module_pin, buckets, size, priority)
    return True
