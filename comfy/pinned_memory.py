import bisect

import comfy.model_management
import comfy.memory_management
import comfy.utils
import comfy_aimdo.host_buffer
import comfy_aimdo.torch
import torch

from comfy.cli_args import args

def _add_to_bucket(module, buckets, size, priority):
    bucket = buckets.setdefault(size, [])
    entry = [-priority, 0, module]
    entry[1] = id(entry)
    bisect.insort(bucket, entry)
    module._pin_balancer_entry = entry

def _steal_pin(module, stack, buckets, size, priority):
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
    module._pin = victim._pin
    module._pin_registered = victim._pin_registered
    module._pin_stack_index = victim._pin_stack_index
    stack[module._pin_stack_index] = (module, stack[module._pin_stack_index][1])

    victim._pin_registered = False
    del victim._pin
    del victim._pin_stack_index
    del victim._pin_balancer_entry

    _add_to_bucket(module, buckets, size, priority)
    return True

def get_pin(module, subset="weights"):
    pin = getattr(module, "_pin", None)
    if pin is None or module._pin_registered or args.disable_pinned_memory:
        return pin

    _, _, stack_split, pinned_size, *_ = module._pin_state[subset]
    size = pin.nbytes
    comfy.model_management.ensure_pin_registerable(size)

    if torch.cuda.cudart().cudaHostRegister(pin.data_ptr(), size, 1) != 0:
        comfy.model_management.discard_cuda_async_error()
        return pin

    module._pin_registered = True
    stack_split[0] = max(stack_split[0], module._pin_stack_index)
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

    hostbuf, stack, stack_split, pinned_size, counter, buckets = pin_state[subset]
    if size is None:
        size = comfy.memory_management.vram_aligned_size([ module.weight, module.bias ])
    offset = hostbuf.size
    registerable_size = size
    priority = getattr(module, "_pin_balancer_priority", None)

    if priority is None:
        priority = comfy.utils.bit_reverse_range(counter[0], 16)
        counter[0] += 1
        module._pin_balancer_priority = priority

    comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
    if (not comfy.model_management.ensure_pin_budget(size) or
        not comfy.model_management.ensure_pin_registerable(registerable_size)):
        return _steal_pin(module, stack, buckets, size, priority)

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
                return _steal_pin(module, stack, buckets, size, priority)
    except RuntimeError:
        if extended:
            hostbuf.truncate(offset, do_unregister=False)
        return _steal_pin(module, stack, buckets, size, priority)

    module._pin = pin
    stack.append((module, offset))
    module._pin_registered = True
    module._pin_stack_index = len(stack) - 1
    stack_split[0] = max(stack_split[0], module._pin_stack_index)
    comfy.model_management.TOTAL_PINNED_MEMORY += size
    pinned_size[0] += size
    _add_to_bucket(module, buckets, size, priority)
    return True
