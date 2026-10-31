"""Run on a disposable CUDA host with the matching Aimdo build."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import comfy_aimdo.control as aimdo

assert aimdo.init("cuda") and aimdo.init_device(0)
from comfy import memory_management, model_management, model_patcher, model_prefetch, ops, pinned_memory, utils

memory_management.aimdo_enabled = True
device = torch.device("cuda", 0)
page = 32 << 20

for writable in (False, True):
    model = torch.nn.ModuleList([ops.WritableWeight((65536,), torch.uint8) for _ in range(2)])
    model.comfy_writable = writable
    state, file = utils.create_safetensors(model.state_dict())
    model.load_state_dict(state, assign=True)
    patcher = model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"))
    patcher.load(device)
    for priority, layer in enumerate(model):
        layer._pins = {"weights": {"balancer_priority": priority}}
    pinned_memory.pin_memory(model[0])
    pin = pinned_memory.get_pin(model[0])
    assert pin is not None
    pin.fill_(73)
    model[0]._pins["weights"]["dirty"] = writable
    with patch.object(model_management, "ensure_pin_budget", return_value=False):
        pinned_memory.pin_memory(model[1])
    if writable:
        assert pinned_memory.get_pin(model[0]) is pin
        assert pinned_memory.get_pin(model[1]) is None
        info = state["0.weight"].untyped_storage()._comfy_tensor_file_slice
        file.seek(info.offset)
        assert file.read(info.size) == bytes(info.size)
    else:
        assert pinned_memory.get_pin(model[0]) is None
        assert pinned_memory.get_pin(model[1]) is pin
    assert bool((pin == 73).all())
    del pin
    patcher.unpatch_model(device_to=torch.device("cpu"))
    vbar = model.dynamic_vbars.pop(device)
    vbar.close()
    file.close()

for fast in (False, True):
    for disable_pins in (False, True):
        model_management.DISABLE_PINNED_MEMORY = disable_pins
        model = torch.nn.ModuleList([ops.WritableWeight((size,), torch.uint8) for size in (page + 123, 13, (64 << 20) + 37)])
        model.comfy_writable = True
        state, file = utils.create_safetensors(model.state_dict())
        model.load_state_dict(state, assign=True)
        patcher = model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"), fast_disk=fast)
        patcher.load(device)
        vbar = patcher._vbar_get()
        for i, layer in enumerate(model):
            model_prefetch.pin_modules([layer], device)
            with layer.access(device) as (value, _):
                assert not value.count_nonzero()
                value.fill_(31 + 47 * i)
                value[-1] = 199 - i
            model_prefetch.cleanup_prefetched_modules(None, [layer])

        # A failed grouped prefetch must release its partial pins, not consume future groups.
        original_fault = vbar.fault
        fault_count = 0
        def fail_second_fault(*args):
            global fault_count
            fault_count += 1
            if fault_count == 2:
                raise RuntimeError("injected prefetch failure")
            return original_fault(*args)
        vbar.fault = fail_second_fault
        queue = model_prefetch.make_prefetch_queue([(model[0], model[1]), (model[2],)], device, {"prefetch_dynamic_vbars": True})
        assert queue is not None
        try:
            model_prefetch.prefetch_queue_pop(queue, device, model[0])
            raise AssertionError("fault injection did not run")
        except RuntimeError as error:
            assert str(error) == "injected prefetch failure"
        finally:
            del vbar.fault
            torch.cuda.synchronize()
            model_prefetch.cleanup_prefetch_queues()
        assert not any(page_state & 2 for page_state in vbar.get_residency())

        # Evict only the last page: the whole tensor must survive in a cast buffer.
        last = max(model, key=lambda m: m._v[1] + m._v[2])
        end = last._v[1] - vbar.base_addr + last._v[2]
        vbar.set_watermark((end - 1) // page * page)
        patcher.partially_unload_ram(1e32)
        for i, layer in enumerate(model):
            model_prefetch.pin_modules([layer], device)
            with layer.access(device) as (value, _):
                assert bool((value[:-1] == 31 + 47 * i).all()) and value[-1].item() == 199 - i
            model_prefetch.cleanup_prefetched_modules(None, [layer])

        # Continue fully offloaded, including tensors below the force-load threshold.
        vbar.set_watermark(0)
        for layer in model:
            model_prefetch.pin_modules([layer], device)
            with layer.access(device) as (value, _):
                value.add_(1)
            model_prefetch.cleanup_prefetched_modules(None, [layer])
        patcher.partially_unload_ram(1e32)
        for i, layer in enumerate(model):
            model_prefetch.pin_modules([layer], device)
            with layer.access(device) as (value, _):
                assert bool((value[:-1] == 32 + 47 * i).all()) and value[-1].item() == 200 - i
            model_prefetch.cleanup_prefetched_modules(None, [layer])
        patcher.unpatch_model(device_to=torch.device("cpu"))
        assert patcher.loaded_ram_size() == 0
        for i, layer in enumerate(model):
            cpu = torch.empty(layer.weight.shape, dtype=layer.weight.dtype)
            assert memory_management.read_tensor_file_slice_into(layer.weight, cpu)
            assert bool((cpu[:-1] == 32 + 47 * i).all()) and cpu[-1].item() == 200 - i
        model.dynamic_vbars.clear()
        vbar.close()
        file.close()

source = torch.arange(17, dtype=torch.float32, device=device)
state, file = utils.create_safetensors({"weight": source})
memory_management.write_tensor_file_slice(state["weight"], source)
restored = torch.empty(17)
assert memory_management.read_tensor_file_slice_into(state["weight"], restored)
assert torch.equal(restored, source.cpu())
file.close()
