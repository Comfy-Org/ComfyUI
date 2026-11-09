"""Bounded persistent-state lifecycle checks; run on a disposable CUDA host."""
import os
import gc
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import safetensors.torch
import torch
import comfy_aimdo.control as aimdo

assert aimdo.init("cuda") and aimdo.init_device(0)
import comfy.ldm.wan.model_animate2  # noqa: F401 -- initialize the registry before the patcher
from comfy import memory_management, model_management as mm, model_patcher, model_prefetch, model_state, ops
import comfy.system_memory

memory_management.aimdo_enabled = True
device = torch.device("cuda", 0)


class Cache(torch.nn.Module):
    comfy_writable = True

    def __init__(self):
        super().__init__()
        self.k = ops.WritableWeight((512, 1024), torch.float32)
        self.v = ops.WritableWeight((512, 1024), torch.float32)


def write(model, value):
    queue = model_prefetch.make_prefetch_queue([model], device, {"prefetch_dynamic_vbars": True})
    def core():
        with model.k.access(device) as (k, _), model.v.access(device) as (v, _):
            k.fill_(value)
            v.fill_(-value)
    model_prefetch.prefetch_queue_pop(queue, device, module=model, core=core)
    model_prefetch.prefetch_queue_pop(queue, device, None)


with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
    owner = model_patcher.ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
    path = os.path.join(directory, "state.safetensors")
    first = model_state.ModelState()
    first.owner = owner
    cache = first.load(Cache(), device)
    patcher = first.patcher
    patch_uuid = patcher.patches_uuid
    write(cache, 3)
    address = cache.k._v[1]
    first.metadata["turn"] = 1
    first.save(path, writethrough=False)
    assert first.tensors is None and not os.path.exists(path)
    reopened = model_state.ModelState.open(path, "never")
    assert reopened._values.version == first._values.version and reopened.patcher is first.patcher
    del reopened
    second = first.copy()
    assert second.load(Cache(), device) is cache
    assert second.patcher is patcher and cache.k._v[1] == address
    assert not os.path.exists(path)
    write(cache, 5)
    second.metadata["turn"] = 2
    second.save(path, writethrough=False)
    assert not os.path.exists(path)
    # A VRAM eviction writes into RAM, but does not commit the named save.
    patcher._vbar_get().set_watermark(0)
    assert patcher.loaded_size() == 0 and patcher.loaded_ram_size() > 0
    assert not os.path.exists(path)
    patcher.partially_unload_ram(1e32)
    assert patcher.loaded_ram_size() == 0
    assert second.tensors is None
    assert safetensors.torch.load_file(path)["k.weight"].eq(5).all()
    assert not patcher.get_attachment("model_state_saves")
    second.flush()
    assert second.tensors["v.weight"].eq(-5).all()
    assert first.load(Cache(), device) is cache
    first.flush()
    assert first.tensors["k.weight"].eq(3).all()
    assert second.load(Cache(), device) is cache
    second.flush()
    assert second.tensors["k.weight"].eq(5).all()
    assert patcher.patches_uuid == patch_uuid and cache.current_weight_patches_uuid == patch_uuid
    # Explicit unload must commit a still-resident dirty state too.
    third = second.copy()
    third.load(Cache(), device)
    write(cache, 7)
    cache.k.writeback(cache.k._v_weight)
    third.save(path, writethrough=False)
    queue = model_prefetch.make_prefetch_queue([cache], device, {"prefetch_dynamic_vbars": True})
    model_prefetch.prefetch_queue_pop(queue, device, cache)
    prefetched = queue[0]
    patcher.partially_unload_ram(1e32)
    assert queue in model_prefetch.PREFETCH_QUEUES and queue[0] is prefetched
    model_prefetch.prefetch_queue_pop(queue, device, None)
    assert safetensors.torch.load_file(path)["k.weight"].eq(7).all()
    # Explicit unload must also commit a still-resident snapshot.
    fourth = third.copy()
    fourth.load(Cache(), device)
    write(cache, 9)
    fourth.save(path, writethrough=False)
    mm.unload_model_and_clones(patcher)
    assert safetensors.torch.load_file(path)["k.weight"].eq(9).all()
    collected = model_state.ModelState()
    orphan = collected.load(Cache(), device)
    write(orphan, 11)
    collected.save(path, writethrough=False)
    del collected, orphan
    gc.collect()
    assert safetensors.torch.load_file(path)["k.weight"].eq(11).all()
    assert safetensors.torch.load_file(path)["v.weight"].eq(-11).all()
    pressure = model_state.ModelState()
    pressured = pressure.load(Cache(), device)
    write(pressured, 13)
    pressure.patcher.partially_unload_ram(1e32)
    pressured.k.writeback(pressured.k._v_weight)
    assert pressure.patcher.loaded_ram_size() == pressured.k.weight.nbytes
    pressure.save(path, writethrough=False)
    pressure.patcher.set_in_use_by_current_prompt(False)
    mm.reset_cast_buffers()
    available = comfy.system_memory.virtual_memory_available
    comfy.system_memory.virtual_memory_available = lambda: 0
    try:
        # Reclaim the first RAM pin while the second weight's native pageout is running.
        pressure.patcher._vbar_get().set_watermark(0)
    finally:
        comfy.system_memory.virtual_memory_available = available
    assert safetensors.torch.load_file(path)["k.weight"].eq(13).all()
    assert safetensors.torch.load_file(path)["v.weight"].eq(-13).all()
    assert not pressure.patcher.get_attachment("model_state_saves")
    assert pressure.tensors is None
    mm.unload_model_and_clones(pressure.patcher)
    other_owner = model_patcher.ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
    moved = model_state.ModelState()
    moved.owner = other_owner
    write(moved.load(Cache(), device), 17)
    source_patcher = moved.patcher
    moved.owner = owner
    write(moved.load(Cache(), device), 19)
    assert moved.patcher is patcher
    other = model_state.ModelState()
    other.owner = other_owner
    write(other.load(Cache(), device), 23)
    moved.flush()
    assert moved.tensors["k.weight"].eq(19).all()
    assert moved.tensors["v.weight"].eq(-19).all()
    unused = moved.copy()
    signature = cache.k._v_signature
    unused.release()
    assert cache.k._v_signature == signature and signature is not None
    assert patcher.get_attachment("model_state") is moved._values
    mm.unload_model_and_clones(source_patcher)
    mm.unload_model_and_clones(patcher)
    print("PERSISTENT_STATE_CUDA_PASS", flush=True)  # noqa: T201
