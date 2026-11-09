"""Bounded hybrid Qwen generation against plain, resident, pinned and file-only model states. Run on a disposable CUDA host."""
import json
import os
from pathlib import Path
import sys
import tempfile
import weakref

os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import safetensors.torch
import torch
import comfy_aimdo.control as aimdo

assert aimdo.init("cuda") and aimdo.init_device(0)
from comfy.cli_args import args
args.use_ck_attention = True
import comfy.ldm.wan.model_animate2  # noqa: F401 -- initialize the model registry before the patcher
from comfy import memory_management, model_management as mm, model_patcher, model_prefetch as prefetch, model_state, ops, storage
from comfy.text_encoders import qwen35 as qwen
from comfy_kitchen.int8_decode import is_available

memory_management.aimdo_enabled = True
device = torch.device("cuda", 0)
qwen.QWEN35_MODELS["probe"] = dict(vision=dict(hidden_size=32, num_heads=2, intermediate_size=64, depth=1))

class Probe(qwen.Qwen35):
    model_type = "probe"

config = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
              layer_types=qwen._qwen35_layer_types(4), num_attention_heads=2, num_key_value_heads=1,
              head_dim=256, linear_num_key_heads=2, linear_num_value_heads=2,
              linear_key_head_dim=128, linear_value_head_dim=128, mtp=True, lm_head=True, stop_tokens=[])

with torch.inference_mode(), tempfile.TemporaryDirectory() as tmp:
    torch.manual_seed(37)
    memory_management.aimdo_enabled = False
    model = Probe(config, torch.bfloat16, "cpu", ops.manual_cast)
    memory_management.aimdo_enabled = True
    for name, p in model.named_parameters():
        p.copy_(torch.randn_like(p) * (0.05 if "norm" in name else 0.08))
    model.model.lm_head.weight[:8].mul_(4)
    patcher = model_patcher.ModelPatcherDynamic(model, device, torch.device("cpu"))
    mm.load_models_gpu([patcher])
    patcher.set_in_use_by_current_prompt(True)
    original_forward = model.model.forward
    outputs = []
    captured = set()
    states = []
    mode = "hard"

    class _ModeState(model_state.ModelState):
        # The hard mode loads without dynamic VRAM; the others toggle it back on for the model under test.
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def load(self, model, device):
            states.append(self)
            memory_management.aimdo_enabled = self.mode != "hard"
            loaded = super().load(model, device)
            memory_management.aimdo_enabled = True
            return loaded

    def forward(*args, **kwargs):
        for s in states:
            if mode in ("ram", "disk") and not outputs or mode == "evict" and len(outputs) == 4:
                s.patcher._vbar_get().set_watermark(0)
            if mode == "evict" and len(outputs) == 8:
                mm.DISABLE_PINNED_MEMORY = True
                s.patcher.partially_unload_ram(1e32)
        result = original_forward(*args, **kwargs)
        for s in states:
            if mode in ("ram", "disk") or mode == "evict" and len(outputs) >= 4:
                assert s.patcher.loaded_size() == 0
            if mode == "disk" or mode == "evict" and len(outputs) >= 8:
                assert s.patcher.loaded_ram_size() == 0
        outputs.append(result[0].cpu().clone())
        captured.update(i for i, layer in enumerate(model.model.layers) if hasattr(layer, "_comfy_graph"))
        return result

    model.model.forward = forward
    for attention, length in (("bf16", 19), ("int8", 19), ("int8", 1533)):
        mm.args.use_ck_attention = attention == "int8"
        embeds = torch.randn(1, length, 64, device=device, dtype=torch.bfloat16)
        continuation = torch.randn(1, 1, 64, device=device, dtype=torch.bfloat16)
        mm.args.disable_fast_disk = length <= 512
        storage.anonymous_fast_disk.cache_clear()
        if attention == "int8":
            assert is_available(device)
        for layer in [model.model.layers[3], model.mtp.layers[0]]:
            layer.self_attn.comfy_attention.method = "comfy_kitchen_int8" if attention == "int8" else "sdpa"
        for graphs, queue in ((False, True), (True, True)):
            mm.args.disable_cuda_graphs = not graphs
            model.model.prefetch_dynamic_vbars = queue
            for mtp in (False, "3"):
                baseline = None
                for mode in ("hard", "resident", "ram", "disk", "evict"):
                    mm.DISABLE_PINNED_MEMORY = mode == "disk"
                    states.clear()
                    outputs.clear()
                    captured.clear()
                    # The resident run is saved to a named file; the others stay anonymous.
                    path = os.path.join(tmp, f"{attention}-{length}-{graphs}-{mtp}.safetensors") if mode == "resident" else None
                    state = _ModeState(mode)
                    mm.register_ephemeral_model(state)
                    ids = model.generate(embeds.clone(), do_sample=True, seed=71, max_length=24, mtp=mtp, model_state=state)
                    # A snapshot does not unload its managed model.
                    dead = [weakref.ref(s.patcher) for s in states] + [weakref.ref(s.patcher.model) for s in states]
                    states.clear()
                    loaded_patcher = state.patcher
                    state.flush()
                    assert state.patcher is loaded_patcher
                    del loaded_patcher
                    saved = state.tensors
                    if mode == "hard":
                        # A plain model state restores its tensors into device memory.
                        reopened = state.load(model.init_kv_cache(1, state.metadata["capacity"], device, torch.bfloat16, mtp=bool(mtp)), device)
                        assert all(torch.equal(v.cpu(), saved[k]) for k, v in reopened.state_dict().items())
                        del reopened
                        state.flush()
                        state.release()
                    if path is not None:
                        state.save(path)
                        assert all(torch.equal(v, saved[k]) for k, v in safetensors.torch.load_file(path).items())
                    # An ephemeral state without an owner still releases its model at node end.
                    mm.reset_cast_buffers()
                    mm.cleanup_ephemeral_models()
                    assert not mm.EPHEMERAL_MODELS and all(ref() is None for ref in dead), (mode, dead)
                    assert any(cache.get("uncommitted") for cache in state.metadata["caches"])  # The last DeltaNet update has not been committed.
                    # The next turn copies the state, reopened by name or still in memory, onto a fresh cache.
                    source = state if path is None else model_state.ModelState.open(path, create="never")
                    assert source.metadata == json.loads(json.dumps(state.metadata))
                    turn = source.copy()
                    assert turn.metadata == source.metadata and turn.metadata is not source.metadata
                    assert all(torch.equal(v, saved[k]) for k, v in turn.tensors.items())
                    restored = model.init_kv_cache(1, turn.metadata["capacity"], device, torch.bfloat16, mtp=bool(mtp)).load(turn, device)
                    restored.restore_state(turn.metadata["caches"])
                    resumed = original_forward(None, embeds=continuation, attention_mask=None, past_key_values=restored.caches)[0].clone()
                    prefetch.cleanup_prefetch_queues()
                    turn.release()
                    del restored
                    if baseline is None:
                        baseline = ids, list(outputs), resumed
                    else:
                        assert ids == baseline[0], (attention, graphs, queue, mtp, mode, ids, baseline[0])
                        assert len(outputs) == len(baseline[1])
                        error = max((a.float() - b.float()).abs().max().item() for a, b in zip(outputs, baseline[1]))
                        assert error == 0, (attention, graphs, queue, mtp, mode, error)
                        assert torch.equal(resumed, baseline[2]), ("resume", attention, graphs, queue, mtp, mode)
                    if graphs and mode in ("hard", "resident", "evict"):
                        assert captured
                    if mode in ("ram", "disk"):
                        assert not captured
                    print("QWEN_WRITABLE_PASS", attention, length, graphs, queue, mtp, mode, len(ids), len(captured), flush=True)  # noqa: T201
                if attention == "bf16" and mtp == "3":
                    mode = "resident"
                    states.clear()
                    outputs.clear()
                    mm.DISABLE_PINNED_MEMORY = False
                    first = model_state.ModelState()
                    first.owner = patcher
                    model.generate(embeds.clone(), do_sample=True, seed=71, max_length=24, mtp=mtp, model_state=first)
                    managed = first.patcher
                    turn = first.copy()
                    reference = model_state.ModelState(dict(turn.tensors), json.loads(json.dumps(turn.metadata)))
                    prefetch.cleanup_prefetch_queues()
                    mm.reset_cast_buffers()
                    mm.cleanup_ephemeral_models()
                    assert turn.patcher is managed and patcher.get_attachment("model_state_model")[1] is managed
                    runs = []
                    for current in (turn, reference):
                        outputs.clear()
                        captured.clear()
                        if current is reference:
                            mm.register_ephemeral_model(current)
                        ids = model.generate(continuation.clone(), do_sample=True, seed=71, max_length=8, mtp=mtp, model_state=current)
                        if current is turn:
                            assert current.patcher is managed
                        if graphs:
                            assert captured
                        runs.append((ids, list(outputs)))
                        prefetch.cleanup_prefetch_queues()
                        mm.reset_cast_buffers()
                        mm.cleanup_ephemeral_models()
                    assert runs[0][0] == runs[1][0] and len(runs[0][1]) == len(runs[1][1])
                    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
                    turn.release()
                    mm.unload_model_and_clones(managed)
                    assert not mm.EPHEMERAL_MODELS
                    print("QWEN_PERSISTENT_PASS", graphs, flush=True)  # noqa: T201
    pending = torch.nn.Sequential(ops.WritableWeight((1,), torch.float32))
    pending.comfy_writable = True
    output = torch.empty(2 ** 20, device=device)
    state = model_state.ModelState()
    mm.register_ephemeral_model(state)
    state.load(pending, device)
    with prefetch.malloc_graph_scope(device):
        temporary = torch.full_like(output, 1.25)
        torch.cuda._sleep(50_000_000)
        output.copy_(temporary)
        del temporary
    del pending
    prefetch.cleanup_prefetch_queues()
    mm.reset_cast_buffers()
    mm.cleanup_ephemeral_models()
    assert torch.all(output.cpu() == 1.25), "Writable teardown freed pages before pending work finished"
    mm.unload_model_and_clones(patcher)
print("QWEN_WRITABLE_ALL_PASS", flush=True)  # noqa: T201
