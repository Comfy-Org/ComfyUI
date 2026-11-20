"""A cast that fails after vbar_fault must not leave a pinned module that no cleanup can reach.

vbar_fault pins the module's pages when it returns a signature. cast_modules_with_vbar makes the
module's _prefetch record right after the fault, so the caller's cleanup (cleanup_prefetched_modules,
directly or through cleanup_prefetch_queues) sees the pin and releases it, even if the record is
only partly filled in when the cast fails.
"""
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.qwen_image21.model as qwen_image21  # noqa: E402
import comfy.memory_management  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.model_prefetch  # noqa: E402
import comfy.ops  # noqa: E402


class _Group:
    def __init__(self, linears):
        self.linears = linears
        self._v_weight = object()
        for m in linears:
            m._combined_linear = self


class _Module:
    # Hashable like the torch modules cast_modules_with_vbar keys its group faults by.
    def __init__(self, name, signature):
        self._v = name
        self._v_signature = signature
        self._combined_linear = None


def _module(name, signature="old"):
    return _Module(name, signature)


@pytest.fixture
def vbar(monkeypatch):
    state = SimpleNamespace(pinned=[], fault=None)

    def fault(v):
        if state.fault is not None:
            signature = state.fault(v)
        else:
            signature = "same" if v.startswith("resident") else "new"
        if signature is not None:
            state.pinned.append(v)
        return signature

    def unpin(v):
        state.pinned.remove(v)

    model_vbar = comfy.ops.comfy_aimdo.model_vbar
    monkeypatch.setattr(model_vbar, "vbar_fault", fault)
    monkeypatch.setattr(model_vbar, "vbar_unpin", unpin)
    monkeypatch.setattr(model_vbar, "vbar_signature_compare", lambda a, b: a is not None and a == b)
    return state


@pytest.fixture
def failing_cast(monkeypatch):
    # Stands in for any error after the fault, such as "VRAM grow failed".
    def fail(*args, **kwargs):
        raise RuntimeError("VRAM grow failed")
    monkeypatch.setattr(comfy.ops, "materialize_meta_param", fail)


def test_failed_cast_leaves_a_record_that_cleanup_releases(vbar, failing_cast):
    s = _module("weight")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, False)
    assert vbar.pinned == ["weight"]
    assert s._prefetch == {"signature": "new", "resident": False}

    comfy.model_prefetch.cleanup_prefetched_modules(None, [s])
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_cleanup_does_not_unpin_a_failed_fault(vbar, failing_cast):
    vbar.fault = lambda v: None
    s = _module("weight")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, False)
    comfy.model_prefetch.cleanup_prefetched_modules(None, [s])
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_batch_failure_cleanup_releases_every_faulted_module(vbar, failing_cast):
    resident = _module("resident", signature="same")
    missed = _module("missed")
    untouched = _module("untouched")
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([resident, missed, untouched], None, "cpu", None, True, return_faulted=True)
    assert vbar.pinned == ["resident", "missed"]
    assert not hasattr(untouched, "_prefetch")

    comfy.model_prefetch.cleanup_prefetched_modules(None, [resident, missed, untouched])
    assert vbar.pinned == []


def test_group_failure_cleanup_releases_peers_faulted_up_front(vbar, failing_cast):
    members = [_module("a"), _module("b"), _module("c")]
    _Group(members)
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar(members, None, "cpu", None, True, return_faulted=True)
    assert vbar.pinned == ["a", "b", "c"]
    # One member missed, so every member's record is a miss.
    assert all(m._prefetch["resident"] is False for m in members)

    comfy.model_prefetch.cleanup_prefetched_modules(None, members)
    assert vbar.pinned == []
    assert all(not hasattr(m, "_prefetch") for m in members)


def test_fault_error_mid_group_cleanup_releases_earlier_peers(vbar):
    members = [_module("a"), _module("b"), _module("c")]
    _Group(members)

    def fault(v):
        if v == "c":
            raise RuntimeError("Fault failed: 2")
        return "new"
    vbar.fault = fault
    with pytest.raises(RuntimeError, match="Fault failed"):
        comfy.ops.cast_modules_with_vbar(members, None, "cpu", None, True, return_faulted=True)
    comfy.model_prefetch.cleanup_prefetched_modules(None, members)
    assert vbar.pinned == []


def test_prefetch_queue_cleanup_releases_a_partial_record(vbar, failing_cast, monkeypatch):
    s = _module("weight")
    # prefetch_queue_pop stores the head this way before pinning, so a failed pin stays reachable.
    monkeypatch.setattr(comfy.model_prefetch, "PREFETCH_QUEUES", [[(None, (None, [s]))]])
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_modules_with_vbar([s], None, "cpu", None, True, return_faulted=True)
    comfy.model_prefetch.cleanup_prefetch_queues()
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_direct_cast_failure_releases_pin_and_record(vbar, failing_cast, monkeypatch):
    s = _module("weight")
    s.weight = torch.zeros(4)
    s.bias = None
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    with pytest.raises(RuntimeError, match="VRAM grow failed"):
        comfy.ops.cast_bias_weight(s, dtype=torch.float32, device=torch.device("meta"), bias_dtype=torch.float32, offloadable=True)
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_direct_cast_resolve_failure_releases_pin_and_record(vbar, monkeypatch):
    s = _module("weight")
    s.weight = torch.zeros(4)
    s.bias = None

    def cast(modules, *args, **kwargs):
        for m in modules:
            m._prefetch = {"signature": comfy.ops.comfy_aimdo.model_vbar.vbar_fault(m._v), "resident": False}

    def resolve(*args, **kwargs):
        raise RuntimeError("Allocation on device failed")
    monkeypatch.setattr(comfy.model_management, "is_device_cpu", lambda device: False)
    monkeypatch.setattr(comfy.model_management, "sync_stream", lambda device, stream: None)
    monkeypatch.setattr(comfy.ops, "cast_modules_with_vbar", cast)
    monkeypatch.setattr(comfy.ops, "resolve_cast_module_with_vbar", resolve)
    with pytest.raises(RuntimeError, match="Allocation on device failed"):
        comfy.ops.cast_bias_weight(s, dtype=torch.float32, device=torch.device("meta"), bias_dtype=torch.float32, offloadable=True)
    assert vbar.pinned == []
    assert not hasattr(s, "_prefetch")


def test_prefix_cache_layer_cleans_slots_when_pinning_fails(monkeypatch):
    slot = SimpleNamespace(comfy_cast_weights=True)

    def pin_modules(modules, device):
        for m in modules:
            m._prefetch = {"signature": "new", "resident": False}
        raise RuntimeError("Pin registration failed")
    cleaned = []
    monkeypatch.setattr(comfy.model_prefetch, "pin_modules", pin_modules)
    monkeypatch.setattr(comfy.model_prefetch, "cleanup_prefetched_modules", lambda module, modules: cleaned.extend(modules))

    cache = qwen_image21.PrefixCache([[slot]])
    with pytest.raises(RuntimeError, match="Pin registration failed"):
        with cache.layer(0, "cpu", write=False):
            pass
    assert cleaned == [slot]


def test_resolve_failure_does_not_record_the_new_signature():
    s = _module("weight", signature="old")
    s.weight = torch.ones(4)
    s.bias = None
    geometry = comfy.memory_management.tensors_to_geometries([s.weight, s.bias])
    xfer_dest = torch.zeros(comfy.memory_management.vram_aligned_size([s.weight, s.bias]), dtype=torch.uint8)

    def lowvram(x):
        raise RuntimeError("LoRA patch failed")
    s.weight_lowvram_function = lowvram
    s._prefetch = {"signature": "new", "resident": False, "xfer_dest": xfer_dest, "cast_dest": None,
                   "cast_geometry": geometry, "needs_cast": False}
    with pytest.raises(RuntimeError, match="LoRA patch failed"):
        comfy.ops.resolve_cast_module_with_vbar(s, torch.float32, torch.device("cpu"), torch.float32, None, False)
    # A retry that faults the same signature must not treat the half-patched weight as resident.
    assert s._v_signature is None
