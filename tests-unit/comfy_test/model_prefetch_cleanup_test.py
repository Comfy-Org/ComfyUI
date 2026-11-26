from contextlib import nullcontext
import threading

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_prefetch as model_prefetch  # noqa: E402


class MTPHead(torch.nn.Module):
    pass


@pytest.fixture(autouse=True)
def _reset_queues():
    model_prefetch.PREFETCH_QUEUES = []
    model_prefetch.GRAPH_WARMED_MODULES.clear()
    yield
    model_prefetch.PREFETCH_QUEUES = []
    model_prefetch.GRAPH_WARMED_MODULES.clear()


def _prefetched(module):
    module._prefetch = {"signature": None}
    return (None, (module, [module]))


@pytest.mark.parametrize("pending_unit", [tuple, list])
def test_cleanup_skips_unconsumed_grouped_unit(pending_unit):
    # Qwen3.5 MTP draft queue interrupted after the embed pop: the embed slot has
    # been pinned, the grouped (head, mtp) slot has not been reached yet.
    embed = torch.nn.Embedding(4, 4)
    head = torch.nn.Linear(4, 4)
    queue = [_prefetched(embed), pending_unit((head, MTPHead())), None]
    model_prefetch.PREFETCH_QUEUES = [queue]

    model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []
    assert not hasattr(embed, "_prefetch")


def test_cleanup_resets_queues_when_entry_cleanup_fails(monkeypatch):
    def boom(module, comfy_modules):
        raise RuntimeError("unpin failed")

    monkeypatch.setattr(model_prefetch, "cleanup_prefetched_modules", boom)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(torch.nn.Linear(4, 4)), None]]

    with pytest.raises(RuntimeError):
        model_prefetch.cleanup_prefetch_queues()
    # A failed cleanup must not poison every later node on the worker.
    assert model_prefetch.PREFETCH_QUEUES == []
    model_prefetch.cleanup_prefetch_queues()


@pytest.mark.parametrize("pending_unit", [tuple, list])
def test_cleanup_skips_unreached_grouped_head(pending_unit):
    # Interrupted between popping the consumed slot and pinning the next one: the
    # head of the queue is still the raw (head, mtp) group, not a pinned slot.
    queue = [pending_unit((torch.nn.Linear(4, 4), MTPHead())), None]
    model_prefetch.PREFETCH_QUEUES = [queue]

    model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []


def test_cleanup_unpins_later_queues_when_an_earlier_one_fails(monkeypatch):
    real_cleanup = model_prefetch.cleanup_prefetched_modules
    failing = torch.nn.Linear(4, 4)
    later = torch.nn.Linear(4, 4)

    def cleanup(module, comfy_modules):
        if module is failing:
            raise RuntimeError("unpin failed")
        real_cleanup(module, comfy_modules)

    monkeypatch.setattr(model_prefetch, "cleanup_prefetched_modules", cleanup)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(failing), None], [_prefetched(later), None]]

    with pytest.raises(RuntimeError, match="unpin failed"):
        model_prefetch.cleanup_prefetch_queues()
    # The first failure is re-raised, but only after every other queue was cleaned up.
    assert not hasattr(later, "_prefetch")
    assert model_prefetch.PREFETCH_QUEUES == []


def test_failed_unpin_still_clears_prefetch_markers(monkeypatch):
    # A module whose unpin raised must not keep _prefetch, or a later cast would skip
    # the fresh cast and reuse the stale transfer state. The rest of the group is still cleaned.
    calls = []

    def vbar_unpin(v):
        calls.append(v)
        if v == "first":
            raise RuntimeError("unpin failed")

    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_unpin", vbar_unpin)
    first = torch.nn.Linear(4, 4)
    second = torch.nn.Linear(4, 4)
    for module, v in ((first, "first"), (second, "second")):
        module._v = v
        module._prefetch = {"signature": "sig"}
    model_prefetch.PREFETCH_QUEUES = [[(None, (first, [first, second])), None]]

    with pytest.raises(RuntimeError, match="unpin failed"):
        model_prefetch.cleanup_prefetch_queues()
    assert calls == ["first", "second"]
    assert not hasattr(first, "_prefetch")
    assert not hasattr(second, "_prefetch")


def test_malloc_graph_cleanup_failure_still_cleans_queues(monkeypatch):
    # After a GPU failure the graph abort can raise too. Queues and _prefetch markers must not
    # survive into the next node, and the graph error still reaches the caller.
    def boom():
        raise RuntimeError("graph abort failed")

    monkeypatch.setattr(model_prefetch, "cleanup_malloc_graph", boom)
    module = torch.nn.Linear(4, 4)
    model_prefetch.PREFETCH_QUEUES = [[_prefetched(module), None]]

    with pytest.raises(RuntimeError, match="graph abort failed"):
        model_prefetch.cleanup_prefetch_queues()

    assert model_prefetch.PREFETCH_QUEUES == []
    assert not hasattr(module, "_prefetch")


def test_cleanup_releases_block_pin_without_module_records(monkeypatch):
    module = torch.nn.Linear(2, 2)
    module._v_block = "block"
    module._v_block_faulted = True
    calls = []
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_unpin", calls.append)
    model_prefetch.PREFETCH_QUEUES = [[(None, (module, []))]]

    model_prefetch.cleanup_prefetch_queues()

    assert calls == ["block"]
    assert not hasattr(module, "_v_block_faulted")
    assert model_prefetch.PREFETCH_QUEUES == []


class _DummyGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _DummyStream:
    def wait_stream(self, _other):
        pass


class _DummyMallocGraph:
    def __init__(self, fail_iterate=False):
        self._comfy_active = True
        self._comfy_cuda_graph_modules = set()
        self.rogue_count = 0
        self.fail_iterate = fail_iterate

    def iterate(self, scope):
        if self.fail_iterate:
            raise RuntimeError("graph break")
        return False

    def use_stream(self, _stream):
        return nullcontext()

    def abort(self):
        pass


def _setup_graph_env(monkeypatch, device, *, signature_match=True, fail_iterate=False):
    pins = {"count": 0, "max": 0}

    def vbar_fault(v):
        pins["count"] += 1
        pins["max"] = max(pins["max"], pins["count"])
        return "sig"

    def vbar_unpin(v):
        pins["count"] -= 1

    stream = _DummyStream()
    monkeypatch.setattr(model_prefetch.args, "disable_cuda_graphs", False)
    monkeypatch.setattr(model_prefetch.comfy.model_management, "is_device_cuda", lambda _d: True)
    monkeypatch.setattr(model_prefetch.comfy.model_management, "current_stream", lambda _d: stream)
    monkeypatch.setattr(model_prefetch.torch.cuda, "stream", lambda _s: nullcontext())
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_fault", vbar_fault)
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_unpin", vbar_unpin)
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda _a, _b: signature_match)
    monkeypatch.setitem(model_prefetch.MALLOC_GRAPHS, threading.get_ident(), _DummyMallocGraph(fail_iterate=fail_iterate))
    monkeypatch.setitem(model_prefetch.GRAPH_CAPTURE_STREAMS, device, stream)
    return pins


def test_repeated_module_graph_hit_unpins_vbar_block(monkeypatch):
    # A module repeated consecutively in one prefetch queue (e.g. RAFT's update_block)
    # must unpin the prior iteration's _v_block before faulting it for the next one.
    device = torch.device("cpu")
    pins = _setup_graph_env(monkeypatch, device, signature_match=True)
    graph = _DummyGraph()
    module = torch.nn.Linear(4, 4)
    module._v_block = "block"
    module._comfy_graph = {"graph": graph, "signature": "sig"}

    for _ in range(2):
        queue = [None, module, module, module, None]
        model_prefetch.PREFETCH_QUEUES.append(queue)
        for _ in range(3):
            model_prefetch.prefetch_queue_pop(queue, device, module, enable_graph=True, malloc_scope="block")
        model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
        assert pins["count"] == 0
        assert not hasattr(module, "_v_block_faulted")

    assert graph.replays == 6
    assert pins["max"] == 1


def test_graph_miss_after_vbar_fault_still_prefetches_and_unpins(monkeypatch):
    device = torch.device("cpu")
    pins = _setup_graph_env(monkeypatch, device, signature_match=False)
    graph = _DummyGraph()
    module = torch.nn.Linear(4, 4)
    module._v = "weight"
    module._v_block = "block"
    module._comfy_graph = {"graph": graph, "signature": "old"}

    pinned_submodules = []

    def fake_pin_modules(comfy_modules, _device, _dtype):
        pinned_submodules.extend(comfy_modules)
        for s in comfy_modules:
            s._prefetch = {"signature": "sig"}
            pins["count"] += 1
        return None, False

    monkeypatch.setattr(model_prefetch, "pin_modules", fake_pin_modules)

    ran_core = []
    queue = [None, module, None]
    model_prefetch.PREFETCH_QUEUES.append(queue)
    model_prefetch.prefetch_queue_pop(queue, device, module, core=lambda: ran_core.append(True), enable_graph=True, malloc_scope="block")
    model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")

    assert graph.replays == 0
    assert ran_core == [True]
    assert pinned_submodules == [module]
    assert pins["count"] == 0
    assert not hasattr(module, "_v_block_faulted")
    assert not hasattr(module, "_prefetch")


def test_vbar_block_unpinned_when_malloc_graph_iterate_raises(monkeypatch):
    device = torch.device("cpu")
    pins = _setup_graph_env(monkeypatch, device, signature_match=True, fail_iterate=True)
    module = torch.nn.Linear(4, 4)
    module._v_block = "block"
    module._comfy_graph = {"graph": _DummyGraph(), "signature": "sig"}

    queue = [None, module, None]
    model_prefetch.PREFETCH_QUEUES.append(queue)
    with pytest.raises(RuntimeError, match="graph break"):
        model_prefetch.prefetch_queue_pop(queue, device, module, enable_graph=True, malloc_scope="block")

    model_prefetch.cleanup_prefetch_queues()
    assert pins["count"] == 0
    assert not hasattr(module, "_v_block_faulted")
