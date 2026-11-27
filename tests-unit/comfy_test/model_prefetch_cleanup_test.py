from unittest.mock import Mock, call

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_prefetch as model_prefetch  # noqa: E402


def test_prefetch_scope_without_graph(monkeypatch):
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPHS", {})
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPH_BREAKS", 0)

    model_prefetch.prefetch_queue_pop(
        None, torch.device("cpu"), torch.nn.Module(), malloc_scope="block_42", malloc_close_scope=True,
    )
    model_prefetch.prefetch_queue_pop(None, torch.device("cpu"), None, malloc_scope="block")

    assert model_prefetch.MALLOC_GRAPH_BREAKS == 0


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("close_scope", [False, True])
@pytest.mark.parametrize("broken", [False, True])
def test_prefetch_closes_scope_before_opening_next(monkeypatch, queued, active, close_scope, broken):
    graph = Mock(_comfy_active=active)
    graph.iterate.return_value = broken
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPHS", {model_prefetch.threading.get_ident(): graph})
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPH_BREAKS", 0)
    monkeypatch.setattr(model_prefetch, "pin_modules", lambda *args: (None, False))
    module = torch.nn.Module()
    queue = [None, module, None] if queued else None
    core = Mock()

    model_prefetch.prefetch_queue_pop(
        queue, torch.device("cpu"), module, core=core,
        malloc_scope="block_42", malloc_close_scope=close_scope,
    )

    expected = ([call(None)] if close_scope else []) + [call("block_42")] if active else []
    assert graph.iterate.call_args_list == expected
    assert model_prefetch.MALLOC_GRAPH_BREAKS == len(expected) * int(broken)
    core.assert_called_once_with()
    if queued:
        assert len(queue) == 2


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("close_scope", [False, True])
def test_prefetch_drain_closes_only_one_scope(monkeypatch, queued, close_scope):
    graph = Mock(_comfy_active=True)
    graph.iterate.return_value = False
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPHS", {model_prefetch.threading.get_ident(): graph})
    queue = [None, None] if queued else None

    model_prefetch.prefetch_queue_pop(
        queue, torch.device("cpu"), None, malloc_scope="block", malloc_close_scope=close_scope,
    )

    graph.iterate.assert_called_once_with(None)


@pytest.mark.parametrize("close_scope", [False, True])
def test_prefetch_replay_closes_only_one_scope(monkeypatch, close_scope):
    graph = Mock(_comfy_active=True)
    graph.iterate.return_value = False
    monkeypatch.setattr(model_prefetch, "MALLOC_GRAPHS", {model_prefetch.threading.get_ident(): graph})
    monkeypatch.setattr(model_prefetch.args, "disable_cuda_graphs", False)
    module = torch.nn.Module()
    module._v_block = object()
    replay = Mock()
    module._comfy_graph = {"graph": replay, "signature": object()}
    monkeypatch.setattr(model_prefetch, "GRAPH_CAPTURE_STREAMS", {torch.device("cuda:0"): object()})
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_fault", lambda block: object())
    monkeypatch.setattr(model_prefetch.comfy_aimdo.model_vbar, "vbar_signature_compare", lambda *args: True)
    core = Mock()

    model_prefetch.prefetch_queue_pop(
        [None, module, None], torch.device("cuda:0"), module, core=core, enable_graph=True,
        malloc_scope="block_42", malloc_close_scope=close_scope,
    )

    graph.iterate.assert_called_once_with(None)
    replay.replay.assert_called_once_with()
    core.assert_not_called()


class MTPHead(torch.nn.Module):
    pass


@pytest.fixture(autouse=True)
def _reset_queues():
    model_prefetch.PREFETCH_QUEUES = []
    yield
    model_prefetch.PREFETCH_QUEUES = []


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
