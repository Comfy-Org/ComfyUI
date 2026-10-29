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
