import asyncio
from types import SimpleNamespace

import pytest

import torch

import nodes
from execution import record_lazy_evidence
from comfy_execution.caching import (
    BasicCache,
    CacheKeySetInputSignature,
    get_lazy_input_keys,
)
from comfy_execution.graph import DynamicPrompt


class StubKeyProducer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stub"
    CATEGORY = "testing"

    def stub(self, value):
        return (torch.zeros(1),)


class StubLazyConsumer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {"lazy": True}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mix"
    CATEGORY = "testing"

    def check_lazy_status(self, mask, image1):
        return [] if mask == 1.0 else ["image1"]

    def mix(self, mask, image1):
        return (image1,)


class StubEagerConsumer(StubLazyConsumer):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "mask": ("MASK",),
            },
        }


@pytest.fixture
def register_stubs(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StubKeyProducer", StubKeyProducer)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StubKeyLazy", StubLazyConsumer)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StubKeyEager", StubEagerConsumer)


class StubIsChanged:
    async def get(self, node_id):
        return 0


def build_prompt(consumer_type, producer_value):
    return {
        "1": {"class_type": "StubKeyProducer", "inputs": {"value": producer_value}, "_meta": {"title": "p"}},
        "2": {"class_type": consumer_type, "inputs": {"image1": ["1", 0], "mask": 1.0}, "_meta": {"title": "c"}},
    }


async def build_consumer_key(prompt, lazy_evaluated):
    dynprompt = DynamicPrompt(prompt)
    key_set = CacheKeySetInputSignature(
        dynprompt, list(prompt.keys()), StubIsChanged(), lazy_evaluated=lazy_evaluated
    )
    await key_set.add_keys(list(prompt.keys()))
    return key_set.get_data_key("2")


def test_no_evidence_keeps_old_including_behavior(register_stubs):
    assert get_lazy_input_keys("StubKeyLazy") == frozenset({"image1"})

    # Before any run recorded evidence, keys must match the old always-include behavior.
    key_before = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 0), {}))
    key_after = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 5), {}))

    assert key_before != key_after


def test_lazy_keys_exclude_unconsumed_ancestors(register_stubs):
    known_unevaluated = {("2", "image1"): False}

    key_before = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 0), known_unevaluated))
    key_after = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 5), known_unevaluated))

    assert key_before == key_after, "upstream churn behind an unevaluated lazy input must not change the key"


def test_consumed_lazy_input_still_invalidates(register_stubs):
    lazy_evaluated = {("2", "image1"): True}

    key_before = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 0), lazy_evaluated))
    key_after = asyncio.run(build_consumer_key(build_prompt("StubKeyLazy", 5), lazy_evaluated))

    assert key_before != key_after


def test_eager_link_still_invalidates(register_stubs):
    key_before = asyncio.run(build_consumer_key(build_prompt("StubKeyEager", 0), {}))
    key_after = asyncio.run(build_consumer_key(build_prompt("StubKeyEager", 5), {}))

    assert key_before != key_after


def test_subcache_shares_evidence_record(register_stubs):
    shared = {}

    async def build():
        bc = BasicCache(CacheKeySetInputSignature, key_class_kwargs={"lazy_evaluated": shared})
        dynprompt = DynamicPrompt(build_prompt("StubKeyLazy", 0))
        await bc.set_prompt(dynprompt, ["1", "2"], StubIsChanged())
        return await bc._ensure_subcache("1", ["2"])

    child = asyncio.run(build())
    assert child.cache_key_set.lazy_evaluated is shared


def test_record_lazy_evidence_marks_only_requested(register_stubs):
    caches = SimpleNamespace(lazy_evaluated={})

    record_lazy_evidence(caches, "2", "StubKeyLazy", {"image1"})
    assert caches.lazy_evaluated == {("2", "image1"): True}


def test_record_lazy_evidence_is_monotonic(register_stubs):
    caches = SimpleNamespace(lazy_evaluated={})

    record_lazy_evidence(caches, "2", "StubKeyLazy", {"image1"})
    # A later run sees the input cached and check_lazy_status stops naming it;
    # the recorded value must not flip or identical re-queues would miss cache.
    record_lazy_evidence(caches, "2", "StubKeyLazy", set())
    assert caches.lazy_evaluated == {("2", "image1"): True}

    record_lazy_evidence(caches, "3", "StubKeyLazy", set())
    assert caches.lazy_evaluated[("3", "image1")] is False


def test_record_lazy_evidence_none_request_is_noop(register_stubs):
    caches = SimpleNamespace(lazy_evaluated={})

    record_lazy_evidence(caches, "2", "StubKeyLazy", None)

    assert caches.lazy_evaluated == {}
