import asyncio

import pytest

import nodes
from comfy_execution.caching import CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt


class Producer:
    SKIP_UNLINKED_OUTPUTS = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}


class PlainProducer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING",)}}


class Consumer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class NotChanged:
    async def get(self, node_id):
        return False


@pytest.fixture(autouse=True)
def node_classes(monkeypatch):
    for cls in (Producer, PlainProducer, Consumer):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)


def prompt(producer, consumer_inputs):
    return {
        "1": {"class_type": producer, "inputs": {"text": "a"}},
        "2": {"class_type": "Consumer", "inputs": consumer_inputs},
    }


def keys(original):
    dynprompt = DynamicPrompt(original)
    key_set = CacheKeySetInputSignature(dynprompt, list(original), NotChanged())
    asyncio.run(key_set.add_keys(list(original)))
    return key_set


def test_get_linked_outputs():
    dynprompt = DynamicPrompt(prompt("Producer", {"a": ["1", 0], "b": ["1", 2], "c": 5}))
    assert dynprompt.get_linked_outputs("1") == {0, 2}
    assert dynprompt.get_linked_outputs("2") == set()


def test_linking_an_output_changes_the_producer_key():
    one = keys(prompt("Producer", {"a": ["1", 0]}))
    both = keys(prompt("Producer", {"a": ["1", 0], "b": ["1", 1]}))
    assert one.get_data_key("1") != both.get_data_key("1")


def test_producer_key_ignores_which_consumer_holds_the_link():
    one = keys(prompt("Producer", {"a": ["1", 0]}))
    other = keys(prompt("Producer", {"z": ["1", 0]}))
    assert one.get_data_key("1") == other.get_data_key("1")


def test_descendant_key_ignores_producer_links_it_does_not_read():
    original = prompt("Producer", {"a": ["1", 0]})
    linked = dict(original, **{"3": {"class_type": "Consumer", "inputs": {"b": ["1", 1]}}})
    assert keys(original).get_data_key("2") == keys(linked).get_data_key("2")


def test_nodes_without_the_flag_keep_their_key():
    one = keys(prompt("PlainProducer", {"a": ["1", 0]}))
    both = keys(prompt("PlainProducer", {"a": ["1", 0], "b": ["1", 1]}))
    assert one.get_data_key("1") == both.get_data_key("1")
