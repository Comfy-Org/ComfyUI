from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy_api.latest import io
from comfy_execution import caching
from comfy_execution.graph import DynamicPrompt
from comfy_extras.nodes_images import GetImageSize


class LegacyUniqueId:
    @classmethod
    def INPUT_TYPES(cls):
        return {"hidden": {"unique_id": "UNIQUE_ID"}}


class SharedV3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__)

    @classmethod
    def execute(cls):
        return io.NodeOutput()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_class, distinct_keys",
    [(GetImageSize, True), (LegacyUniqueId, True), (SharedV3, False)],
)
async def test_cache_keys_respect_hidden_unique_id(monkeypatch, node_class, distinct_keys):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "CacheTestNode", node_class)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})
    prompt = DynamicPrompt({
        node_id: {"class_type": "CacheTestNode", "inputs": {}}
        for node_id in ("first", "second")
    })
    changes = SimpleNamespace(get=AsyncMock(return_value=False))
    keys = caching.CacheKeySetInputSignature(prompt, ["first", "second"], changes)
    await keys.add_keys(["first", "second"])

    assert (keys.get_data_key("first") != keys.get_data_key("second")) == distinct_keys
