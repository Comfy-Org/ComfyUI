import pytest

import nodes
from comfy_api.latest import io
from comfy_execution import caching
from comfy_execution.graph import DynamicPrompt


class V1WithUniqueId:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"x": ("INT",)}, "hidden": {"unique_id": "UNIQUE_ID"}}


class V3WithUniqueId(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3WithUniqueId", inputs=[io.Int.Input("x")], outputs=[io.Int.Output()],
                         hidden=[io.Hidden.unique_id])

    @classmethod
    def execute(cls, x):
        return io.NodeOutput(x)


class V3WithoutUniqueId(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V3WithoutUniqueId", inputs=[io.Int.Input("x")], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, x):
        return io.NodeOutput(x)


class _NotChanged:
    async def get(self, node_id):
        return None


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    for node in (V1WithUniqueId, V3WithUniqueId, V3WithoutUniqueId):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node.__name__, node)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})


async def _keys_of_two_identical_nodes(class_type):
    prompt = DynamicPrompt({node_id: {"class_type": class_type, "inputs": {"x": 1}} for node_id in ("1", "2")})
    keys = caching.CacheKeySetInputSignature(prompt, ["1", "2"], _NotChanged())
    await keys.add_keys(["1", "2"])
    return keys.get_data_key("1"), keys.get_data_key("2")


@pytest.mark.asyncio
@pytest.mark.parametrize("class_type", ["V1WithUniqueId", "V3WithUniqueId"])
async def test_a_node_that_reads_its_unique_id_is_not_served_another_nodes_cached_result(class_type):
    first, second = await _keys_of_two_identical_nodes(class_type)
    assert first != second


@pytest.mark.asyncio
async def test_identical_nodes_that_do_not_read_their_unique_id_share_a_cache_key():
    first, second = await _keys_of_two_identical_nodes("V3WithoutUniqueId")
    assert first == second
