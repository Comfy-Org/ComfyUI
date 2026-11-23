import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy_extras.nodes_images as nodes_images
import comfy_execution.caching as caching
import nodes
from comfy_api.latest import io
from execution import PromptExecutor


class Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self):
        return (torch.zeros(1, 8, 16, 3),)


class Widths:
    values = []

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first": ("INT",), "second": ("INT",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(self, first, second):
        self.values.append((first, second))
        return ()


class Server:
    client_id = None

    def send_sync(self, *args, **kwargs):
        pass


class ProgressText:
    def __init__(self):
        self.node_ids = []

    def send_progress_text(self, text, node_id):
        self.node_ids.append(node_id)


def v1_node(hidden):
    return type("V1Node", (), {"INPUT_TYPES": classmethod(lambda cls: {"required": {}, "hidden": hidden})})


class V3Output(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="TestV3Output", inputs=[io.Int.Input("value")], is_output_node=True)

    @classmethod
    def execute(cls, value):
        return io.NodeOutput()


class V3ApiNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="TestV3ApiNode", inputs=[io.Int.Input("value")], hidden=[io.Hidden.unique_id], is_api_node=True)

    @classmethod
    def execute(cls, value):
        return io.NodeOutput()


@pytest.mark.parametrize(("node_class", "expected"), [
    (nodes_images.GetImageSize, True),
    (V3ApiNode, False),
    (v1_node({"unique_id": "UNIQUE_ID"}), True),
    (type("V1ApiNode", (v1_node({"unique_id": "UNIQUE_ID"}),), {"API_NODE": True}), True),
    (v1_node({"unique_id": ("UNIQUE_ID",)}), False),
    (v1_node({"prompt": "PROMPT"}), False),
    (V3Output, False),
])
def test_only_nodes_given_their_unique_id_key_on_it(monkeypatch, node_class, expected):
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TestNode", node_class)

    assert caching.include_unique_id_in_input("TestNode") == expected


def test_identical_get_image_size_nodes_each_run_and_report_their_own_size(monkeypatch):
    progress = ProgressText()
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})
    monkeypatch.setattr(nodes_images, "PromptServer", type("PromptServer", (), {"instance": progress}))
    for name, node_class in {"GetImageSize": nodes_images.GetImageSize, "TestImage": Image, "TestWidths": Widths}.items():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, node_class)
    Widths.values = []
    prompt = {
        "image": {"class_type": "TestImage", "inputs": {}},
        "first": {"class_type": "GetImageSize", "inputs": {"image": ["image", 0]}},
        "second": {"class_type": "GetImageSize", "inputs": {"image": ["image", 0]}},
        "widths": {"class_type": "TestWidths", "inputs": {"first": ["first", 0], "second": ["second", 0]}},
    }
    executor = PromptExecutor(Server(), cache_type=False, cache_args={"ram": 0, "ram_inactive": 0})

    executor.execute(prompt, "two-image-sizes", execute_outputs=["widths"])

    assert executor.success
    assert Widths.values == [(16, 16)]
    assert sorted(progress.node_ids) == ["first", "second"]
