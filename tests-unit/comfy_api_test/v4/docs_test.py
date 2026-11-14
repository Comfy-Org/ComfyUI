import re
import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

import nodes
import server
from comfy_api import v4

DOCS = Path(v4.__file__).parent / "docs"

HELPERS = """\
import torch

from comfy.sd import VAE
from comfy_api.latest import ComfyExtension, io, ui


class TinyVae(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DocsTinyVae", outputs=[io.Vae.Output()])

    @classmethod
    def execute(cls):
        vae = VAE(sd={"pixel_space_vae": torch.tensor(1.0)}, device=torch.device("cpu"), dtype=torch.float32)
        return io.NodeOutput(vae)


class Shape(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="DocsShape", inputs=[io.AnyType.Input("value")], is_output_node=True)

    @classmethod
    def execute(cls, value):
        tensor = value["samples"] if isinstance(value, dict) else value
        return io.NodeOutput(ui=ui.PreviewText(f"{type(value).__name__} {list(tensor.shape)}"))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [TinyVae, Shape]


async def comfy_entrypoint():
    return Extension()
"""

# What each example node gives for a 1x8x16 RGB image and a pixel-space VAE.
SHAPES = {
    "ExampleVaeRoundTrip": "Tensor [1, 8, 16, 3]",
    "ExampleVaeRoundTripSync": "Tensor [1, 8, 16, 3]",
    "ExampleEncodeImage": "dict [1, 3, 8, 16]",
}

# What the script examples leave in these names.
RESULTS = {
    "smaller": lambda value: list(value.shape) == [1, 32, 32, 3],
    "supported": lambda value: value is True,
}


def _examples() -> list:
    return [
        pytest.param(code, id=f"{page.stem}:{number}")
        for page in sorted(DOCS.glob("*.md"))
        if page.name != "reference.md"
        for number, code in enumerate(
            re.findall(
                r"^```python\n(.*?)^```$", page.read_text(encoding="utf-8"), re.M | re.S
            ),
            1,
        )
    ]


def _prompt(node_id: str) -> dict:
    return {
        "image": {
            "class_type": "EmptyImage",
            "inputs": {"width": 16, "height": 8, "batch_size": 1, "color": 0},
        },
        "vae": {"class_type": "DocsTinyVae", "inputs": {}},
        "node": {
            "class_type": node_id,
            "inputs": {"image": ["image", 0], "vae": ["vae", 0]},
        },
        "shape": {"class_type": "DocsShape", "inputs": {"value": ["node", 0]}},
    }


def test_every_code_block_in_the_guides_is_python_or_shell():
    for page in DOCS.glob("*.md"):
        if page.name != "reference.md":
            text = page.read_text(encoding="utf-8")
            assert not re.search(r"^[ \t]+```", text, re.M), page.name
            assert set(re.findall(r"^```(.*)$", text, re.M)[::2]) <= {"python", "sh"}


def test_every_call_the_guides_name_exists():
    roots = {"eng": v4.ComfyAPI_v4(), "util": v4.util, "folders": v4.util.folders}
    named = {
        name
        for page in DOCS.glob("*.md")
        if page.name != "reference.md"
        for name in re.findall(
            r"\b(?:eng|util|folders)(?:\.\w+)+", page.read_text("utf-8")
        )
    }

    assert len(named) > 20
    for name in named:
        root, *attributes = name.split(".")
        value = roots[root]
        for attribute in attributes:
            assert hasattr(value, attribute), name
            value = getattr(value, attribute)


def test_every_result_the_test_checks_is_in_an_example():
    assert all(
        any(f"{name} = " in e.values[0] for e in _examples()) for name in RESULTS
    )


def test_every_node_the_examples_define_is_run_by_this_test():
    defined = {
        node_id
        for example in _examples()
        for node_id in re.findall(r'node_id="(\w+)"', example.values[0])
    }

    assert defined == set(SHAPES)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("code", _examples())
async def test_guide_example_runs(code, load_pack, run_prompt, monkeypatch, tmp_path):
    example = tmp_path / "example.py"
    example.write_text(code, encoding="utf-8")
    namespace = runpy.run_path(str(example))
    for name, expected in RESULTS.items():
        assert name not in namespace or expected(namespace[name]), name
    if "comfy_entrypoint" not in namespace:
        return
    extension = await namespace["comfy_entrypoint"]()
    routes = await extension.get_routes() if hasattr(extension, "get_routes") else []
    if routes:
        monkeypatch.setattr(server.PromptServer, "instance", None, raising=False)
        host = server.PromptServer(None, MagicMock(enabled=False))
    await load_pack(HELPERS, "docs_helpers")
    before = set(nodes.NODE_CLASS_MAPPINGS)
    await load_pack(code, "example_pack")

    for node_id in set(nodes.NODE_CLASS_MAPPINGS) - before:
        executor, sent = await run_prompt(_prompt(node_id), ["shape"])
        assert executor.success, executor.status_messages
        shown = executor.history_result["outputs"]["shape"]
        assert shown == {"text": [SHAPES[node_id]]}
        progress = [
            n
            for e, d, _ in sent.sent
            if e == "progress_state"
            for n in d["nodes"].values()
        ]
        assert all(n["value"] <= n["max"] for n in progress)

    if not routes:
        return
    guides = "".join(
        p.read_text("utf-8") for p in DOCS.glob("*.md") if p.name != "reference.md"
    )
    documented = set(re.findall(r"`(/[\w/]+)`", guides))
    host.add_routes()
    async with TestClient(TestServer(host.app)) as client:
        for route in routes:
            paths = {route["path"], "/api" + route["path"]}
            for path in paths | {p for p in documented if p.endswith(route["path"])}:
                response = await client.request(route["method"], path + "?name=docs")
                assert response.status == 200, path
                assert await response.json() == {"hello": "docs"}
            response = await client.request(route["method"], route["path"])
            assert await response.json() == {"hello": "world"}
