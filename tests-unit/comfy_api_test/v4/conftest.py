import sys
from textwrap import indent

import pytest
import torch

from comfy.cli_args import args
from comfy.model_base import BaseModel, ModelType
from comfy.model_patcher import ModelPatcher
from comfy.supported_models import SD15

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy.model_management import interrupt_current_processing
from comfy_api.v4 import _packs
from app.node_replace_manager import NodeReplaceManager
from execution import PromptExecutor, validate_prompt
from server import PromptServer

PACK = """\
import asyncio

import comfy.model_management
import torch
from comfy_api.v4 import (
    ComfyAPISync,
    ComfyExtension,
    InterruptProcessingException,
    eng,
    io,
)

api = {api}

class Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V4Node", is_output_node=True)

    @classmethod
    {definition} execute(cls):
{body}
        return io.NodeOutput()

class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Node]

async def comfy_entrypoint():
    return Extension()
"""


class RecordingServer:
    client_id = None
    last_node_id = None
    send_progress_text = PromptServer.send_progress_text

    def __init__(self):
        self.sockets_metadata = {
            "tab": {"feature_flags": {"supports_preview_metadata": True}}
        }
        self.sent = []
        self.node_replace_manager = NodeReplaceManager()

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data, sid))


@pytest.fixture
def tiny_sd15():
    """Build a seeded SD1.5-shaped model on the CPU; keyword arguments override its UNet config."""

    def build(**unet):
        config = SD15(
            {
                "image_size": 32,
                "in_channels": 4,
                "model_channels": 32,
                "out_channels": 4,
                "num_res_blocks": [1],
                "transformer_depth": [0],
                "transformer_depth_output": [0, 0],
                "transformer_depth_middle": -1,
                "channel_mult": [1],
                "use_spatial_transformer": False,
                "dtype": torch.float32,
                **unet,
            }
        )
        native = BaseModel(config, model_type=ModelType.EPS, device=torch.device("cpu"))
        generator = torch.Generator().manual_seed(321)
        with torch.no_grad():
            for parameter in native.parameters():
                parameter.normal_(0, 0.02, generator=generator)
        return ModelPatcher(native, torch.device("cpu"), torch.device("cpu"))

    return build


@pytest.fixture(autouse=True)
def no_cancel_left_pending():
    yield
    interrupt_current_processing(False)


@pytest.fixture(autouse=True)
def no_pack_left_registered(monkeypatch):
    monkeypatch.setattr(_packs, "_packs", {})
    monkeypatch.setattr(_packs, "_serving", False)


@pytest.fixture
def load_pack(tmp_path, monkeypatch):
    """Load `source` as the pack `name`; ComfyUI's node tables are restored after the test."""
    for attribute in (
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "LOADED_MODULE_DIRS",
        "EXTENSION_WEB_DIRS",
    ):
        monkeypatch.setattr(nodes, attribute, dict(getattr(nodes, attribute)))

    async def load(source: str, name: str = "v4_node_pack"):
        module = tmp_path / f"{name}.py"
        module.write_text(source, encoding="utf-8")
        monkeypatch.setitem(sys.modules, str(module.with_suffix("")), None)
        assert await nodes.load_custom_node(str(module))

    return load


@pytest.fixture
def run_prompt(monkeypatch):
    """Run `prompt` through the real executor, queued by `client_id`, sends recorded."""

    async def run(prompt: dict, outputs: list[str], *, client_id: str | None = "tab"):
        valid, error, execute_outputs, node_errors = await validate_prompt(
            "v4-node", prompt, outputs
        )
        assert valid, (error, node_errors)
        server = RecordingServer()
        monkeypatch.setattr(PromptServer, "instance", server, raising=False)
        executor = PromptExecutor(server, cache_args={"ram": 0, "ram_inactive": 0})
        await executor.execute_async(
            prompt, "v4-node", {"client_id": client_id}, execute_outputs=execute_outputs
        )
        return executor, server

    return run


@pytest.fixture
def run_node(load_pack, run_prompt):
    """Run a prompt of one V4 node, "v4", queued by client "tab". Once per test.

    `body` is the node's execute body. `{call}` in it is the API, awaited or sync.
    """

    async def run(
        body: str, synchronous: bool = False, *, client_id: str | None = "tab"
    ):
        await load_pack(
            PACK.format(
                api="ComfyAPISync()" if synchronous else "eng",
                definition="def" if synchronous else "async def",
                body=indent(
                    body.format(call="api" if synchronous else "await api"), " " * 8
                ),
            )
        )
        return await run_prompt(
            {"v4": {"class_type": "V4Node", "inputs": {}}}, ["v4"], client_id=client_id
        )

    return run
