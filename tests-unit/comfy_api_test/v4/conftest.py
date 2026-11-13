import asyncio
import sys
import time
from textwrap import indent
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import torch

from comfy.cli_args import args
from comfy.model_base import BaseModel, ModelType
from comfy.model_patcher import ModelPatcher
from comfy.supported_models import SD15

if not torch.cuda.is_available():
    args.cpu = True

import nodes
from comfy.model_management import interrupt_current_processing
from comfy_api.v4 import _pack_host, _pack_processes, _packs
from comfy_execution.utils import CurrentNodeContext
from app.node_replace_manager import NodeReplaceManager
from execution import PromptExecutor, PromptQueue, validate_prompt
from server import PromptServer


def pytest_addoption(parser):
    parser.addoption(
        "--pack-launcher",
        metavar="MODULE:FUNCTION",
        help="Start the tests' pack processes with this launcher, as ComfyUI's option does.",
    )


def pytest_configure(config):
    args.pack_launcher = config.getoption("--pack-launcher")


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

ISOLATED_PACK = """\
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


class Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V4Node", inputs=[io.String.Input("body")], is_output_node=True)

    @classmethod
    async def execute(cls, body):
        namespace = dict(globals(), api=eng)
        exec(body, namespace)
        await namespace["run"](cls)
        return io.NodeOutput()


class SyncNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="V4NodeSync", inputs=[io.String.Input("body")], is_output_node=True)

    @classmethod
    def execute(cls, body):
        namespace = dict(globals(), api=ComfyAPISync())
        exec(body, namespace)
        namespace["run"](cls)
        return io.NodeOutput()


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Node, SyncNode]


async def comfy_entrypoint():
    return Extension()
"""

# A pack's schema helper, and its node that answers with the pid of the process it runs in.
WHERE = """
def schema(node_id, inputs=(), outputs=(), **options):
    return io.Schema(node_id=node_id, inputs=list(inputs), outputs=list(outputs), **options)


class Where(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Where", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(os.getpid())
"""

needs_pack_process = pytest.mark.skipif(
    sys.platform == "win32",
    reason="--pack-process is not supported on Windows in this release",
)

# A node body that has ComfyUI cancel the prompt, as the user does who presses cancel
# when the node says it runs (see `RecordingServer.send_sync`).
CANCEL = '{call}.ui.send("cancel", {{}})\n'


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
        self.prompt_queue = PromptQueue(self)

    def queue_updated(self):
        pass

    def send_sync(self, event, data, sid=None):
        if event == "cancel":
            nodes.interrupt_processing()
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


def wait_until(condition, what: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def no_cancel_left_pending():
    yield
    interrupt_current_processing(False)


@pytest.fixture(autouse=True)
def no_pack_left_registered(monkeypatch):
    monkeypatch.setattr(_packs, "_packs", {})
    monkeypatch.setattr(_packs, "_serving", False)
    monkeypatch.setattr(_packs, "_retired", [])


@pytest.fixture
def node_tables(monkeypatch):
    """ComfyUI's node and folder tables, restored after the test."""
    for attribute in (
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
        "LOADED_MODULE_DIRS",
        "EXTENSION_WEB_DIRS",
    ):
        monkeypatch.setattr(nodes, attribute, dict(getattr(nodes, attribute)))


@pytest_asyncio.fixture
async def host(node_tables, monkeypatch):
    """ComfyUI's server on the test's loop, with a prompt queue and no frontend."""
    monkeypatch.setattr(PromptServer, "instance", None, raising=False)
    prompt_server = PromptServer(asyncio.get_running_loop(), MagicMock(enabled=False))
    prompt_server.prompt_queue = PromptQueue(prompt_server)
    return prompt_server


@pytest.fixture
def load_pack(tmp_path, monkeypatch, node_tables):
    """Load `source` as the pack `name`; ComfyUI's node tables are restored after the test."""

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


@pytest.fixture(scope="module")
def pack_process_nodes(tmp_path_factory):
    """One pack process for the test module; its nodes run the function their input defines."""
    path = tmp_path_factory.mktemp("pack_process") / "v4_node_pack.py"
    path.write_text(ISOLATED_PACK, encoding="utf-8")
    module = asyncio.run(_pack_host.load(str(path), "v4_node_pack"))
    extension = module.comfy_entrypoint()
    node_classes = {
        node_cls.GET_SCHEMA().node_id: node_cls for node_cls in extension.nodes
    }
    for node_cls in node_classes.values():
        node_cls.RELATIVE_PYTHON_MODULE = "custom_nodes.v4_node_pack"
    yield node_classes
    _pack_processes.unload(str(path))


@pytest.fixture(
    params=["in_process", pytest.param("pack_process", marks=needs_pack_process)]
)
def run_node(request, load_pack, run_prompt, monkeypatch):
    """Run a prompt of one V4 node, "v4", queued by client "tab". Once per test.

    `body` is the node's execute body. `{call}` in it is the API, awaited or sync.
    The pack runs in ComfyUI's process, or in its own (one per test module).
    """
    if request.param == "pack_process":
        node_classes = request.getfixturevalue("pack_process_nodes")

    async def run(
        body: str, synchronous: bool = False, *, client_id: str | None = "tab"
    ):
        body = body.format(call="api" if synchronous else "await api")
        if request.param == "in_process":
            await load_pack(
                PACK.format(
                    api="ComfyAPISync()" if synchronous else "eng",
                    definition="def" if synchronous else "async def",
                    body=indent(body, " " * 8),
                )
            )
            node, inputs = "V4Node", {}
        else:
            node = "V4NodeSync" if synchronous else "V4Node"
            monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, node, node_classes[node])
            definition = "def" if synchronous else "async def"
            inputs = {"body": f"{definition} run(cls):\n{indent(body, ' ' * 4)}\n"}
        return await run_prompt(
            {"v4": {"class_type": node, "inputs": inputs}}, ["v4"], client_id=client_id
        )

    return run


PROBE = """\
import asyncio
import collections
import dataclasses
import datetime
import decimal
import enum
import gc
import hashlib
import os
import pathlib
import subprocess
import sys
import threading
import time
import weakref

import numpy
import torch
from comfy_api.v4 import ComfyAPISync, ComfyExtension, InterruptProcessingException, eng, io
from comfy_api.v4.util import folders

OUTPUT_AT_IMPORT = folders.get_output_directory()


class SeedError(Exception):
    pass


class Color(enum.Enum):
    RED = "red"


class Shade(str, enum.Enum):
    DARK = "dark"


class Level(enum.IntEnum):
    HIGH = 2


class PosixPath(pathlib.PurePosixPath):
    pass


class Name(str):
    pass


@dataclasses.dataclass
class Box:
    value: int


Pair = collections.namedtuple("Pair", "left right")


def where():
    return os.getpid(), threading.current_thread().name


def outer(host_step):
    return [where()] + host_step(inner)


def inner():
    return [where()]


def fail(error):
    raise error


def caught(function):
    try:
        function()
    except BaseException as error:
        return type(error).__module__, type(error).__qualname__, str(error)


def in_a_thread(function):
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(caught(function)))
    thread.start()
    thread.join()
    return outcome[0]


def modes():
    return torch.is_inference_mode_enabled(), torch.is_grad_enabled()


errors = []


def keep_error(function):
    try:
        function()
    except BaseException as error:
        errors.append((type(error).__qualname__, isinstance(error, Exception)))


class Thing:
    pass


made = weakref.WeakSet()
watched = weakref.WeakValueDictionary()
# What the pack started, as a pack starts ffmpeg.
programs = []


def start_program(code="import time; time.sleep(600)", own_session=False):
    programs.append(
        subprocess.Popen([sys.executable, "-c", code], start_new_session=own_session)
    )
    return programs[-1].pid


def watch(name, obj):
    watched[name] = obj
    return obj


def make():
    thing = Thing()
    made.add(thing)
    return thing


shared = {}
kept = {}


def keep(name, value):
    kept[name] = value
    return value


def digest(tensor):
    return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()


def hold_until_asked(value):
    shared["held"] = value
    while "asked" not in shared:
        time.sleep(0.01)


def ask_held():
    while "held" not in shared:
        time.sleep(0.01)
    try:
        return caught(shared["held"])
    finally:
        shared["asked"] = True


def make_slowly():
    time.sleep(0.3)
    return make()


def alive():
    gc.collect()
    return len(made)


def seen(value):
    seen = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, torch.Tensor):
        return seen, str(value.dtype), tuple(value.shape), value.device.type
    return seen


class Eval(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Eval", outputs=[io.AnyType.Output()])

    @classmethod
    def execute(cls, expression, value=None):
        return io.NodeOutput(eval(expression, globals(), {"value": value}))


class CleansUp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CleansUp", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls, seconds=60, cleanup=0.5, calls=False, answers=False):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            await asyncio.sleep(cleanup)
            if calls:
                await eng.execution.is_interrupted()
            if answers:
                return io.NodeOutput(make())
            raise
        return io.NodeOutput(7)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Eval, CleansUp]


async def comfy_entrypoint():
    return Extension()
"""
MODULE = "v4_probe_pack"


def start_probe(folder):
    path = folder / f"{MODULE}.py"
    path.write_text(PROBE, encoding="utf-8")
    asyncio.run(_pack_host.load(str(path), MODULE))
    return _pack_processes.processes[str(path)]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """One pack process of the probe pack, shared by the tests of a module."""
    process = start_probe(tmp_path_factory.mktemp("probe"))
    yield process
    _pack_processes.unload(process.module_path)


def in_a_node(function, *args):
    """Call `function` as ComfyUI's prompt thread calls a pack's objects: inside a node."""
    with CurrentNodeContext("prompt", "node"):
        return function(*args)


async def evaluate(process, expression, value=None):
    """Evaluate `expression` in the probe pack; `value` is sent to it as a node input."""
    inputs = {"expression": expression, "value": value}
    return (await process.run(("node", "Eval", "execute"), (inputs, {}))).args[0]
