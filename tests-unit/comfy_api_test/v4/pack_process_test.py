import asyncio
import copy
import logging
import os
import sys

import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
import nodes
import server
from comfy.cli_args import args
from comfy.model_management import get_torch_device
from comfy_api.v4 import _codec, _pack_host, _pack_launch, _pack_processes
from execution import PromptExecutor, validate_prompt
from conftest import WHERE, RecordingServer, evaluate, needs_pack_process

pytestmark = needs_pack_process

COMFY_ROOT = os.path.dirname(os.path.abspath(nodes.__file__))

# Ends a pack's source: `meet` waits until `count` callers have arrived, or gives up after 30 s.
MEET = """

import os


async def meet(folder, name, count):
    open(os.path.join(folder, name), "w").close()
    for _ in range(3000):
        if len(os.listdir(folder)) >= count:
            return True
        await asyncio.sleep(0.01)
    return False
"""

BEHAVIOUR_PACK = (
    """\
import asyncio
import collections
import itertools
import os
from fractions import Fraction

import torch
from comfy_api.v4 import (
    ComfyAPISync,
    ComfyExtension,
    InputImpl,
    InterruptProcessingException,
    Types,
    eng,
    io,
    ui,
)
from comfy_api.v4.util import folders
from comfy_execution.graph_utils import ExecutionBlocker

Pair = collections.namedtuple("Pair", "left right")
PAIR = io.Custom("PAIR")
FUNCTION = io.Custom("FUNCTION")
fingerprints = itertools.count()


class SeedError(Exception):
    pass


def double(image):
    return image * 2

"""
    + WHERE
    + """

class Meet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.String.Input(name) for name in ("folder", "name")] + [io.Int.Input("count")]
        return schema("Meet", inputs, [io.Boolean.Output()])

    @classmethod
    async def execute(cls, folder, name, count):
        return io.NodeOutput(await meet(folder, name, count))


class MakePair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("MakePair", [io.Int.Input("value")], [PAIR.Output()])

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(Pair(value, value + 1))


class Unpair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Unpair", [PAIR.Input("pair")], [io.Int.Output(), io.String.Output()])

    @classmethod
    def execute(cls, pair):
        return io.NodeOutput(pair.left, type(pair).__name__)


class Block(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Block", outputs=[io.Int.Output(), io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(1, ExecutionBlocker(None))


class Raise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Raise", [io.String.Input("error")], [io.Int.Output()])

    @classmethod
    async def execute(cls, error):
        raise {
            "built-in": ValueError("bad value"),
            "cancel": InterruptProcessingException(),
            "own": SeedError("seed out of range"),
        }[error]


class Folders(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Folders", [io.String.Input("folder")], [io.AnyType.Output()])

    @classmethod
    def execute(cls, folder):
        folders.add_model_folder_path("v4_pack_process_test", folder)
        return io.NodeOutput(
            (
                folders.get_output_directory(),
                folders.output_directory,
                folders.get_folder_paths("v4_pack_process_test"),
                ComfyAPISync().device.info()["torch_device"],
            )
        )


class Preview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Preview", is_output_node=True)

    @classmethod
    def execute(cls):
        return io.NodeOutput(ui=ui.PreviewImage(torch.full((1, 4, 4, 3), 0.5)))


class Video(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Video", outputs=[io.Video.Output()])

    @classmethod
    def execute(cls):
        images = torch.linspace(0, 1, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3)
        components = Types.VideoComponents(images=images, frame_rate=Fraction(25))
        return io.NodeOutput(InputImpl.VideoFromComponents(components))


class Function(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Function", outputs=[FUNCTION.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(double)


class MakeImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("MakeImage", outputs=[io.Image.Output()])

    @classmethod
    async def execute(cls):
        await asyncio.sleep(0.05)
        await eng.device.info()
        return io.NodeOutput(torch.full((1, 8, 8, 3), 0.25))


class CallsComfyUI(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("CallsComfyUI", [io.String.Input("folder")], [io.String.Output()])

    @classmethod
    async def execute(cls, folder):
        open(os.path.join(folder, "calling"), "w").close()
        return io.NodeOutput((await eng.device.info())["torch_device"])


class Validated(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("value"), io.Int.Input("other")]
        return schema("Validated", inputs, [io.Int.Output()])

    @classmethod
    def validate_inputs(cls, value):
        return value % 2 == 0 or "value must be even"

    @classmethod
    def execute(cls, value, other):
        return io.NodeOutput(value + other)


class Lazy(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Int.Input("a", lazy=True), io.Int.Input("b", lazy=True)]
        return schema("Lazy", inputs, [io.Int.Output()])

    @classmethod
    def check_lazy_status(cls, a=None, b=None):
        return ["b"] if a == 0 else []

    @classmethod
    def execute(cls, a, b=None):
        return io.NodeOutput(b if a == 0 else a)


class Fingerprinted(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Fingerprinted", [io.Int.Input("value")], [io.Int.Output()])

    @classmethod
    def fingerprint_inputs(cls, value):
        return next(fingerprints)

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            Where, Meet, MakePair, Unpair, Block, Raise, Folders, Preview, Video, Function,
            MakeImage, CallsComfyUI, Validated, Lazy, Fingerprinted,
        ]


async def comfy_entrypoint():
    return Extension()
"""
    + MEET
)

OTHER_PACK = """\
from comfy_api.v4 import ComfyExtension, io


class TakePair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Custom("PAIR").Input("pair")]
        return io.Schema(node_id="TakePair", inputs=inputs, is_output_node=True)

    @classmethod
    def execute(cls, pair):
        return io.NodeOutput()


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [TakePair]


async def comfy_entrypoint():
    return Extension()
"""

HOST_PACK = (
    """\
import asyncio
import os
import time

from comfy_api.v4 import ComfyExtension, io
from server import PromptServer


class Keep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Keep", inputs=[io.AnyType.Input("value")], is_output_node=True)

    @classmethod
    def execute(cls, value):
        PromptServer.instance.sent.append(("kept", value, None))
        return io.NodeOutput()


class HostMeet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.String.Input(name) for name in ("folder", "name")] + [io.Int.Input("count")]
        return io.Schema(node_id="HostMeet", inputs=inputs, outputs=[io.Boolean.Output()])

    @classmethod
    async def execute(cls, folder, name, count):
        return io.NodeOutput(await meet(folder, name, count))


class ApplyLater(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Image.Input("image"), io.Custom("FUNCTION").Input("function"), io.String.Input("folder")]
        return io.Schema(node_id="ApplyLater", inputs=inputs, outputs=[io.Image.Output()])

    @classmethod
    def execute(cls, image, function, folder):
        # Holds ComfyUI's thread until the pack's other node calls ComfyUI, then waits on the pack.
        for _ in range(3000):
            if os.path.exists(os.path.join(folder, "calling")):
                break
            time.sleep(0.01)
        return io.NodeOutput(function(image))


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Keep, HostMeet, ApplyLater]


async def comfy_entrypoint():
    return Extension()
"""
    + MEET
)

SCHEMA_PACK = """\
import enum
import os

import torch
from comfy_api.v4 import ComfyExtension, io


@io.comfytype(io_type="POINT")
class Point(io.ComfyTypeIO):
    Type = tuple


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


class Rich(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="V4Rich",
            display_name="Rich node",
            category="v4/test",
            description="Every kind of input",
            inputs=[
                io.Int.Input("count", default=2, min=0, max=10, step=2, tooltip="how many"),
                io.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.01, optional=True),
                io.String.Input("text", multiline=True, default="hi"),
                io.Combo.Input("mode", options=Mode, default=Mode.SLOW),
                io.Boolean.Input("flag", default=True),
                io.Image.Input("image", optional=True),
                io.Custom("V4_THING").Input("thing", optional=True),
                Point.Input("point", optional=True),
                io.DynamicCombo.Input(
                    "choice",
                    options=[
                        io.DynamicCombo.Option("a", [io.Int.Input("a_value", default=1)]),
                        io.DynamicCombo.Option("b", []),
                    ],
                ),
            ],
            outputs=[
                io.Int.Output(display_name="pid"),
                io.Float.Output(is_output_list=True),
                Point.Output(),
                io.Custom("V4_THING").Output(),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, count, text, mode, flag, choice, strength=0.5, image=None, thing=None, point=None):
        values = (torch.arange(count) * strength).tolist()
        return io.NodeOutput(os.getpid(), values, (count, len(text)), None)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Rich]


async def comfy_entrypoint():
    return Extension()
"""

V1_PACK = """\
class Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"


NODE_CLASS_MAPPINGS = {"V1Node": Node}
"""

LATEST_PACK = """\
from comfy_api.latest import ComfyExtension


class Extension(ComfyExtension):
    async def get_node_list(self):
        return []


async def comfy_entrypoint():
    return Extension()
"""

ROUTE_PACK = """\
import os

from comfy_api.v4 import ComfyExtension


async def where(request):
    body = f"{os.getpid()} {request['match_info']['name']} {request['body'].decode()}"
    return {"status": 200, "headers": {"content-type": "text/plain"}, "body": body.encode()}


class Extension(ComfyExtension):
    async def get_node_list(self):
        return []

    async def get_routes(self):
        return [{"method": "POST", "path": "/v4_route_pack/{name}", "handler": where}]


async def comfy_entrypoint():
    return Extension()
"""


def start(folder, name, source):
    """Start a pack process for `source` as the loader would; return the pack's extension."""
    path = folder / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    module = asyncio.run(_pack_host.load(str(path), str(path).replace(".", "_x_")))
    return module.comfy_entrypoint()


async def started(folder, name, source):
    """`start`, from a running event loop."""
    path = folder / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return (await _pack_host.load(str(path), name)).comfy_entrypoint()


@pytest.fixture(scope="module")
def behaviour(tmp_path_factory):
    """The pack whose nodes most tests use, in one process for this module."""
    folder = tmp_path_factory.mktemp("behaviour")
    yield start(folder, "v4_behaviour_pack", BEHAVIOUR_PACK)
    _pack_processes.unload(str(folder / "v4_behaviour_pack.py"))


@pytest.fixture(scope="module")
def other(tmp_path_factory):
    """A second isolated pack, in its own process."""
    folder = tmp_path_factory.mktemp("other")
    yield start(folder, "v4_other_pack", OTHER_PACK)
    _pack_processes.unload(str(folder / "v4_other_pack.py"))


@pytest.fixture
def isolated(behaviour, other, load_pack, monkeypatch):
    """Register the isolated packs' nodes and load the in-process host pack; return the loader."""
    for extension in (behaviour, other):
        for node_cls in extension.nodes:
            monkeypatch.setitem(
                nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
            )

    async def load_host_pack():
        await load_pack(HOST_PACK, "v4_host_pack")

    return load_host_pack


@pytest.fixture
def pack_processes(monkeypatch):
    """The pack processes ComfyUI's loader starts in the test; they end with the test."""
    started = []

    class Recorded(_pack_host.PackProcess):
        def __init__(self, *args):
            super().__init__(*args)
            started.append(self)

    monkeypatch.setattr(_pack_host, "PackProcess", Recorded)
    yield started
    for process in started:
        _pack_processes.unload(process.module_path)


def kept(server):
    return [data for event, data, _ in server.sent if event == "kept"]


def kept_by_type(server):
    """What the Keep nodes kept, by type: they run in no fixed order."""
    return {type(value): value for value in kept(server)}


def link(node_id, output=0):
    return [node_id, output]


def keep(value):
    return {"class_type": "Keep", "inputs": {"value": value}}


def failure(executor):
    [(event, data)] = [
        m
        for m in executor.status_messages
        if m[0].startswith("execution_")
        and m[0] not in ("execution_start", "execution_cached")
    ]
    return event, data


@pytest.mark.asyncio
async def test_a_pack_in_its_own_process_registers_the_same_nodes_and_gives_the_same_results(
    load_pack, run_prompt, pack_processes, monkeypatch
):
    def described():
        node_cls = nodes.NODE_CLASS_MAPPINGS["V4Rich"]
        info = {**node_cls.GET_NODE_INFO_V1(), "python_module": None}
        return (
            info,
            node_cls.INPUT_TYPES(),
            node_cls.RETURN_TYPES,
            node_cls.OUTPUT_IS_LIST,
        )

    prompt = {
        "rich": {
            "class_type": "V4Rich",
            "inputs": {
                "count": 4,
                "text": "abc",
                "mode": "fast",
                "flag": False,
                "choice": "b",
                "strength": 0.25,
            },
        },
        "pid": keep(link("rich", 0)),
        "values": keep(link("rich", 1)),
        "point": keep(link("rich", 2)),
    }

    await load_pack(HOST_PACK, "v4_host_pack")
    await load_pack(SCHEMA_PACK, "v4_schema_pack")
    in_process = described()
    _, server = await run_prompt(prompt, ["pid", "values", "point"])
    results = sorted(kept(server), key=repr)
    monkeypatch.setattr(args, "pack_process", ["v4_schema_isolated"])
    await load_pack(SCHEMA_PACK, "v4_schema_isolated")
    isolated = described()
    _, server = await run_prompt(prompt, ["pid", "values", "point"])
    isolated_results = sorted(kept(server), key=repr)

    assert isolated == in_process
    assert results == sorted([os.getpid(), (4, 3), 0.0, 0.25, 0.5, 0.75], key=repr)
    assert isolated_results == sorted(
        [pack_processes[0].process.pid, (4, 3), 0.0, 0.25, 0.5, 0.75], key=repr
    )
    assert not [
        m for name, m in sys.modules.items() if "v4_schema_isolated" in name and m
    ]


@pytest.mark.asyncio
async def test_with_no_pack_named_a_v4_pack_loads_and_runs_in_comfyuis_process(
    load_pack, run_prompt, monkeypatch
):
    def no_process(start):
        raise AssertionError("a pack process was started")

    monkeypatch.setattr(_pack_launch, "launcher", lambda: no_process)
    monkeypatch.setattr(args, "pack_process", [])
    await load_pack(BEHAVIOUR_PACK, "v4_behaviour_pack")
    await load_pack(HOST_PACK, "v4_host_pack")

    _, server = await run_prompt(
        {"where": {"class_type": "Where", "inputs": {}}, "keep": keep(link("where"))},
        ["keep"],
    )

    assert kept(server) == [os.getpid()]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [V1_PACK, LATEST_PACK], ids=["V1", "V3 latest"])
async def test_a_pack_that_is_not_a_v4_pack_does_not_load_in_its_own_process(
    tmp_path, monkeypatch, caplog, pack_processes, source
):
    monkeypatch.setattr(args, "pack_process", ["not_v4_pack"])
    (tmp_path / "not_v4_pack.py").write_text(source, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert not await nodes.load_custom_node(str(tmp_path / "not_v4_pack.py"))

    assert "is not a V4 pack" in caplog.text
    assert pack_processes[0].process.wait(timeout=10) is not None


@pytest.mark.asyncio
async def test_a_route_of_a_pack_in_its_own_process_answers_through_comfyui(
    host, tmp_path, monkeypatch, pack_processes
):
    monkeypatch.setattr(args, "pack_process", ["v4_route_pack"])
    (tmp_path / "v4_route_pack.py").write_text(ROUTE_PACK, encoding="utf-8")
    assert await nodes.load_custom_node(str(tmp_path / "v4_route_pack.py"))

    host.add_routes()
    async with TestClient(TestServer(host.app)) as client:
        response = await client.post("/api/v4_route_pack/alpha", data=b"payload")
        text = await response.text()

    assert (response.status, text) == (
        200,
        f"{pack_processes[0].process.pid} alpha payload",
    )


@pytest.mark.asyncio
async def test_three_async_nodes_of_a_pack_in_its_own_process_run_side_by_side(
    isolated, run_prompt, tmp_path
):
    await isolated()

    def meet(class_type, name):
        inputs = {"folder": str(tmp_path), "name": name, "count": 4}
        return {"class_type": class_type, "inputs": inputs}

    prompt = {f"meet{i}": meet("Meet", f"meet{i}") for i in range(3)}
    prompt["host"] = meet("HostMeet", "host")
    prompt |= {f"keep_{node}": keep(link(node)) for node in list(prompt)}

    executor, server = await run_prompt(
        prompt, [n for n in prompt if n.startswith("keep_")]
    )

    assert kept(server) == [True] * 4


@pytest.mark.asyncio
async def test_a_blocked_output_blocks_only_the_branch_it_feeds(isolated, run_prompt):
    await isolated()

    executor, server = await run_prompt(
        {
            "block": {"class_type": "Block", "inputs": {}},
            "first": keep(link("block", 0)),
            "second": keep(link("block", 1)),
        },
        ["first", "second"],
    )

    assert executor.success, executor.status_messages
    assert kept(server) == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, event, exception_type",
    [
        ("built-in", "execution_error", "ValueError"),
        ("own", "execution_error", "{module}.SeedError"),
        ("cancel", "execution_interrupted", None),
    ],
)
async def test_an_error_in_a_pack_process_reaches_comfyui_as_the_same_class(
    isolated, run_prompt, behaviour, error, event, exception_type
):
    await isolated()

    executor, _ = await run_prompt(
        {
            "raise": {"class_type": "Raise", "inputs": {"error": error}},
            "keep": keep(link("raise")),
        },
        ["keep"],
    )

    reported, data = failure(executor)
    assert (reported, data["node_id"]) == (event, "raise")
    if exception_type is not None:
        module = behaviour.nodes[0].__module__
        assert data["exception_type"] == exception_type.format(module=module)


@pytest.mark.asyncio
async def test_a_packs_own_object_reaches_its_next_node_as_itself_and_no_other_pack(
    isolated, run_prompt
):
    await isolated()
    pair = {"class_type": "MakePair", "inputs": {"value": 5}}

    executor, server = await run_prompt(
        {
            "pair": pair,
            "unpair": {"class_type": "Unpair", "inputs": {"pair": link("pair")}},
            "left": keep(link("unpair", 0)),
            "type": keep(link("unpair", 1)),
            "raw": keep(link("pair")),
        },
        ["left", "type", "raw"],
    )
    by_type = kept_by_type(server)
    left, type_name, raw = by_type[int], by_type[str], by_type[_codec.Remote]
    executor_other, _ = await run_prompt(
        {
            "pair": pair,
            "take": {"class_type": "TakePair", "inputs": {"pair": link("pair")}},
        },
        ["take"],
    )

    assert (left, type_name) == (5, "Pair")
    assert type(raw) is _codec.Remote
    with pytest.raises(AttributeError, match="lives in pack v4_behaviour_pack.py"):
        raw.left
    event, data = failure(executor_other)
    assert (event, data["node_id"], data["exception_type"]) == (
        "execution_error",
        "take",
        "TypeError",
    )
    assert "only that pack's nodes can use it" in data["exception_message"]


@pytest.mark.asyncio
async def test_a_video_from_a_pack_process_reaches_comfyuis_own_node_as_a_video(
    isolated, run_prompt, monkeypatch
):
    await isolated()
    monkeypatch.setitem(
        sys.modules, os.path.join(COMFY_ROOT, "comfy_extras", "nodes_video"), None
    )
    assert await nodes.load_custom_node(
        os.path.join(COMFY_ROOT, "comfy_extras", "nodes_video.py"),
        module_parent="comfy_extras",
    )

    executor, server = await run_prompt(
        {
            "video": {"class_type": "Video", "inputs": {}},
            "parts": {
                "class_type": "GetVideoComponents",
                "inputs": {"video": link("video")},
            },
            "images": keep(link("parts", 0)),
            "rate": keep(link("parts", 2)),
        },
        ["images", "rate"],
    )

    assert executor.success, executor.status_messages
    by_type = kept_by_type(server)
    images, rate = by_type[torch.Tensor], by_type[float]
    assert torch.equal(images, torch.linspace(0, 1, 2 * 4 * 4 * 3).reshape(2, 4, 4, 3))
    assert rate == 25


@pytest.mark.asyncio
async def test_a_prompt_mixes_a_pack_process_an_in_process_pack_and_comfyuis_nodes(
    isolated, run_prompt, tmp_path
):
    """While ComfyUI waits on the pack's function, it serves the pack's other node's call."""
    await isolated()

    executor, server = await run_prompt(
        {
            "function": {"class_type": "Function", "inputs": {}},
            "image": {"class_type": "MakeImage", "inputs": {}},
            "invert": {"class_type": "ImageInvert", "inputs": {"image": link("image")}},
            "apply": {
                "class_type": "ApplyLater",
                "inputs": {
                    "image": link("invert"),
                    "function": link("function"),
                    "folder": str(tmp_path),
                },
            },
            "calls": {
                "class_type": "CallsComfyUI",
                "inputs": {"folder": str(tmp_path)},
            },
            "keep_apply": keep(link("apply")),
            "keep_calls": keep(link("calls")),
        },
        ["keep_apply", "keep_calls"],
    )

    assert executor.success, executor.status_messages
    by_type = kept_by_type(server)
    applied, device = by_type[torch.Tensor], by_type[str]
    assert torch.equal(applied, torch.full((1, 8, 8, 3), 1.5))
    assert device == str(get_torch_device())


@pytest.mark.asyncio
async def test_util_folders_and_the_sync_api_in_a_pack_process_answer_with_comfyuis(
    isolated, run_prompt, monkeypatch, tmp_path
):
    await isolated()
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))

    executor, server = await run_prompt(
        {
            "folders": {"class_type": "Folders", "inputs": {"folder": str(tmp_path)}},
            "keep": keep(link("folders")),
        },
        ["keep"],
    )

    assert executor.success, executor.status_messages
    [(output, attribute, added, device)] = kept(server)
    assert output == attribute == folder_paths.get_output_directory()
    assert (
        added
        == folder_paths.get_folder_paths("v4_pack_process_test")
        == [str(tmp_path)]
    )
    assert device == str(get_torch_device())


@pytest.mark.asyncio
async def test_a_preview_saved_in_a_pack_process_is_in_comfyuis_temp_folder(
    isolated, run_prompt
):
    await isolated()

    executor, server = await run_prompt(
        {"preview": {"class_type": "Preview", "inputs": {}}}, ["preview"]
    )

    assert executor.success, executor.status_messages
    [output] = [data["output"] for event, data, _ in server.sent if event == "executed"]
    [image] = output["images"]
    path = os.path.join(
        folder_paths.get_temp_directory(), image["subfolder"], image["filename"]
    )
    assert image["type"] == "temp" and os.path.isfile(path)
    os.remove(path)


@pytest.mark.asyncio
async def test_a_packs_validation_runs_with_the_inputs_its_signature_names(
    isolated, run_prompt
):
    await isolated()

    def prompt(value):
        node = {"class_type": "Validated", "inputs": {"value": value, "other": 1}}
        return {"node": node, "keep": keep(link("node"))}

    valid, _, _, errors = await validate_prompt("v4-node", prompt(3), ["keep"])
    executor, server = await run_prompt(prompt(4), ["keep"])

    assert not valid
    assert errors["node"]["errors"][0]["details"] == "value - value must be even"
    assert kept(server) == [5]


@pytest.mark.asyncio
async def test_a_packs_lazy_status_decides_which_inputs_are_evaluated(
    isolated, run_prompt
):
    await isolated()

    executor, server = await run_prompt(
        {
            "fails": {"class_type": "Raise", "inputs": {"error": "built-in"}},
            "seven": {"class_type": "Lazy", "inputs": {"a": 7, "b": link("fails")}},
            "lazy": {"class_type": "Lazy", "inputs": {"a": 0, "b": link("seven")}},
            "keep": keep(link("lazy")),
        },
        ["keep"],
    )

    assert executor.success, executor.status_messages
    assert kept(server) == [7]


@pytest.mark.asyncio
async def test_a_packs_fingerprint_decides_whether_its_node_runs_again(
    isolated, monkeypatch
):
    await isolated()
    recording = RecordingServer()
    monkeypatch.setattr(server.PromptServer, "instance", recording, raising=False)
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    prompt = {
        "printed": {"class_type": "Fingerprinted", "inputs": {"value": 1}},
        "where": {"class_type": "Where", "inputs": {}},
        "keep_printed": keep(link("printed")),
        "keep_where": keep(link("where")),
    }

    for _ in range(2):
        await executor.execute_async(
            copy.deepcopy(prompt),
            "v4-node",
            {},
            execute_outputs=["keep_printed", "keep_where"],
        )

    [cached] = [
        data["nodes"]
        for event, data in executor.status_messages
        if event == "execution_cached"
    ]
    assert "where" in cached and "printed" not in cached


EXPANDING_PACK = """\
from comfy_api.v4 import ComfyExtension, io
from comfy_execution.graph_utils import GraphBuilder


class Add{n}(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Add{n}", inputs=[io.Int.Input("a")], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, a):
        return io.NodeOutput(a + {n})


class Expand{n}(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Expand{n}", inputs=[io.Int.Input("a")], outputs=[io.Int.Output()], enable_expand=True
        )

    @classmethod
    def execute(cls, a):
        graph = GraphBuilder()
        added = graph.node("Add{n}", a=a)
        return io.NodeOutput(added.out(0), expand=graph.finalize())


class Count{n}(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Count{n}", outputs=[io.Int.Output(is_output_list=True)])

    @classmethod
    def execute(cls):
        return io.NodeOutput([1, 2])


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Add{n}, Expand{n}, Count{n}]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
async def test_the_nodes_pack_processes_expand_into_are_named_as_in_process(
    load_pack, run_prompt, tmp_path, monkeypatch
):
    prompt = {
        "list": {"class_type": "Count10", "inputs": {}},
        "each": {"class_type": "Expand10", "inputs": {"a": link("list")}},
        "one": {"class_type": "Expand1000", "inputs": {"a": 2}},
        "keep_each": keep(link("each")),
        "keep_one": keep(link("one")),
    }

    async def run():
        _, server = await run_prompt(prompt, ["keep_each", "keep_one"])
        executed = {
            data["node"] for event, data, _ in server.sent if event == "executing"
        }
        return executed, sorted(map(str, kept(server)))

    await load_pack(HOST_PACK, "v4_host_pack")
    for n in (10, 1000):
        await load_pack(EXPANDING_PACK.format(n=n), f"v4_expanding_{n}")
    in_process = await run()
    for n in (10, 1000):
        extension = await started(
            tmp_path, f"v4_isolated_{n}", EXPANDING_PACK.format(n=n)
        )
        for node_cls in extension.nodes:
            monkeypatch.setitem(
                nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
            )
    try:
        isolated = await run()
    finally:
        for n in (10, 1000):
            _pack_processes.unload(str(tmp_path / f"v4_isolated_{n}.py"))

    assert isolated == in_process
    assert {"each.0.0.1", "each.1.0.1", "one.0.0.1"} <= in_process[0]


FOLDERS_PACK = """\
from comfy_api.v4 import ComfyExtension, io
from comfy_api.v4.util import folders


class Assign(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Assign", inputs=[io.String.Input("folder")], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, folder):
        folders.folder_names_and_paths["v4_assigned"] = ([folder], {".bin"})
        return io.NodeOutput(len(folders.get_folder_paths("v4_assigned")))


class Add(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Add", inputs=[io.String.Input("folder")], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, folder):
        folders.add_model_folder_path("v4_assigned", folder)
        return io.NodeOutput(len(folders.get_folder_paths("v4_assigned")))


class Changes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Changes", outputs=[io.AnyType.Output()])

    @classmethod
    def execute(cls):
        refused = []
        for change in (
            lambda table: table.setdefault("v4_other", ([], set())),
            lambda table: table.pop("checkpoints"),
            lambda table: table.update(v4_other=([], set())),
            lambda table: table["checkpoints"][0].append("/v4"),
            lambda table: table["checkpoints"][1].add(".v4"),
        ):
            try:
                change(folders.folder_names_and_paths)
            except (TypeError, AttributeError) as error:
                refused.append(type(error).__name__)
        return io.NodeOutput(refused)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Assign, Add, Changes]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
async def test_a_pack_process_assigning_a_model_folder_entry_changes_comfyuis_table_and_other_changes_are_refused(
    load_pack, run_prompt, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    checkpoints = copy.deepcopy(folder_paths.folder_names_and_paths["checkpoints"])
    await load_pack(HOST_PACK, "v4_host_pack")
    extension = await started(tmp_path, "v4_folders_pack", FOLDERS_PACK)
    for node_cls in extension.nodes:
        monkeypatch.setitem(
            nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
        )
    prompt = {
        "assign": {"class_type": "Assign", "inputs": {"folder": str(tmp_path)}},
        "changes": {"class_type": "Changes", "inputs": {}},
        "keep_assign": keep(link("assign")),
        "keep_changes": keep(link("changes")),
    }

    try:
        _, server = await run_prompt(prompt, ["keep_assign", "keep_changes"])
    finally:
        _pack_processes.unload(str(tmp_path / "v4_folders_pack.py"))

    assert kept_by_type(server) == {
        int: 1,
        list: ["TypeError"] * 3 + ["AttributeError"] * 2,
    }
    assert folder_paths.folder_names_and_paths["v4_assigned"] == (
        [str(tmp_path)],
        {".bin"},
    )
    assert folder_paths.folder_names_and_paths["checkpoints"] == checkpoints
    assert "v4_other" not in folder_paths.folder_names_and_paths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("where", "accepted"),
    [
        ("the pack's own folder", True),
        ("ComfyUI's models folder", True),
        ("a folder ComfyUI lists already", True),
        ("another folder", False),
    ],
)
@pytest.mark.parametrize("node", ["Assign", "Add"])
async def test_a_pack_process_makes_a_model_folder_only_where_comfyui_lists_or_in_its_own_folder(
    load_pack, run_prompt, tmp_path, monkeypatch, node, where, accepted
):
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    folders = {
        "the pack's own folder": tmp_path / "v4_folders_pack" / "weights",
        "ComfyUI's models folder": tmp_path / "models" / "mine",
        "a folder ComfyUI lists already": tmp_path / "listed",
        "another folder": tmp_path / "elsewhere",
    }
    for folder in folders.values():
        folder.mkdir(parents=True)
    (folders["another folder"] / "private.bin").write_bytes(b"not the pack's")
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    folder_paths.add_model_folder_path("v4_listed", str(tmp_path / "listed"))
    pack = tmp_path / "v4_folders_pack"
    (pack / "__init__.py").write_text(FOLDERS_PACK, encoding="utf-8")
    await load_pack(HOST_PACK, "v4_host_pack")
    extension = (await _pack_host.load(str(pack), "v4_folders_pack")).comfy_entrypoint()
    for node_cls in extension.nodes:
        monkeypatch.setitem(
            nodes.NODE_CLASS_MAPPINGS, node_cls.GET_SCHEMA().node_id, node_cls
        )
    prompt = {
        "node": {"class_type": node, "inputs": {"folder": str(folders[where])}},
        "keep": keep(link("node")),
    }

    try:
        executor, server = await run_prompt(prompt, ["keep"])
    finally:
        _pack_processes.unload(str(pack))

    assert executor.success is accepted, executor.status_messages
    assert ("v4_assigned" in folder_paths.folder_names_and_paths) is accepted
    if accepted:
        assert kept(server) == [1]
    else:
        event, data = failure(executor)
        assert (event, data["exception_type"]) == ("execution_error", "ValueError")
        assert "cannot be a model folder" in data["exception_message"]


@pytest.mark.parametrize(
    "use",
    [
        "__import__('sys').stdout.reconfigure(encoding='utf-8')",
        "__import__('sys').stdout.buffer.write(b'')",
    ],
    ids=["reconfigure", "buffer"],
)
def test_a_packs_stdout_has_what_comfyuis_stream_has(probe, use):
    assert asyncio.run(evaluate(probe, f"caught(lambda: {use})")) is None


@pytest.mark.asyncio
async def test_what_a_pack_process_prints_and_logs_goes_where_comfyuis_own_output_goes(
    run_node, capsys, caplog
):
    body = (
        "import logging, sys\n"
        "print('printed', 1)\n"
        "print('printed', 2)\n"
        "logging.getLogger('v4.pack').warning('logged %s', 3)\n"
        "sys.stderr.write('written to stderr\\n')\n"
    )

    await run_node(body)

    out, err = capsys.readouterr()
    assert "printed 1\nprinted 2\n" in out
    assert "written to stderr\n" in err
    assert [
        (r.name, r.levelname, r.getMessage())
        for r in caplog.records
        if r.name == "v4.pack"
    ] == [("v4.pack", "WARNING", "logged 3")]


PRINTS_AND_DIES = """\
import os
import sys

from comfy_api.v4 import ComfyExtension, io


class PrintsAndDies(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="PrintsAndDies", is_output_node=True)

    @classmethod
    def execute(cls):
        print("the last words")
        sys.stderr.write("and the very last")
        os._exit(3)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [PrintsAndDies]


async def comfy_entrypoint():
    return Extension()
"""


@pytest.mark.asyncio
async def test_what_a_pack_process_printed_before_it_died_is_not_lost(
    run_prompt, tmp_path, monkeypatch, capsys
):
    extension = await started(tmp_path, "v4_dies_pack", PRINTS_AND_DIES)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PrintsAndDies", extension.nodes[0])

    try:
        executor, _ = await run_prompt(
            {"dies": {"class_type": "PrintsAndDies", "inputs": {}}}, ["dies"]
        )
    finally:
        _pack_processes.unload(str(tmp_path / "v4_dies_pack.py"))

    out, err = capsys.readouterr()
    assert failure(executor)[0] == "execution_error"
    assert "the last words\n" in out and "and the very last" in err


@pytest.mark.asyncio
async def test_naming_one_of_comfyuis_own_node_modules_does_not_run_it_in_a_pack_process(
    monkeypatch,
):
    def no_process(start):
        raise AssertionError("a pack process was started")

    monkeypatch.setattr(_pack_launch, "launcher", lambda: no_process)
    monkeypatch.setattr(args, "pack_process", ["nodes_upscale_model"])
    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", dict(nodes.NODE_CLASS_MAPPINGS))
    monkeypatch.setitem(
        sys.modules,
        os.path.join(COMFY_ROOT, "comfy_extras", "nodes_upscale_model"),
        None,
    )

    assert await nodes.load_custom_node(
        os.path.join(COMFY_ROOT, "comfy_extras", "nodes_upscale_model.py"),
        module_parent="comfy_extras",
    )
    assert "UpscaleModelLoader" in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_the_web_folder_of_a_pack_in_its_own_process_is_served_by_comfyui(
    tmp_path, monkeypatch, pack_processes, node_tables
):
    monkeypatch.setattr(args, "pack_process", ["v4_web_pack"])
    (tmp_path / "v4_web_pack_js").mkdir()
    (tmp_path / "v4_web_pack.py").write_text(
        'WEB_DIRECTORY = "v4_web_pack_js"\n'
        + ROUTE_PACK.replace("/v4_route_pack", "/v4_web"),
        encoding="utf-8",
    )

    assert await nodes.load_custom_node(str(tmp_path / "v4_web_pack.py"))

    assert nodes.EXTENSION_WEB_DIRS[str(tmp_path / "v4_web_pack")] == str(
        tmp_path / "v4_web_pack_js"
    )
