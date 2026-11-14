import asyncio
import contextlib
import copy
import gc
import logging
import os
import signal
import subprocess
import sys
import threading
import time

import psutil
import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
import nodes
import server
from comfy.cli_args import args
from comfy.model_management import interrupt_current_processing, processing_interrupted
from comfy_api.latest import io
from comfy_api.v4 import (
    InterruptProcessingException,
    _channel,
    _pack_host,
    _pack_nodes,
    _pack_processes,
    _packs,
)
from comfy_execution.utils import CurrentNodeContext
from conftest import (
    PROBE,
    WHERE,
    RecordingServer,
    evaluate,
    needs_pack_process,
    start_probe,
    wait_until,
)
from execution import PromptExecutor, validate_prompt
from model_test import model as model_fixture

pytestmark = needs_pack_process

model = model_fixture

LIFETIME_PACK = (
    """\
import asyncio
import gc
import os
import socket
import subprocess
import time
import weakref

import torch

from comfy_api.v4 import ComfyAPISync, ComfyExtension, eng, io
from comfy_api.v4.util import folders

hooks = weakref.WeakSet()
awaiting = []
cancelled = []

"""
    + WHERE
    + """

class WhereFingerprinted(Where):
    @classmethod
    def define_schema(cls):
        return schema("WhereFingerprinted", outputs=[io.Int.Output()])

    @classmethod
    def fingerprint_inputs(cls):
        return "the same"


class Crash(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Crash", [io.AnyType.Input("after", optional=True)], [io.Int.Output()])

    @classmethod
    def execute(cls, after=None):
        os._exit(3)


def start_a_process(how, folder):
    # A process that lives on and would hold this one's socket; its pid is recorded.
    if how == "a forked child":
        child = os.fork()
        if child == 0:
            time.sleep(60)
            os._exit(0)
    elif how == "a program it started":
        os.system(f"sleep 60 & echo $! > {folder}/program.pid")
        return
    elif how == "a program given its descriptors":
        sockets = [s.fileno() for s in gc.get_objects() if isinstance(s, socket.socket)]
        child = subprocess.Popen(["sleep", "60"], pass_fds=sockets).pid
    else:
        return
    with open(f"{folder}/{child}.pid", "w") as file:
        file.write(str(child))


class Large(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Large", outputs=[io.Image.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(torch.zeros(1, 256, 256, 3))


class DiesLeaving(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.String.Input("how"), io.String.Input("folder"), io.AnyType.Input("after")]
        return schema("DiesLeaving", inputs, [io.Int.Output()])

    @classmethod
    def execute(cls, how, folder, after):
        start_a_process(how, folder)
        os._exit(3)


class Spin(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.String.Input("how", optional=True), io.String.Input("folder", optional=True)]
        return schema("Spin", inputs, [io.Int.Output()])

    @classmethod
    def execute(cls, how=None, folder=None):
        start_a_process(how, folder)
        ComfyAPISync().ui.send("cancel", {})
        while True:
            time.sleep(0.01)


class Chatty(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Chatty", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        await eng.ui.send("cancel", {})
        while True:
            await eng.device.info()


class Checks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Checks", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        await eng.ui.send("cancel", {})
        while True:
            await eng.execution.check_interrupted()
            await asyncio.sleep(0.01)


class Awaits(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Awaits", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        awaiting.append(True)
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise


class AfterAwaits(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("AfterAwaits", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        while not awaiting:
            await asyncio.sleep(0.01)
        return io.NodeOutput(len(awaiting))


class Cancelled(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Cancelled", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(len(cancelled))


class PatchCfg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("PatchCfg", [io.Model.Input("model")], [io.Model.Output()])

    @classmethod
    async def execute(cls, model):
        def cfg(args):
            return args["cond"]

        hooks.add(cfg)
        return io.NodeOutput(await eng.model.set_cfg(model, cfg))


class Hooks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Hooks", [io.AnyType.Input("after", optional=True)], [io.Int.Output()])

    @classmethod
    def execute(cls, after=None):
        gc.collect()
        return io.NodeOutput(len(hooks))


class Raises(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema("Raises", [io.String.Input("which")], [io.Int.Output()])

    @classmethod
    async def execute(cls, which):
        raise {
            "SystemExit": SystemExit(3),
            "KeyboardInterrupt": KeyboardInterrupt(),
            "GeneratorExit": GeneratorExit(),
        }[which]


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            Where, Crash, Spin, Chatty, Checks, Awaits, AfterAwaits, Cancelled, PatchCfg, Hooks, Raises,
            Large, DiesLeaving, WhereFingerprinted,
        ]


async def comfy_entrypoint():
    return Extension()
"""
)

ROUTE_PACK = """\
import os
import time

from comfy_api.v4 import ComfyExtension, io


class Where(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="RouteWhere", outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(os.getpid())


async def answer(request):
    what = request["match_info"]["what"]
    if what == "crash":
        os._exit(3)
    if what == "raise":
        raise ValueError("the route failed")
    if what == "busy":
        time.sleep(float(request["body"]))
    return {"status": 200, "headers": {"content-type": "text/plain"}, "body": str(os.getpid()).encode()}


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Where]

    async def get_routes(self):
        return [{"method": "POST", "path": "/v4_lifetime/{what}", "handler": answer}]


async def comfy_entrypoint():
    return Extension()
"""

LISTED_PACK = """\
from comfy_api.v4 import ComfyExtension, io
from comfy_api.v4.util import folders


class Listed(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.Combo.Input("name", options=folders.get_filename_list("v4_listed"))]
        return io.Schema(node_id="Listed", inputs=inputs, outputs=[io.String.Output()])

    @classmethod
    def execute(cls, name):
        return io.NodeOutput(name)


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Listed]


async def comfy_entrypoint():
    return Extension()
"""


class Keep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [io.AnyType.Input("value")]
        return io.Schema(node_id="LifetimeKeep", inputs=inputs, is_output_node=True)

    @classmethod
    def execute(cls, value):
        server.PromptServer.instance.sent.append(("kept", value, None))
        return io.NodeOutput()


class Value(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LifetimeValue",
            inputs=[io.Int.Input("value")],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(value)


class Fail(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LifetimeFail",
            inputs=[io.AnyType.Input("after")],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, after):
        raise ValueError("an unrelated node failed")


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def keep(node_id):
    return node("LifetimeKeep", value=[node_id, 0])


def kept(recording):
    return [data for event, data, _ in recording.sent if event == "kept"]


def outcome(executor):
    """The event the prompt ended with, and its data."""
    [(event, data)] = [
        m
        for m in executor.status_messages
        if m[0] in ("execution_success", "execution_error", "execution_interrupted")
    ]
    return event, data


def ended(popen):
    """Whether the process has ended, or ends within 10 s."""
    try:
        popen.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return False
    return True


async def gone(chains):
    while chains:
        await asyncio.sleep(0.01)


def cached(executor):
    return {
        n
        for event, data in executor.status_messages
        if event == "execution_cached"
        for n in data["nodes"]
    }


async def started(folder, name, source):
    """Start `source` in its own process as the loader would; return its extension and process."""
    path = folder / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    module = await _pack_host.load(str(path), name)
    return module.comfy_entrypoint(), _pack_processes.processes[str(path)]


@pytest.fixture
def table(node_tables):
    """ComfyUI's node table, holding this module's in-process nodes; restored after the test."""
    for node_cls in (Keep, Value, Fail):
        nodes.NODE_CLASS_MAPPINGS[node_cls.GET_SCHEMA().node_id] = node_cls
    return nodes.NODE_CLASS_MAPPINGS


@pytest.fixture
def recording(monkeypatch):
    recording = RecordingServer()
    monkeypatch.setattr(server.PromptServer, "instance", recording, raising=False)
    return recording


@pytest.fixture
def isolated(tmp_path, table):
    """Start packs in their own processes, their nodes in ComfyUI's table; ended after the test."""
    processes = []

    async def start(source, name="v4_lifetime_pack"):
        extension, process = await started(tmp_path, name, source)
        processes.append(process)
        for node_cls in extension.nodes:
            table[node_cls.GET_SCHEMA().node_id] = node_cls
        return process

    yield start
    for process in processes:
        _pack_processes.unload(process.module_path)


@pytest.fixture
def loaded(tmp_path, table, host, monkeypatch):
    """Load packs through ComfyUI's loader, isolated, with a server for their routes. Returns
    whether the loader loaded the pack, and its process."""
    processes = []

    class Recorded(_pack_host.PackProcess):
        def __init__(self, *args):
            super().__init__(*args)
            processes.append(self)

    monkeypatch.setattr(_pack_host, "PackProcess", Recorded)

    async def load(source, name):
        monkeypatch.setattr(args, "pack_process", [*args.pack_process, name])
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        return await nodes.load_custom_node(str(path)), processes[-1]

    load.host = host
    yield load
    for process in processes:
        _pack_processes.unload(process.module_path)
        process.stop()


async def execute(executor, prompt, outputs):
    await executor.execute_async(
        copy.deepcopy(prompt), "v4", {"client_id": "tab"}, execute_outputs=outputs
    )
    return executor


async def work(executor, prompt, outputs):
    """Run `prompt` as ComfyUI's prompt worker runs a queued one (`main.prompt_worker`): while it
    runs, the queue says so, and a flush keeps the node classes it uses."""
    queue = server.PromptServer.instance.prompt_queue
    queue.put((0, "v4", copy.deepcopy(prompt), {"client_id": "tab"}, outputs, {}))
    item, item_id = queue.get(timeout=0)
    await executor.execute_async(item[2], item[1], item[3], item[4])
    queue.task_done(item_id, executor.history_result, status=None)
    return executor


@pytest.mark.asyncio
async def test_a_pack_that_crashes_mid_node_fails_that_node_and_runs_again_in_a_new_process(
    isolated, recording
):
    process = await isolated(LIFETIME_PACK)
    first = process.process.pid
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    prompt = {
        "where": node("Where"),
        "keep_where": keep("where"),
        "value": node("LifetimeValue", value=5),
        "keep_value": keep("value"),
    }
    # The crash comes after the pack's other node has run.
    crash = {"crash": node("Crash", after=["where", 0]), "keep_crash": keep("crash")}

    await execute(executor, prompt | crash, ["keep_where", "keep_value", "keep_crash"])
    event, data = outcome(executor)
    ran_before = sorted(kept(recording))
    recording.sent.clear()
    await execute(executor, prompt, ["keep_where", "keep_value"])

    assert (event, data["node_id"], data["exception_type"]) == (
        "execution_error",
        "crash",
        "comfy_api.v4._channel.PackStopped",
    )
    assert (
        data["exception_message"].strip()
        == "pack v4_lifetime_pack.py stopped (exit code 3) while running node crash"
    )
    assert ran_before == [5, first]
    assert process.chains == {}
    assert outcome(executor)[0] == "execution_success"
    second = process.process.pid
    assert second != first and kept(recording) == [second]
    assert {"value", "keep_value"} <= cached(executor)
    assert not {"where", "keep_where"} & cached(executor)


def end_started(folder):
    """End the processes a pack node started and recorded in `folder`."""
    for pid_file in folder.glob("*.pid"):
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "busy, how",
    [
        ("Spin", None),
        ("Chatty", None),
        ("Spin", "a forked child"),
        ("Spin", "a program it started"),
    ],
    ids=[
        "without calling",
        "calling without checking",
        "after forking a child",
        "after starting a program",
    ],
)
async def test_a_busy_pack_is_killed_a_grace_period_after_the_user_cancels_and_the_next_prompt_works(
    isolated, run_prompt, monkeypatch, tmp_path, busy, how
):
    monkeypatch.setattr(_pack_host, "GRACE", 1.0)
    process = await isolated(LIFETIME_PACK)
    first = process.process
    inputs = {} if how is None else {"how": how, "folder": str(tmp_path)}

    started_at = time.monotonic()
    # The bound only turns a missing kill into a failure instead of a hang.
    try:
        executor, recording = await asyncio.wait_for(
            run_prompt({"spin": node(busy, **inputs), "keep": keep("spin")}, ["keep"]),
            60,
        )
    finally:
        end_started(tmp_path)
    took = time.monotonic() - started_at
    spun = outcome(executor)[0], "cancel" in [event for event, _, _ in recording.sent]
    cancel_cleared = not processing_interrupted()
    executor, recording = await run_prompt(
        {"where": node("Where"), "keep": keep("where")}, ["keep"]
    )

    assert spun == ("execution_interrupted", True)
    assert took >= 1.0
    assert first.poll() is not None
    assert cancel_cleared
    assert kept(recording) == [process.process.pid] and process.process.pid != first.pid


@pytest.mark.asyncio
async def test_a_pack_that_checks_for_the_cancel_is_interrupted_and_keeps_its_process(
    isolated, run_prompt
):
    process = await isolated(LIFETIME_PACK)
    pid = process.process.pid

    executor, recording = await run_prompt(
        {"checks": node("Checks"), "keep": keep("checks")}, ["keep"]
    )

    assert outcome(executor)[0] == "execution_interrupted"
    assert "cancel" in [event for event, _, _ in recording.sent]
    assert (process.process.pid, process.process.poll()) == (pid, None)
    assert not processing_interrupted()


@pytest.mark.asyncio
async def test_a_prompt_that_ends_while_a_pack_node_awaits_cancels_that_nodes_task_in_the_pack(
    isolated, table, recording
):
    process = await isolated(LIFETIME_PACK)
    pid = process.process.pid
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    prompt = {
        "awaits": node("Awaits"),
        "keep_awaits": keep("awaits"),
        "after": node("AfterAwaits"),
        "fail": node("LifetimeFail", after=["after", 0]),
    }

    # As ComfyUI runs a prompt: asyncio.run, which cancels the tasks still pending at the end.
    await asyncio.to_thread(
        executor.execute, prompt, "v4", {"client_id": "tab"}, ["keep_awaits", "fail"]
    )
    event, data = outcome(executor)
    recording.sent.clear()
    await execute(
        executor, {"cancelled": node("Cancelled"), "keep": keep("cancelled")}, ["keep"]
    )

    # The abandoned request's record goes once its answer has come, within one poll.
    await asyncio.wait_for(gone(process.chains), 10)

    assert (event, data["node_id"]) == ("execution_error", "fail")
    assert kept(recording) == [1]
    assert process.process.pid == pid


@pytest.mark.asyncio
async def test_a_hook_comfyui_held_is_released_in_the_pack_when_comfyui_drops_the_patched_model(
    isolated, table, recording, model
):
    class Source(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="LifetimeModel", outputs=[io.Model.Output()])

        @classmethod
        def execute(cls):
            return io.NodeOutput(model)

    table["LifetimeModel"] = Source
    await isolated(LIFETIME_PACK)
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    prompt = {
        "model": node("LifetimeModel"),
        "patched": node("PatchCfg", model=["model", 0]),
        "keep_patched": keep("patched"),
        "hooks": node("Hooks", after=["patched", 0]),
        "keep_hooks": keep("hooks"),
    }

    await execute(executor, prompt, ["keep_patched", "keep_hooks"])
    held = [value for value in kept(recording) if isinstance(value, int)]
    del executor
    recording.sent.clear()
    gc.collect()
    await execute(
        PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0}),
        {"hooks": node("Hooks"), "keep_hooks": keep("hooks")},
        ["keep_hooks"],
    )

    assert held == [1]
    assert kept(recording) == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["SystemExit", "KeyboardInterrupt", "GeneratorExit"])
async def test_an_exception_that_would_end_a_thread_fails_only_the_pack_node_that_raised_it(
    isolated, run_prompt, which
):
    process = await isolated(LIFETIME_PACK)
    pid = process.process.pid

    executor, _ = await run_prompt(
        {"raises": node("Raises", which=which), "keep": keep("raises")}, ["keep"]
    )
    failed = outcome(executor)
    executor, recording = await run_prompt(
        {"where": node("Where"), "keep": keep("where")}, ["keep"]
    )

    assert (failed[0], failed[1]["node_id"], failed[1]["exception_type"]) == (
        "execution_error",
        "raises",
        which,
    )
    assert kept(recording) == [pid]


@pytest.mark.asyncio
async def test_a_route_answers_503_when_the_pack_dies_in_it_and_the_next_request_starts_it_again(
    loaded,
):
    _, process = await loaded(ROUTE_PACK, "v4_lifetime_routes")
    first = process.process.pid

    loaded.host.add_routes()
    async with TestClient(TestServer(loaded.host.app)) as client:
        died = await client.post("/api/v4_lifetime/crash")
        died = died.status, await died.text()
        again = await client.post("/v4_lifetime/where")
        again = again.status, await again.text()

    assert died == (503, "pack process stopped")
    assert again == (200, str(process.process.pid)) and process.process.pid != first


@pytest.mark.asyncio
async def test_a_pack_comfyui_refuses_after_it_was_loaded_does_not_keep_its_process(
    loaded,
):
    accepted, first = await loaded(ROUTE_PACK, "v4_lifetime_routes")
    refused, second = await loaded(ROUTE_PACK, "v4_same_route")

    assert (accepted, refused) == (True, False)
    assert ended(second.process)
    assert first.process.poll() is None


# The pack process asks the node's schema once when it loads; ComfyUI asks it again before it
# records the pack (`add_pack`) and once more when its loader registers the node.
SCHEMA_FAILS_AT = ROUTE_PACK.replace(
    "    def define_schema(cls):\n",
    "    def define_schema(cls):\n"
    "        os.environ['V4_ASKED'] = str(int(os.getenv('V4_ASKED', '0')) + 1)\n"
    "        if int(os.environ['V4_ASKED']) >= ASK:\n"
    "            raise ValueError('no schema')\n",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ask", [2, 3], ids=["before the pack is recorded", "after the pack is recorded"]
)
async def test_a_pack_whose_loading_fails_after_its_process_started_does_not_keep_it(
    loaded, ask
):
    loaded_ok, process = await loaded(
        SCHEMA_FAILS_AT.replace("ASK", str(ask)), "v4_failing"
    )

    assert loaded_ok is False
    assert ended(process.process)


@pytest.mark.asyncio
async def test_a_flushed_pack_has_its_process_ended_and_its_routes_answer_404(loaded):
    _, process = await loaded(ROUTE_PACK, "v4_lifetime_routes")
    loaded.host.add_routes()

    async with TestClient(TestServer(loaded.host.app)) as client:
        _packs.flush_pack_resources([("v4_lifetime_routes", "v4_lifetime_routes")])
        response = await client.post("/v4_lifetime/where")

    assert ended(process.process)
    assert response.status == 404
    assert "RouteWhere" not in nodes.NODE_CLASS_MAPPINGS


@pytest.mark.asyncio
async def test_a_pack_whose_process_keeps_crashing_is_disabled(loaded, recording):
    _, process = await loaded(LIFETIME_PACK, "v4_crashing")
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    pids = []

    prompt = {"crash": node("Crash"), "keep": keep("crash")}

    for _ in range(_pack_host.CRASHES):
        await work(executor, prompt, ["keep"])
        assert "exit code 3" in outcome(executor)[1]["exception_message"]
        pids.append(process.process.pid)
    await work(executor, prompt, ["keep"])
    event, data = outcome(executor)
    await work(executor, prompt, ["keep"])
    later = outcome(executor)[1]["exception_message"]
    _packs.flush_pack_resources([])

    assert len(set(pids)) == _pack_host.CRASHES and process.process.pid == pids[-1]
    assert (event, data["exception_type"]) == (
        "execution_error",
        "comfy_api.v4._channel.PackStopped",
    )
    disabled = "pack v4_crashing.py is disabled: its process stopped 3 times in 600 s"
    assert disabled in data["exception_message"] and disabled in later
    assert "Crash" not in nodes.NODE_CLASS_MAPPINGS
    assert process.module_path not in _pack_processes.processes


@pytest.mark.asyncio
async def test_a_pack_that_declares_other_nodes_when_it_starts_again_is_disabled(
    loaded, tmp_path
):
    _, process = await loaded(ROUTE_PACK, "v4_changing")
    (tmp_path / "v4_changing.py").write_text(
        ROUTE_PACK.replace("RouteWhere", "Renamed"), encoding="utf-8"
    )
    process.kill()
    seen_stopped(process)
    executor = PromptExecutor(
        RecordingServer(), cache_args={"ram": 0, "ram_inactive": 0}
    )

    await work(executor, {"where": node("RouteWhere")}, ["where"])
    event, data = outcome(executor)
    _packs.flush_pack_resources([])

    assert (event, data["exception_type"]) == (
        "execution_error",
        "comfy_api.v4._channel.PackStopped",
    )
    assert (
        "declared other nodes or routes when its process started again"
        in data["exception_message"]
    )
    assert "RouteWhere" not in nodes.NODE_CLASS_MAPPINGS
    assert (
        ended(process.process) and process.module_path not in _pack_processes.processes
    )


@pytest.mark.asyncio
async def test_a_new_model_file_shows_in_a_pack_nodes_choices_and_validates_without_a_restart(
    isolated, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        copy.deepcopy(folder_paths.folder_names_and_paths),
    )
    listed = tmp_path / "listed"
    listed.mkdir()
    (listed / "a.safetensors").write_bytes(b"")
    folder_paths.add_model_folder_path("v4_listed", str(listed))
    process = await isolated(LISTED_PACK, "v4_listed_pack")
    pid = process.process.pid
    node_cls = nodes.NODE_CLASS_MAPPINGS["Listed"]

    before = node_cls.INPUT_TYPES()["required"]["name"][1]["options"]
    (listed / "b.safetensors").write_bytes(b"")
    after = node_cls.GET_NODE_INFO_V1()["input"]["required"]["name"][1]["options"]
    valid, *_ = await validate_prompt(
        "v4", {"n": node("Listed", name="b.safetensors"), "k": keep("n")}, ["k"]
    )

    assert before == ["a.safetensors"]
    assert after == ["a.safetensors", "b.safetensors"]
    assert valid and process.process.pid == pid


@pytest.mark.asyncio
async def test_a_pack_that_is_busy_or_stopped_never_holds_up_a_schema_request(
    isolated, monkeypatch
):
    monkeypatch.setattr(_pack_host, "SCHEMA_WAIT", 2.0)
    process = await isolated(ROUTE_PACK, "v4_busy_pack")
    node_cls = nodes.NODE_CLASS_MAPPINGS["RouteWhere"]
    expected = node_cls.GET_NODE_INFO_V1()
    asked = []
    call = process.channel.call
    monkeypatch.setattr(
        process.channel, "call", lambda *a, **k: asked.append(a[1]) or call(*a, **k)
    )
    request = {
        "method": "POST",
        "path": "/v4_lifetime/busy",
        "match_info": {"what": "busy"},
        "query": [],
        "headers": {},
        "body": b"60",
        "remote": None,
    }

    # The route keeps the pack's server lane, where schemas are asked, busy for 60 s.
    busy = asyncio.create_task(process.run(("route", 0), (request,), "server"))
    await asyncio.sleep(0)
    started_at = time.monotonic()
    while_busy = [node_cls.GET_NODE_INFO_V1() for _ in range(2)]
    took = time.monotonic() - started_at
    process.kill()
    with pytest.raises(_channel.PackStopped):
        await busy
    while_stopped = node_cls.GET_NODE_INFO_V1()

    assert while_busy == [expected] * 2 and while_stopped == expected
    assert 2.0 <= took < 60
    assert asked == [("schema", "RouteWhere")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "how",
    [
        "nothing",
        "a forked child",
        "a program it started",
        "a program given its descriptors",
    ],
)
async def test_a_pack_that_dies_after_a_large_output_fails_its_node_at_once_whatever_it_left_running(
    isolated, recording, tmp_path, how
):
    """The processes the pack starts may hold its socket after it has died."""
    process = await isolated(LIFETIME_PACK)
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    dies = node("DiesLeaving", how=how, folder=str(tmp_path), after=["large", 0])
    prompt = {"large": node("Large"), "dies": dies, "keep": keep("dies")}

    try:
        # The bound only turns a wait that never ends into a failure.
        await asyncio.wait_for(execute(executor, prompt, ["keep"]), 30)
        event, data = outcome(executor)
        recording.sent.clear()
        await execute(executor, {"where": node("Where"), "k": keep("where")}, ["k"])
    finally:
        end_started(tmp_path)

    assert (event, data["node_id"], data["exception_type"]) == (
        "execution_error",
        "dies",
        "comfy_api.v4._channel.PackStopped",
    )
    assert "exit code 3" in data["exception_message"]
    assert kept(recording) == [process.process.pid]


@pytest.fixture
def probe(tmp_path):
    """A pack process of the probe pack for this test only, which may end it."""
    process = start_probe(tmp_path)
    # A test may point the process at another source; it is listed under the first.
    path = process.module_path
    yield process
    _pack_processes.unload(path)


def seen_stopped(process):
    wait_until(lambda: process.channel.closed, "ComfyUI to see the process end")


DIES_WHILE_READING = """
import os, signal
import comfy_api.v4._channel as channel
read = channel._read
def read_then_die(sock, size):
    if size < (64 << 20):
        return read(sock, size)
    read(sock, 1 << 20)
    os.kill(os.getpid(), signal.SIGKILL)
channel._read = read_then_die
"""


@pytest.mark.parametrize("how", ["nothing", "a forked child", "a program it started"])
@pytest.mark.parametrize("direction", ["to the pack", "from the pack"])
def test_a_pack_killed_in_the_middle_of_a_large_tensor_fails_its_node_once_and_the_next_runs(
    probe, monkeypatch, tmp_path, how, direction
):
    leaves = {
        "nothing": "None",
        "a forked child": "(lambda child: child or (time.sleep(60), os._exit(0)))(os.fork())",
        "a program it started": f"os.system('sleep 60 & echo $! > {tmp_path}/program.pid')",
    }[how]
    child = asyncio.run(evaluate(probe, leaves))
    if how == "a forked child":
        (tmp_path / "child.pid").write_text(str(child))
    pid, size = probe.process.pid, 64 << 20
    if direction == "to the pack":
        asyncio.run(evaluate(probe, "exec(value, globals())", DIES_WHILE_READING))
        expression, value = "value.shape", torch.ones(size // 4)
    else:
        read = _channel._read

        def read_then_kill(sock, wanted, wait):
            if wanted < size:
                return read(sock, wanted, wait)
            first = read(
                sock, 1 << 20, wait
            )  # ComfyUI has the first megabyte of the tensor
            probe.process.kill()
            return memoryview(bytes(first) + read(sock, wanted - len(first), wait))

        monkeypatch.setattr(_channel, "_read", read_then_kill)
        expression, value = f"torch.ones({size // 4})", None

    async def node():
        with CurrentNodeContext("prompt", "large"):
            return await asyncio.wait_for(evaluate(probe, expression, value), 30)

    try:
        with pytest.raises(_channel.PackStopped) as stopped:
            asyncio.run(node())
        seen_stopped(probe)
        crashes = len(probe.crashes)
        next_pid = asyncio.run(evaluate(probe, "os.getpid()"))
    finally:
        end_started(tmp_path)

    assert str(stopped.value).endswith(
        "stopped (exit code -9) while running node large"
    )
    assert crashes == 1 and next_pid not in (pid, None)


def cleans_up(process, lane="exec", **inputs):
    """Run the probe's CleansUp node: it waits, and when cancelled takes `cleanup` seconds."""
    return process.run(("node", "CleansUp", "execute"), (inputs, {}), lane)


async def cancelled_while(coroutine, after=0.3):
    """Run `coroutine` as a node of the prompt; the user cancels `after` seconds in. Returns
    what it ends with."""
    with CurrentNodeContext("prompt", "node"):
        task = asyncio.create_task(coroutine)
        await asyncio.sleep(after)
        interrupt_current_processing(True)
        try:
            return (await asyncio.wait_for(task, 30)).args[0]
        except BaseException as error:  # noqa: B036
            return type(error).__name__


def test_a_pack_that_stops_within_the_grace_period_after_a_cancel_keeps_its_process(
    probe,
):
    pid = probe.process.pid

    ended_with = asyncio.run(cancelled_while(cleans_up(probe)))

    assert ended_with == "InterruptProcessingException"
    assert (probe.process.pid, probe.process.poll()) == (pid, None)


def test_the_users_cancel_leaves_a_route_or_a_validation_to_finish(probe):
    async def run():
        request = asyncio.create_task(cleans_up(probe, "server", seconds=0.5))
        await asyncio.sleep(0.1)
        interrupt_current_processing(True)
        return (await asyncio.wait_for(request, 30)).args[0]

    assert asyncio.run(run()) == 7
    assert processing_interrupted()


@pytest.mark.parametrize("how", ["cancelled", "its loop ended"])
def test_a_call_from_a_node_comfyui_stopped_waiting_for_leaves_the_pack_usable(
    probe, how
):
    pid = probe.process.pid

    async def abandon():
        task = asyncio.create_task(cleans_up(probe, cleanup=0.3, calls=True))
        await asyncio.sleep(0.3)
        if how == "cancelled":
            task.cancel()
            await asyncio.sleep(1.0)

    asyncio.run(abandon())
    # The node's cleanup calls ComfyUI in the request ComfyUI left.
    time.sleep(1.0)

    assert asyncio.run(asyncio.wait_for(evaluate(probe, "os.getpid()"), 30)) == pid


def test_a_pack_killed_after_the_grace_period_is_not_counted_as_a_crash_and_both_are_logged(
    probe, monkeypatch, caplog
):
    monkeypatch.setattr(_pack_host, "GRACE", 0.3)

    with caplog.at_level(logging.WARNING):
        asyncio.run(cancelled_while(evaluate(probe, "time.sleep(30)")))
        seen_stopped(probe)
        interrupt_current_processing(False)
        killed = probe.crashes.copy()
        asyncio.run(evaluate(probe, "1"))
        probe.process.kill()  # then a crash ComfyUI did not cause
        seen_stopped(probe)

    assert (killed, len(probe.crashes)) == ([], 1)
    assert [r.getMessage() for r in caplog.records if probe.peer in r.getMessage()] == [
        f"{probe.peer} had not stopped 0.3 s after the cancel; its process is ended.",
        f"{probe.peer} stopped (exit code -9); it starts again when it is next used.",
    ]


def test_a_pack_waiting_on_comfyui_when_the_user_cancels_is_not_killed(
    probe, monkeypatch
):
    monkeypatch.setattr(_pack_host, "GRACE", 1.0)
    pid = probe.process.pid

    def slow_comfyui_work():
        interrupt_current_processing(True)  # the user cancels meanwhile
        time.sleep(3)  # as a model load does, without checking the cancel
        return "loaded"

    async def node():
        with CurrentNodeContext("prompt", "loads"):
            with contextlib.suppress(InterruptProcessingException):
                await evaluate(probe, "value()", slow_comfyui_work)

    asyncio.run(node())

    assert (probe.process.pid, probe.process.poll()) == (pid, None)


def test_a_cancelled_request_of_a_stopped_process_does_not_kill_the_next_one(
    probe, monkeypatch
):
    monkeypatch.setattr(_pack_host, "GRACE", 0.3)
    first = probe.process
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    release = threading.Event()
    try:

        async def node():
            with CurrentNodeContext("prompt", "busy"):
                return await evaluate(probe, "time.sleep(30)")

        asyncio.run_coroutine_threadsafe(node(), loop)
        while not probe.chains:
            time.sleep(0.01)
        # The prompt thread runs a long node of its own, so the request's record stays.
        busy = threading.Event()
        loop.call_soon_threadsafe(lambda: (busy.set(), release.wait()))
        busy.wait()
        interrupt_current_processing(True)
        first.wait(timeout=30)  # the grace kill
        seen_stopped(probe)
        interrupt_current_processing(False)
        # A route or a validation starts the pack again meanwhile.
        pid = asyncio.run(evaluate(probe, "os.getpid()"))
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(30)

    assert pid == probe.process.pid != first.pid


def test_a_restart_the_server_thread_makes_while_the_user_cancels_leaves_the_cancel_to_the_prompt(
    probe,
):
    probe.kill()
    seen_stopped(probe)
    interrupt_current_processing(True)

    restarted = asyncio.run(evaluate(probe, "time.sleep(1) or 1 + 1"))

    assert restarted == 2 and processing_interrupted()


def test_a_process_that_dies_right_after_answering_its_load_is_started_again(
    probe, monkeypatch
):
    probe.kill()
    seen_stopped(probe)
    call = _pack_host.PackProcess._call

    async def dies_after_loading(self, target, args, lane):
        value = await call(self, target, args, lane)
        if target == ("load",):
            # Before the coroutine that awaited the answer goes on, as when its loop is busy.
            self.process.kill()
            seen_stopped(self)
        return value

    monkeypatch.setattr(_pack_host.PackProcess, "_call", dies_after_loading)
    with pytest.raises(_channel.PackStopped, match="after it loaded"):
        asyncio.run(evaluate(probe, "1"))
    monkeypatch.setattr(_pack_host.PackProcess, "_call", call)

    assert asyncio.run(evaluate(probe, "1 + 1")) == 2


@pytest.mark.parametrize(
    "fails, raised",
    [("os._exit(4)", _channel.PackStopped), ("raise ValueError('no')", ValueError)],
    ids=["its process ends", "its import raises"],
)
def test_a_process_that_fails_while_it_starts_again_is_counted_once(
    probe, tmp_path, fails, raised
):
    crashes = tmp_path / "crash-on-import"
    probe.module_path = str(tmp_path / "v4_crashes_on_import.py")
    (tmp_path / "v4_crashes_on_import.py").write_text(
        f"import os\nif os.path.exists({str(crashes)!r}):\n    {fails}\n" + PROBE,
        encoding="utf-8",
    )
    probe.process.kill()  # one crash ComfyUI did not cause
    seen_stopped(probe)
    crashes.write_text("")
    with pytest.raises(raised):
        asyncio.run(evaluate(probe, "1"))
    crashes.unlink()

    assert len(probe.crashes) == 2
    assert asyncio.run(evaluate(probe, "1 + 1")) == 2


def test_a_late_call_answered_as_the_pack_dies_does_not_end_comfyuis_reader(
    probe, monkeypatch
):
    schedule = probe.schedule

    def dies_first(call):
        probe.process.kill()
        probe.process.wait()
        schedule(call)

    monkeypatch.setattr(probe, "schedule", dies_first)

    async def run():
        waiter = asyncio.create_task(cleans_up(probe, "server"))
        # The node returns; a callback it left calls ComfyUI in its ended request.
        await evaluate(
            probe,
            "__import__('asyncio').get_running_loop().call_later(0.2, value) and None",
            lambda: None,
        )
        return await asyncio.wait_for(waiter, 30)

    with pytest.raises(_channel.PackStopped):
        asyncio.run(run())


SLOW_DECODE = """
import threading, time
import comfy_api.v4._channel as channel
outcome = channel.Channel.outcome
def slow(self, reply):
    if threading.current_thread().name == "pack server lane":
        time.sleep(0.5)
    return outcome(self, reply)
channel.Channel.outcome = slow
"""


def test_an_object_comfyui_hands_back_is_not_released_before_the_pack_has_it(probe):
    """ComfyUI answers with one of the pack's objects and drops it; the release goes with its
    next message, which the pack reads before its server lane has decoded the answer."""
    asyncio.run(evaluate(probe, "exec(value, globals()) or 0", SLOW_DECODE))

    async def run():
        inputs = {"expression": "type(value(make())).__name__", "value": lambda x: x}
        handed_back = asyncio.create_task(
            probe.run(("node", "Eval", "execute"), (inputs, {}), "server")
        )
        await asyncio.sleep(0.2)
        gc.collect()
        await evaluate(probe, "1")
        return (await asyncio.wait_for(handed_back, 30)).args[0]

    assert asyncio.run(run()) == "Thing"


EXITS_WHILE_THE_PACK_HOLDS_THE_GIL = """
import asyncio, sys, threading, time
sys.path.insert(0, {tree!r})
from comfy.cli_args import args
args.cpu = True
args.pack_process = ["v4_stuck"]
import nodes
from comfy_api.v4 import _pack_processes
assert asyncio.run(nodes.load_custom_node({path!r}))
process = _pack_processes.processes[{path!r}]
print(process.process.pid, flush=True)
inputs = {{"expression": "__import__('re').match(r'(a+)+$', 'a' * 40 + 'b')", "value": None}}
threading.Thread(
    target=lambda: asyncio.run(process.run(("node", "Eval", "execute"), (inputs, {{}}))),
    daemon=True,
).start()
time.sleep(1)
print("stuck", flush=True)
if {how!r} == "exits":
    sys.exit(0)
time.sleep(600)
"""


@pytest.mark.parametrize("how", ["exits", "is killed"])
def test_every_pack_process_ends_when_comfyui_exits_even_one_stuck_in_native_code(
    tmp_path, how
):
    path = tmp_path / "v4_stuck.py"
    path.write_text(PROBE, encoding="utf-8")
    tree = os.getcwd()
    script = EXITS_WHILE_THE_PACK_HOLDS_THE_GIL.format(
        tree=tree, path=str(path), how=how
    )
    host = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, cwd=tree
    )
    pack = int(host.stdout.readline())
    host.stdout.readline()
    if how == "is killed":
        host.kill()

    def gone():
        try:
            os.kill(pack, 0)
        except ProcessLookupError:
            return True
        return False

    try:
        host.wait(timeout=60)
        deadline = time.monotonic() + 10
        while not gone() and time.monotonic() < deadline:
            time.sleep(0.05)
        ended = gone()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pack, signal.SIGKILL)

    assert ended


SENDS_A_BAD_MESSAGE = (
    "[__import__('comfy_api.v4._channel').v4._channel.write_frame(s, "
    "b'\\x80\\x05ccomfy_api.latest._io.nothing\\nX\\n.', b'') for s in gc.get_objects() "
    "if type(s).__name__ == 'socket'] and time.sleep(30)"
)


def test_a_message_comfyui_cannot_read_ends_the_pack_process_and_fails_the_wait(probe):
    with pytest.raises(_channel.PackStopped):
        asyncio.run(asyncio.wait_for(evaluate(probe, SENDS_A_BAD_MESSAGE), 30))

    assert ended(probe.process)


def test_a_node_whose_pack_is_unloaded_while_it_runs_fails_saying_so(probe):
    async def run():
        with CurrentNodeContext("prompt", "unloaded"):
            node = asyncio.create_task(evaluate(probe, "time.sleep(30)"))
            await asyncio.sleep(0.3)
            _pack_processes.unload(probe.module_path)
            await asyncio.wait_for(node, 30)

    with pytest.raises(_channel.PackStopped) as stopped:
        asyncio.run(run())

    assert str(stopped.value) == (
        "pack v4_probe_pack.py was unloaded while running node unloaded"
    )


def test_the_answer_to_a_request_comfyui_stopped_waiting_for_is_released_in_the_pack(
    probe,
):
    async def abandon():
        task = asyncio.create_task(cleans_up(probe, cleanup=0.3, answers=True))
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.sleep(
            1.0
        )  # the node answers with an object of the pack meanwhile

    asyncio.run(abandon())

    # Dropping the answer released the object, with the next message.
    assert asyncio.run(evaluate(probe, "alive()")) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["Where", "WhereFingerprinted"])
async def test_after_its_process_stopped_a_packs_node_runs_again_instead_of_coming_from_the_cache(
    isolated, recording, where
):
    process = await isolated(LIFETIME_PACK)
    executor = PromptExecutor(recording, cache_args={"ram": 0, "ram_inactive": 0})
    prompt = {"where": node(where), "keep": keep("where")}
    await execute(executor, prompt, ["keep"])
    first = kept(recording)
    process.process.kill()
    seen_stopped(process)
    recording.sent.clear()

    await execute(executor, prompt, ["keep"])

    assert first != kept(recording) == [process.process.pid]


def test_after_many_kill_cycles_comfyui_has_started_no_process_but_pack_processes(
    probe,
):
    before = {child.pid for child in psutil.Process().children(recursive=True)}
    left_by_packs = []

    for _ in range(6):
        with CurrentNodeContext("prompt", "node"):
            asyncio.run(evaluate(probe, "value * 2", torch.ones(1024, 1024)))
        left_by_packs += psutil.Process(probe.process.pid).children(recursive=True)
        probe.kill()  # as after a cancel the pack ignores
        seen_stopped(probe)
    running = asyncio.run(evaluate(probe, "os.getpid()"))
    started = [
        child.pid
        for child in psutil.Process().children(recursive=True)
        if child.pid not in before
    ]

    assert left_by_packs == []
    assert sorted(started) == sorted([running, probe.process.watchdog.pid])


def holds_the_gil(flag, seconds):
    """A node that marks that it has started, then holds the pack's GIL in native code."""
    return (
        f"(open({flag!r}, 'w').close(), "
        f"__import__('ctypes').PyDLL(None).sleep({seconds}))"
    )


@pytest.mark.parametrize("message", [1024, 4 << 20], ids=["small", "large"])
def test_the_users_cancel_ends_a_pack_stuck_in_native_code_whatever_comfyui_was_sending_it(
    probe, monkeypatch, tmp_path, message
):
    monkeypatch.setattr(_pack_host, "GRACE", 0.5)
    flag = tmp_path / "holding"
    outcome, schema_took = [], []

    async def prompt():
        with CurrentNodeContext("prompt", "stuck"):
            stuck = asyncio.create_task(evaluate(probe, holds_the_gil(str(flag), 120)))
            while not flag.exists():
                await asyncio.sleep(0.01)
            second = asyncio.create_task(evaluate(probe, "len(value)", "x" * message))
            results = await asyncio.gather(stuck, second, return_exceptions=True)
            return [type(result).__name__ for result in results]

    def ask_schema():
        started = time.monotonic()
        probe.schema("Eval")
        schema_took.append(time.monotonic() - started)

    thread = threading.Thread(
        target=lambda: outcome.append(asyncio.run(prompt())), daemon=True
    )
    thread.start()
    wait_until(flag.exists, "the pack to hold its GIL")
    asker = threading.Thread(target=ask_schema, daemon=True)
    asker.start()
    interrupt_current_processing(True)
    thread.join(15)
    asker.join(5)
    ended, schema_answered = not thread.is_alive(), bool(schema_took)
    probe.process.kill()
    thread.join(30)
    asker.join(30)

    assert (ended, schema_answered) == (True, True)
    assert outcome == [["InterruptProcessingException"] * 2]


def test_a_pack_whose_call_waits_for_comfyuis_loop_when_the_user_cancels_is_not_killed(
    probe, monkeypatch
):
    monkeypatch.setattr(_pack_host, "GRACE", 1.0)
    pid = probe.process.pid
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    release = threading.Event()

    def slow_comfyui_work():
        time.sleep(3)  # as a model load does, without checking the cancel
        return "loaded"

    async def node():
        with CurrentNodeContext("prompt", "loads"):
            with contextlib.suppress(InterruptProcessingException):
                return await evaluate(
                    probe, "time.sleep(0.5) or value()", slow_comfyui_work
                )

    try:
        done = asyncio.run_coroutine_threadsafe(node(), loop)
        wait_until(lambda: probe.chains, "the request")
        record = next(iter(probe.chains.values()))
        busy = threading.Event()
        # ComfyUI's prompt loop is busy with other work when the pack's call comes.
        loop.call_soon_threadsafe(lambda: (busy.set(), release.wait()))
        busy.wait()
        wait_until(lambda: record.calls, "the pack's call to be queued")
        interrupt_current_processing(True)
        wait_until(lambda: record.deadline is not None, "the cancel")
        release.set()
        done.result(30)
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(30)

    assert (probe.process.pid, probe.process.poll()) == (pid, None)


@pytest.mark.parametrize(
    "written", ["b'bytes'", "numpy.str_('text')"], ids=["bytes", "a str subclass"]
)
def test_what_a_pack_writes_to_stdout_never_ends_its_process(capfd, probe, written):
    pid = probe.process.pid

    outcome = asyncio.run(
        evaluate(probe, f"caught(lambda: __import__('sys').stdout.write({written}))")
    )
    asyncio.run(evaluate(probe, "1"))
    printed = capfd.readouterr().out

    assert (probe.process.pid, probe.process.poll(), probe.crashes) == (pid, None, [])
    if written == "b'bytes'":
        assert outcome == (
            "builtins",
            "TypeError",
            "write() argument must be str, not bytes",
        )
    else:
        assert outcome is None and "text" in printed


EXITS_WHILE_A_PACK_LOADS = """
import asyncio, sys, threading, time
sys.path.insert(0, {tree!r})
from comfy.cli_args import args
args.cpu = True
args.pack_process = ["v4_stuck_at_import"]
import nodes
from comfy_api.v4 import _pack_launch
launch = _pack_launch.resolve({launcher!r}) if {launcher!r} else _pack_launch.launch
def printing(start):
    process, sock = launch(start)
    print(process.pid, flush=True)
    return process, sock
args.pack_launcher = "__main__:printing"
threading.Thread(
    target=lambda: asyncio.run(nodes.load_custom_node({path!r})), daemon=True
).start()
time.sleep(4)
sys.exit(0)
"""


def test_a_pack_process_still_loading_ends_when_comfyui_exits(tmp_path):
    path = tmp_path / "v4_stuck_at_import.py"
    path.write_text(
        "import ctypes\nctypes.PyDLL(None).sleep(60)\n" + PROBE, encoding="utf-8"
    )
    script = EXITS_WHILE_A_PACK_LOADS.format(
        tree=os.getcwd(), path=str(path), launcher=args.pack_launcher
    )
    host = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    pack = psutil.Process(int(host.stdout.readline()))

    try:
        host.wait(timeout=60)
        gone, _ = psutil.wait_procs([pack], timeout=10)
    finally:
        with contextlib.suppress(psutil.NoSuchProcess):
            pack.kill()

    assert gone == [pack]


SENDS_LATE = """
def send_late(value):
    def call():
        try:
            value(watch("late", Thing()))
        except Exception as error:
            late["error"] = str(error)
    asyncio.get_running_loop().call_later(0.2, call)
late = {}
"""


def test_a_packs_object_in_a_call_comfyui_refuses_is_let_go_in_the_pack(probe):
    asyncio.run(evaluate(probe, "exec(value, globals())", SENDS_LATE))
    asyncio.run(evaluate(probe, "send_late(value)", lambda thing: None))
    wait_until(lambda: asyncio.run(evaluate(probe, "'error' in late")), "the late call")

    assert asyncio.run(evaluate(probe, "late['error']")) == (
        "the request this call belongs to has ended"
    )
    assert asyncio.run(evaluate(probe, "(gc.collect(), 'late' in watched)[1]")) is False


FORKS_WHILE_A_THREAD_PRINTS = """
import gc, os, signal, sys, threading, time
from comfy_api.v4._channel import Channel
def fork_while_printing():
    channel = next(o for o in gc.get_objects() if type(o) is Channel)
    printer = threading.Thread(target=lambda: sys.stdout.write("z" * (64 << 20)))
    printer.start()
    while not channel.send_lock.locked():
        pass
    child = os.fork()
    if child == 0:
        print("the child prints")
        os._exit(0)
    printer.join()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if os.waitpid(child, os.WNOHANG) != (0, 0):
            return "the child ended"
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    os.waitpid(child, 0)
    return "the child hung"
"""


def test_a_child_forked_while_another_thread_prints_can_print(capfd, probe):
    asyncio.run(evaluate(probe, "exec(value, globals())", FORKS_WHILE_A_THREAD_PRINTS))

    outcome = asyncio.run(evaluate(probe, "fork_while_printing()"))
    capfd.readouterr()

    assert outcome == "the child ended"


# The program leaves the pack's process group, as a daemon does, so ComfyUI does not end it.
HOLDS_THE_SOCKET = """
import gc, socket, subprocess
def die_leaving_a_program_holding_the_socket(folder, ask):
    sockets = [s.fileno() for s in gc.get_objects() if isinstance(s, socket.socket)]
    child = subprocess.Popen(["sleep", "60"], pass_fds=sockets, start_new_session=True).pid
    with open(f"{folder}/{child}.pid", "w") as file:
        file.write(str(child))
    threading.Timer(0.5, os._exit, (3,)).start()
    return ask()
"""


def test_an_answer_to_a_pack_that_died_leaving_its_socket_held_does_not_wait_for_ever(
    probe, tmp_path
):
    asyncio.run(evaluate(probe, "exec(value, globals())", HOLDS_THE_SOCKET))

    def ask():
        wait_until(lambda: probe.channel.closed, "ComfyUI to see the pack end")
        return b"x" * (64 << 20)

    outcome = []

    def run():
        try:
            asyncio.run(
                evaluate(
                    probe,
                    "die_leaving_a_program_holding_the_socket(*value)",
                    (str(tmp_path), ask),
                )
            )
        except BaseException as error:
            outcome.append(type(error).__name__)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(20)
    answered = not thread.is_alive()
    end_started(tmp_path)
    thread.join(30)

    assert answered and outcome == ["PackStopped"]


def test_the_answer_to_a_request_whose_loop_has_closed_is_released_in_the_pack(probe):
    async def abandon():
        asyncio.create_task(cleans_up(probe, cleanup=0.3, answers=True))
        # asyncio.run then cancels the task and closes the loop.
        await asyncio.sleep(0.3)

    asyncio.run(abandon())
    wait_until(lambda: not probe.chains, "the late answer")

    assert asyncio.run(evaluate(probe, "alive()")) == 0


def test_an_answer_queued_on_a_loop_that_stopped_leaves_no_record_and_no_kill(
    probe, monkeypatch
):
    monkeypatch.setattr(_pack_host, "GRACE", 1.0)
    pid = probe.process.pid
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    release = threading.Event()

    async def node():
        with CurrentNodeContext("prompt", "abandoned"):
            return await cleans_up(probe, cleanup=0.3)

    try:
        task = asyncio.run_coroutine_threadsafe(node(), loop)
        wait_until(lambda: probe.chains, "the request")
        record = next(iter(probe.chains.values()))
        task.cancel()
        wait_until(lambda: record.abandoned, "the abandonment")
        loop.call_soon_threadsafe(lambda: (release.wait(), loop.stop()))
        wait_until(lambda: record.request not in probe.channel.waiters, "the answer")
        release.set()
        thread.join(30)
        loop.close()
        wait_until(lambda: not probe.chains, "the record to go")
        time.sleep(1.5)  # past the grace period, in which a kill would come
    finally:
        release.set()

    assert (probe.process.pid, probe.process.poll()) == (pid, None)


def test_a_pack_that_goes_on_after_comfyui_answered_its_call_with_the_cancel_is_killed(
    probe, monkeypatch
):
    from comfy.model_management import throw_exception_if_processing_interrupted

    monkeypatch.setattr(_pack_host, "GRACE", 0.5)
    pid = probe.process.pid

    def comfyui_work_that_checks():
        interrupt_current_processing(True)  # the user cancels meanwhile
        time.sleep(0.3)
        throw_exception_if_processing_interrupted()

    async def node():
        with CurrentNodeContext("prompt", "goes_on"):
            return await asyncio.wait_for(
                evaluate(
                    probe,
                    "keep_error(value) or time.sleep(30)",
                    comfyui_work_that_checks,
                ),
                15,
            )

    with pytest.raises(InterruptProcessingException):
        asyncio.run(node())
    probe.process.wait(timeout=10)

    assert probe.process.pid == pid and probe.crashes == []


def test_every_node_waiting_on_a_restart_that_fails_gets_its_error(probe, tmp_path):
    flag = tmp_path / "fail-on-import"
    path = tmp_path / "v4_fails_on_import.py"
    path.write_text(
        f"import os\nif os.path.exists({str(flag)!r}):\n    raise ValueError('no')\n"
        + PROBE,
        encoding="utf-8",
    )
    probe.module_path = str(path)
    probe.process.kill()
    seen_stopped(probe)
    flag.write_text("")

    async def both():
        return await asyncio.gather(
            evaluate(probe, "1"), evaluate(probe, "2"), return_exceptions=True
        )

    outcomes = asyncio.run(both())
    flag.unlink()

    assert [type(outcome).__name__ for outcome in outcomes] == ["ValueError"] * 2


def test_a_validation_is_answered_while_the_pack_runs_a_long_node(probe, tmp_path):
    node_class = _pack_nodes._node_class(
        probe, "v4_probe_pack", "Eval", {"validate_inputs": (["expression"], False)}
    )
    clone = node_class.PREPARE_CLASS_CLONE({"hidden_inputs": {}})
    flag = tmp_path / "running"
    long_node = f"(open({str(flag)!r}, 'w').close(), time.sleep(20))"

    def run_long_node():
        with contextlib.suppress(_channel.PackStopped):
            asyncio.run(evaluate(probe, long_node))

    async def validate():
        with contextlib.suppress(Exception):
            await clone.validate_inputs(expression="1")
        return "answered"

    runner = threading.Thread(target=run_long_node, daemon=True)
    runner.start()
    wait_until(flag.exists, "the long node to run")
    try:
        answered = asyncio.run(asyncio.wait_for(validate(), 10))
    finally:
        probe.process.kill()
        runner.join(30)

    assert answered == "answered"


def test_pack_output_during_a_tensor_transfer_leaves_both_intact(capfd, probe):
    line = "y" * 1023
    printer = f"threading.Thread(target=lambda: [print({line!r}) for _ in range(1024)])"

    tensor = asyncio.run(
        evaluate(probe, f"keep('printer', {printer}).start() or torch.ones(2048, 2048)")
    )
    # The pack sends its output before this answer, and ComfyUI reads in order.
    asyncio.run(evaluate(probe, "kept['printer'].join()"))

    assert torch.equal(tensor, torch.ones(2048, 2048))
    assert capfd.readouterr().out.count(line) == 1024


@pytest.mark.asyncio
async def test_a_route_that_raises_in_a_pack_process_fails_with_its_error_and_keeps_the_process(
    isolated,
):
    process = await isolated(ROUTE_PACK, "v4_raising_route")
    request = {
        "method": "POST",
        "path": "/v4_lifetime/raise",
        "match_info": {"what": "raise"},
        "query": [],
        "headers": {},
        "body": b"",
        "remote": None,
    }

    with pytest.raises(ValueError, match="the route failed"):
        await process.run(("route", 0), (request,), "server")

    assert (process.process.poll(), process.crashes) == (None, [])


def ends_within(pid: int, seconds: float = 10) -> bool:
    """Whether process `pid` ends (or is a zombie, ended but not reaped) within `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize(
    ("how", "own_session"),
    [
        ("the user cancels", False),
        ("it is killed from outside", False),
        ("the pack is unloaded", False),
        ("the user cancels", True),
        ("the pack is unloaded", True),
    ],
    ids=lambda value: {True: "in a session of its own", False: "in its group"}.get(
        value, value
    ),
)
def test_a_program_the_pack_started_ends_when_its_process_ends(
    probe, monkeypatch, tmp_path, how, own_session
):
    monkeypatch.setattr(_pack_host, "GRACE", 0.3)
    started = tmp_path / "program.pid"
    starts = f"start_program(own_session={own_session})"
    runs = f"open({str(started)!r}, 'w').write(str({starts})) and time.sleep(60)"
    outcome = []

    async def node():
        with CurrentNodeContext("prompt", "converts"):
            await evaluate(probe, runs)

    def run():
        try:
            asyncio.run(node())
        except BaseException as error:
            outcome.append(type(error).__name__)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    wait_until(lambda: started.exists() and started.read_text(), "the program to start")
    program = int(started.read_text())
    alive_while_the_node_runs = running(program)
    try:
        if how == "the user cancels":
            interrupt_current_processing(True)
        elif how == "it is killed from outside":
            os.kill(probe.process.pid, signal.SIGKILL)
        else:
            _pack_processes.unload(probe.module_path)
        ended = ends_within(program)
    finally:
        if running(program):
            os.kill(program, signal.SIGKILL)
        thread.join(30)

    assert alive_while_the_node_runs and ended
    assert len(outcome) == 1


def test_a_program_the_pack_started_that_finishes_leaves_its_file_for_a_later_node(
    probe, tmp_path
):
    converted = tmp_path / "converted.txt"
    code = f"open({str(converted)!r}, 'w').write('converted')"

    program = asyncio.run(evaluate(probe, "start_program(value)", code))
    read = asyncio.run(
        evaluate(probe, "programs[-1].wait() or open(value).read()", str(converted))
    )

    assert read == "converted" and not running(program)
    assert probe.process.poll() is None


def test_a_program_the_pack_starts_reads_end_of_file_on_stdin(probe):
    reads = "import sys; print(len(sys.stdin.buffer.read()))"

    read = asyncio.run(
        evaluate(
            probe,
            "subprocess.run([sys.executable, '-c', value], capture_output=True, timeout=30)"
            ".stdout.strip()",
            reads,
        )
    )

    assert read == b"0"


HOST_WITH_A_PROGRAM = """
import asyncio, sys, time
sys.path.insert(0, {tree!r})
from comfy.cli_args import args
args.cpu = True
args.pack_process = ["v4_programs"]
args.pack_launcher = {launcher!r}
import nodes
from comfy_api.v4 import _pack_processes
assert asyncio.run(nodes.load_custom_node({path!r}))
process = _pack_processes.processes[{path!r}]
inputs = {{"expression": "start_program()", "value": None}}
program = asyncio.run(process.run(("node", "Eval", "execute"), (inputs, {{}}))).args[0]
print(process.process.pid, program, flush=True)
if {how!r} == "exits":
    sys.exit(0)
# Short sleeps: a Ctrl+C the system hands to another thread is acted on only between them.
for _ in range(6000):
    time.sleep(0.1)
"""


@pytest.mark.parametrize("how", ["exits", "is killed", "gets Ctrl+C"])
def test_when_comfyui_ends_every_pack_process_and_program_it_started_ends(
    tmp_path, how
):
    path = tmp_path / "v4_programs.py"
    path.write_text(PROBE, encoding="utf-8")
    script = HOST_WITH_A_PROGRAM.format(
        tree=os.getcwd(), path=str(path), how=how, launcher=args.pack_launcher
    )
    # A session of its own, as a terminal's foreground job: Ctrl+C goes to its whole group.
    host = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    pack, program = map(int, host.stdout.readline().split())
    try:
        if how == "is killed":
            host.kill()
        elif how == "gets Ctrl+C":
            os.killpg(host.pid, signal.SIGINT)
        host.wait(timeout=60)
        ended = ends_within(pack) and ends_within(program)
    finally:
        for pid in (pack, program):
            if running(pid):
                os.kill(pid, signal.SIGKILL)
        if host.poll() is None:
            host.kill()

    assert ended


def test_a_cancel_that_could_not_be_sent_at_once_reaches_the_pack_later(
    probe, monkeypatch
):
    monkeypatch.setattr(_pack_host, "GRACE", 2.0)
    pid = probe.process.pid
    outcome = []

    async def node():
        with CurrentNodeContext("prompt", "waits"):
            await cleans_up(probe, cleanup=0.1)

    def run():
        try:
            asyncio.run(node())
        except BaseException as error:
            outcome.append(type(error).__name__)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    wait_until(lambda: probe.chains, "the request")
    record = next(iter(probe.chains.values()))
    # Another thread is sending when the user cancels.
    with probe.channel.send_lock:
        interrupt_current_processing(True)
        wait_until(lambda: record.deadline is not None, "the cancel")
        told_at_once = record.told
    thread.join(30)

    assert told_at_once is False
    assert outcome == ["InterruptProcessingException"]
    assert (probe.process.pid, probe.process.poll()) == (pid, None)
