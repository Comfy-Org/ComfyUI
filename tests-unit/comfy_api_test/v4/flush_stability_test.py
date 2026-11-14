import asyncio
import contextlib
import sys
import threading

import pytest

import nodes
from comfy_api.v4 import _packs
from execution import PromptExecutor, PromptQueue, validate_prompt

PACK = """\
import asyncio

from comfy_api.v4 import ComfyExtension, io

STARTED = asyncio.Event()
RELEASE = asyncio.Event()


class Wait(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="{tag}Wait", outputs=[io.Int.Output()])

    @classmethod
    async def execute(cls):
        STARTED.set()
        await RELEASE.wait()
        return io.NodeOutput(1)


class Show(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="{tag}Show", inputs=[io.Int.Input("value")], is_output_node=True
        )

    @classmethod
    def execute(cls, value):
        return io.NodeOutput(ui={{"value": [value]}})


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Wait, Show]


async def comfy_entrypoint():
    return Extension()
"""


class HeadlessServer:
    client_id = None
    last_node_id = None

    def __init__(self):
        self.sockets_metadata = {}

    def send_sync(self, event, data, sid=None):
        pass


@pytest.fixture
def load(tmp_path, monkeypatch):
    async def load(tag, folder=None):
        """Load a pack whose node ids start with `tag`, from the folder `folder` (or `tag`)."""
        folder = tmp_path / (folder or tag)
        folder.mkdir()
        (folder / "__init__.py").write_text(PACK.format(tag=tag), encoding="utf-8")
        module_name = str(folder).replace(".", "_x_")
        monkeypatch.setitem(sys.modules, module_name, None)
        assert await nodes.load_custom_node(str(folder))
        return sys.modules[module_name]

    return load


async def queue(host, tag):
    """Validate and queue a two-node prompt of pack `tag`, as the server's /prompt does."""
    prompt = {
        "wait": {"class_type": f"{tag}Wait", "inputs": {}},
        "show": {"class_type": f"{tag}Show", "inputs": {"value": ["wait", 0]}},
    }
    valid, error, outputs, node_errors = await validate_prompt(tag, prompt, ["show"])
    assert valid, (error, node_errors)
    host.prompt_queue.put((0, tag, prompt, {}, outputs, {}))


async def work(host):
    """What ComfyUI's prompt worker does with the next queued prompt (`main.prompt_worker`)."""
    item, item_id = host.prompt_queue.get(timeout=0)
    executor = PromptExecutor(
        HeadlessServer(), cache_args={"ram": 0, "ram_inactive": 0}
    )
    await executor.execute_async(item[2], item[1], item[3], item[4])
    status = PromptQueue.ExecutionStatus(
        "success" if executor.success else "error", executor.success, []
    )
    host.prompt_queue.task_done(item_id, executor.history_result, status=status)
    return executor


async def unknown(node_id):
    prompt = {"1": {"class_type": node_id, "inputs": {}}}
    return (await validate_prompt("check", prompt, ["1"]))[1][
        "type"
    ] == "missing_node_type"


@pytest.mark.asyncio
async def test_a_flush_while_its_node_runs_lets_the_prompt_finish_and_removes_its_nodes_after(
    host, load
):
    module = await load("alpha")
    await queue(host, "alpha")
    running = asyncio.create_task(work(host))
    await asyncio.wait_for(module.STARTED.wait(), 10)

    _packs.flush_pack_resources([("alpha", "alpha")])
    assert "alphaShow" in nodes.NODE_CLASS_MAPPINGS
    module.RELEASE.set()
    executor = await asyncio.wait_for(running, 10)

    assert executor.success, executor.status_messages
    assert executor.history_result["outputs"]["show"] == {"value": [1]}
    _packs.flush_pack_resources([])
    assert await unknown("alphaWait") and await unknown("alphaShow")


@pytest.mark.asyncio
async def test_a_queued_prompt_keeps_a_flushed_packs_nodes_until_it_has_run(host, load):
    module = await load("alpha")
    await queue(host, "alpha")

    _packs.flush_all_pack_resources()
    assert "alphaWait" in nodes.NODE_CLASS_MAPPINGS
    module.RELEASE.set()
    executor = await work(host)

    assert executor.success, executor.status_messages
    _packs.flush_all_pack_resources()
    assert await unknown("alphaWait")


@pytest.mark.asyncio
async def test_flushing_twice_or_flushing_an_unknown_pack_is_harmless(host, load):
    await load("alpha")
    beta = await load("beta")

    _packs.flush_pack_resources([("alpha", "alpha")])
    _packs.flush_pack_resources([("alpha", "alpha")])
    _packs.flush_pack_resources([("not-loaded", "not-loaded")])

    assert await unknown("alphaWait")
    beta.RELEASE.set()
    await queue(host, "beta")
    assert (await work(host)).success
    _packs.flush_all_pack_resources()
    _packs.flush_all_pack_resources()
    assert await unknown("betaWait")


@pytest.mark.asyncio
async def test_a_node_a_flushed_pack_replaced_stays_gone_when_the_pack_it_replaced_is_flushed(
    host, load
):
    await load("alpha", folder="first")
    second = await load("alpha", folder="second")
    await queue(host, "alpha")

    _packs.flush_pack_resources([("second", "second")])
    _packs.flush_pack_resources([("first", "first")])
    second.RELEASE.set()
    assert (await work(host)).success
    _packs.flush_pack_resources([])

    assert await unknown("alphaWait") and await unknown("alphaShow")


def test_two_threads_flushing_at_once_both_succeed(monkeypatch):
    """As the prompt thread disabling a pack and the host flushing every pack at once."""
    packs = [_packs._Pack(f"p{i}", f"p{i}", f"/nowhere/p{i}", []) for i in range(2)]
    monkeypatch.setattr(_packs, "_packs", {(p.name, p.folder): p for p in packs})
    monkeypatch.setattr(_packs, "_give_back_folders", lambda *args: None)
    # Each flush checks whether a prompt uses its pack's nodes while the other runs.
    both_checking = threading.Barrier(2, timeout=2)

    def in_use(node_classes):
        with contextlib.suppress(threading.BrokenBarrierError):
            both_checking.wait()
        return False

    monkeypatch.setattr(_packs, "_in_use", in_use)
    errors = []

    def flush(selected):
        try:
            _packs._flush(selected)
        except Exception as error:
            errors.append(repr(error))

    threads = [
        threading.Thread(target=flush, args=([packs[0]],)),
        threading.Thread(target=flush, args=(list(packs),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert errors == [] and _packs._packs == {} and _packs._retired == []
