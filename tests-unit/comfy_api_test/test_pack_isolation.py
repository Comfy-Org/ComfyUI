"""COMFY_PACK_ISOLATION=true: a pack's engine-API nodes run in the pack's own process."""
import asyncio
import os
import socket
import textwrap

import pytest

from comfy_api.latest import _engine, _runtime
from comfy_api.latest import pack_isolation

PACK = '''
import os
from comfy_api.v4 import eng, io


class OOPDouble(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPDouble", inputs=[io.Image.Input("image")],
                         outputs=[io.Image.Output(), io.Int.Output()])

    @classmethod
    async def execute(cls, image):
        doubled = await eng.ext.oop_tool.scale(image, 2.0)
        return io.NodeOutput(doubled, os.getpid())


class OOPReport(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPReport", inputs=[io.Image.Input("image"), io.Int.Input("pid")],
                         outputs=[], is_output_node=True)

    @classmethod
    async def execute(cls, image, pid):
        return io.NodeOutput(ui={"pid": [pid], "here": [os.getpid()], "sum": [float(image.sum())],
                                 "unique_id": [cls.hidden.unique_id]})


class OOPObject(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPObject", inputs=[io.Image.Input("image")], outputs=[io.Image.Output()])

    @classmethod
    async def execute(cls, image):
        return io.NodeOutput(object())


class OOPExit(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPExit", inputs=[io.Image.Input("image")], outputs=[io.Image.Output()])

    @classmethod
    async def execute(cls, image):
        os._exit(3)


class OOPLegacy:
    RETURN_TYPES = ("INT",)
    FUNCTION = "go"
    CATEGORY = "test"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def go(self, image):
        return {"ui": {"legacy_pid": [os.getpid()]}, "result": (os.getpid(),)}


NODE_CLASS_MAPPINGS = {"OOPDouble": OOPDouble, "OOPReport": OOPReport, "OOPObject": OOPObject,
                       "OOPExit": OOPExit, "OOPLegacy": OOPLegacy}
'''

TOOL = b'''
class EngineOps:
    def scale(self, image, factor):
        return image * factor
'''


@pytest.fixture
def oop(tmp_path, monkeypatch):
    import nodes

    root = tmp_path / "custom_nodes" / "oop_pack"
    (root / "v2" / "engine").mkdir(parents=True)
    (root / "__init__.py").write_text(textwrap.dedent(PACK))
    (root / "v2" / "engine" / "oop_tool.py").write_bytes(TOOL)
    monkeypatch.setenv(_engine.DEV_EXTENSIONS_ENV, "1")
    monkeypatch.setenv(pack_isolation.ENV, "true")
    monkeypatch.setattr(_engine, "_extensions", {})
    monkeypatch.setattr(_engine, "_pack_extensions", {})
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    saved = _runtime.providers.execution_backend
    before = dict(nodes.NODE_CLASS_MAPPINGS)
    assert asyncio.run(nodes.load_custom_node(str(root)))
    assert _runtime.load_execution_provider() is True
    backend = _runtime.providers.execution_backend
    assert isinstance(backend, pack_isolation.PackIsolationBackend)
    yield backend
    backend.stop()
    _runtime.providers.execution_backend = saved
    nodes.NODE_CLASS_MAPPINGS.clear()
    nodes.NODE_CLASS_MAPPINGS.update(before)
    nodes.LOADED_MODULE_DIRS.pop("oop_pack", None)


class _Server:
    client_id = None
    last_node_id = None
    last_prompt_id = None

    def send_sync(self, *args, **kwargs):
        pass


def _run(prompt, outputs):
    import execution

    executor = execution.PromptExecutor(_Server(), cache_type=execution.CacheType.NONE,
                                        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    executor.execute(prompt, "oop-test", {}, outputs)
    return executor


IMAGE = {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0x010101}}


def test_engine_api_nodes_run_in_the_pack_worker(oop):
    executor = _run({
        "1": IMAGE,
        "2": {"class_type": "OOPDouble", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "OOPReport", "inputs": {"image": ["2", 0], "pid": ["2", 1]}},
    }, ["3"])
    assert executor.success, executor.status_messages
    ui = executor.history_result["outputs"]["3"]
    assert ui["pid"][0] == ui["here"][0] != os.getpid()  # both nodes ran in the same worker
    assert ui["sum"][0] == pytest.approx(8 * 8 * 3 * (1 / 255) * 2, rel=1e-5)  # the extension ran
    assert ui["unique_id"] == ["3"]
    assert len(oop.workers) == 1  # one worker for the pack


def test_extensions_still_load_in_comfyui(oop):
    assert any(stem == "oop_tool" for files in _engine._pack_extensions.values() for stem in files)


def test_nodes_without_engine_api_stay_in_comfyui(oop):
    executor = _run({"1": IMAGE, "2": {"class_type": "OOPLegacy", "inputs": {"image": ["1", 0]}}}, ["2"])
    assert executor.success, executor.status_messages
    assert executor.history_result["outputs"]["2"]["legacy_pid"] == [os.getpid()]
    assert not oop.workers


def test_an_object_a_pack_returns_stays_in_its_process(oop):
    # OOPObject returns object(); it stays in the pack's process, and the pack's next node gets that
    # very object (and fails using it as an image), not a copy or a handle.
    executor = _run({"1": IMAGE, "2": {"class_type": "OOPObject", "inputs": {"image": ["1", 0]}},
                     "3": {"class_type": "OOPReport", "inputs": {"image": ["2", 0], "pid": 0}}}, ["3"])
    assert not executor.success
    assert "'object' object has no attribute 'sum'" in str(executor.status_messages)


def test_a_worker_that_dies_is_replaced(oop):
    executor = _run({"1": IMAGE, "2": {"class_type": "OOPExit", "inputs": {"image": ["1", 0]}},
                     "3": {"class_type": "OOPReport", "inputs": {"image": ["2", 0], "pid": 0}}}, ["3"])
    assert not executor.success
    assert "WorkerError" in str(executor.status_messages)
    (dead,) = oop.workers.values()
    assert not dead.alive()
    executor = _run({"1": IMAGE, "2": {"class_type": "OOPDouble", "inputs": {"image": ["1", 0]}},
                     "3": {"class_type": "OOPReport", "inputs": {"image": ["2", 0], "pid": ["2", 1]}}}, ["3"])
    assert executor.success, executor.status_messages
    assert list(oop.workers.values()) != [dead]


def test_a_lost_connection_is_a_worker_error():
    ours, theirs = socket.socketpair()
    theirs.close()
    try:
        with pytest.raises(pack_isolation.WorkerError):
            pack_isolation.Channel(ours, lambda message: None).request({"op": "hello"})
    finally:
        ours.close()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(pack_isolation.ENV, raising=False)
    monkeypatch.delenv(pack_isolation.LEGACY_ENV, raising=False)
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    assert _runtime.load_execution_provider() is False
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(pack_isolation.ENV, value)
        assert pack_isolation.enabled()
    monkeypatch.setenv(pack_isolation.ENV, "0")
    assert not pack_isolation.enabled()


INIT_PACK = """
INIT_RAN = True
from ._v2_nodes import NODE_CLASS_MAPPINGS
"""

INIT_NODES = """
import sys
from comfy_api.v4 import io

_INIT_RAN = bool(getattr(sys.modules[__package__], "INIT_RAN", False))


class OOPInit(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPInit", inputs=[io.Image.Input("image")], outputs=[], is_output_node=True)

    @classmethod
    async def execute(cls, image):
        return io.NodeOutput(ui={"init_ran": [_INIT_RAN]})


NODE_CLASS_MAPPINGS = {"OOPInit": OOPInit}
"""


def test_the_worker_runs_the_packs_init(oop, tmp_path):
    """A generated v2 pack's node module reads its package's state at import; the worker imports the package first."""
    import nodes

    root = tmp_path / "custom_nodes" / "init_pack" / "v2"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(textwrap.dedent(INIT_PACK))
    (root / "_v2_nodes.py").write_text(textwrap.dedent(INIT_NODES))
    assert asyncio.run(nodes.load_custom_node(str(root), module_name="init_pack"))
    try:
        executor = _run({"1": IMAGE, "2": {"class_type": "OOPInit", "inputs": {"image": ["1", 0]}}}, ["2"])
        assert executor.success, executor.status_messages
        assert executor.history_result["outputs"]["2"]["init_ran"] == [True]
    finally:
        nodes.NODE_CLASS_MAPPINGS.pop("OOPInit", None)
        nodes.LOADED_MODULE_DIRS.pop("init_pack", None)


FOLDERS_PACK = """
from comfy_api.v4 import io, util


class OOPFolders(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="OOPFolders", inputs=[io.Image.Input("image")], outputs=[], is_output_node=True)

    @classmethod
    async def execute(cls, image):
        return io.NodeOutput(ui={"output": [util.folders.get_output_directory()],
                                 "temp": [util.folders.get_temp_directory()]})


NODE_CLASS_MAPPINGS = {"OOPFolders": OOPFolders}
"""


def test_the_worker_uses_comfyuis_folders(oop, tmp_path, monkeypatch):
    """ComfyUI's --output-directory and the rest reach the worker's folder_paths, so util.folders agrees."""
    import folder_paths
    import nodes

    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path / "out"))
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path / "tmp"))
    root = tmp_path / "custom_nodes" / "folders_pack"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text(textwrap.dedent(FOLDERS_PACK))
    assert asyncio.run(nodes.load_custom_node(str(root)))
    try:
        executor = _run({"1": IMAGE, "2": {"class_type": "OOPFolders", "inputs": {"image": ["1", 0]}}}, ["2"])
        assert executor.success, executor.status_messages
        ui = executor.history_result["outputs"]["2"]
        assert ui["output"] == [str(tmp_path / "out")] and ui["temp"] == [str(tmp_path / "tmp")]
    finally:
        nodes.NODE_CLASS_MAPPINGS.pop("OOPFolders", None)
        nodes.LOADED_MODULE_DIRS.pop("folders_pack", None)
