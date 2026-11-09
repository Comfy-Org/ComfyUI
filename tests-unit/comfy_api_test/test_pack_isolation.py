"""OOP mode as shipped: ComfyUI imports no pack. Packs start when a prompt or a route needs them,
their nodes register then, idle packs stop after a TTL, and a launcher can be dropped in."""
import asyncio
import json
import os
import pathlib
import stat
import sys
import textwrap
import time
import types

import pytest

from comfy_api._impl import _engine, _runtime
from comfy_api._impl import isolation as oop
from comfy_api._impl.isolation import launcher, packs, server

V3_PACK = '''
import os
from aiohttp import web
from server import PromptServer
from comfy_api.index import ComfyExtension, eng, io


class PackDouble(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="PackDouble", category="test",
                         inputs=[io.Image.Input("image"),
                                 io.Combo.Input("lora", options=io.ModelFolder("loras"), optional=True)],
                         outputs=[io.Image.Output(), io.Int.Output()])

    @classmethod
    async def execute(cls, image, lora=None):
        return io.NodeOutput(await eng.ext.pack_tool.scale(image, 2.0), os.getpid())


class PackReport(io.ComfyNode):
    ENGINE_API = True

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="PackReport", category="test", is_output_node=True,
                         inputs=[io.Image.Input("image"), io.Int.Input("pid")], outputs=[])

    @classmethod
    def execute(cls, image, pid):
        return io.NodeOutput(ui={"pid": [pid], "here": [os.getpid()], "sum": [float(image.sum())],
                                 "unique_id": [cls.hidden.unique_id]})


@PromptServer.instance.routes.get("/pack_demo/hello")
async def hello(request):
    return web.json_response({"hello": request.query.get("name"), "pid": os.getpid()})


@PromptServer.instance.routes.post("/pack_demo/echo")
async def echo(request):
    return web.json_response({"got": await request.json(), "pid": os.getpid()})


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [PackDouble, PackReport]


async def comfy_entrypoint():
    return Extension()

WEB_DIRECTORY = "./web"
'''

TOOL = b'''
class EngineOps:
    def scale(self, image, factor):
        return image * factor
'''

V1_PACK = '''
import os


class AnyType(str):
    def __ne__(self, other):
        return False


class V1Invert:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "anything": (AnyType("*"), {})}}
    RETURN_TYPES = ("IMAGE", AnyType("*"))
    FUNCTION = "run"
    CATEGORY = "test"
    OUTPUT_NODE = True

    def run(self, image, anything):
        return {"ui": {"pid": [os.getpid()]}, "result": (1.0 - image, anything)}


NODE_CLASS_MAPPINGS = {"V1Invert": V1Invert}
NODE_DISPLAY_NAME_MAPPINGS = {"V1Invert": "Invert (V1)"}
'''


DECLARED_ROUTES = """
import json
import os


def hello(request):
    return {"body": json.dumps({"hello": request["query"].get("name"), "path": request["path"], "pid": os.getpid()})}


async def notes(request):
    return {"status": 201, "body": "saved " + request["body"], "content_type": "text/plain"}
"""


def _write_corpus(root: pathlib.Path) -> pathlib.Path:
    """A pack corpus: packs.json + <slug>/<key>/<Pack>-HEAD/, the V3 pack's code in v2/."""
    corpus = root / "packs"
    v3 = corpus / "v3-pack" / "x1111111" / "V3Pack-HEAD"
    (v3 / "v2" / "v2" / "engine").mkdir(parents=True)
    (v3 / "__init__.py").write_text("raise ImportError('the pristine pack must not be loaded')\n")
    (v3 / "v2" / "__init__.py").write_text(textwrap.dedent(V3_PACK))
    (v3 / "v2" / "v2" / "engine" / "pack_tool.py").write_bytes(TOOL)
    (v3 / "v2" / "_routes.py").write_text(DECLARED_ROUTES)
    (v3 / "v2" / "secure-nodes.json").write_text(json.dumps({"routes": [
        {"method": "get", "path": "/hello", "module": "_routes", "handler": "hello"},
        {"method": "post", "path": "/notes", "module": "_routes", "handler": "notes"}]}))
    (v3 / "v2" / "web").mkdir()
    (v3 / "v2" / "web" / "pack.js").write_text("// pack js\n")
    v1 = corpus / "v1-pack" / "x2222222" / "V1Pack-HEAD"
    v1.mkdir(parents=True)
    (v1 / "__init__.py").write_text(textwrap.dedent(V1_PACK))
    (corpus / "packs.json").write_text(json.dumps({
        "v3-pack": {"key": "x1111111", "pack": "V3Pack-HEAD"},
        "v1-pack": {"key": "x2222222", "pack": "V1Pack-HEAD"}}))
    return root


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return _write_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture
def mode(corpus, tmp_path, monkeypatch):
    """OOP mode over the corpus. Yields the backend; stops every pack server after."""
    import nodes

    monkeypatch.setenv(oop.ENV, "true")
    monkeypatch.setenv(_engine.DEV_EXTENSIONS_ENV, "1")
    monkeypatch.setenv(server.RUN_DIR_ENV, str(tmp_path / "run"))
    monkeypatch.delenv(_runtime.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(corpus / "index-cache"))
    os.makedirs(corpus / "index-cache", exist_ok=True)
    monkeypatch.setattr(_engine, "_extensions", {})
    monkeypatch.setattr(_engine, "_pack_extensions", {})
    saved = _runtime.providers.execution_backend
    before = dict(nodes.NODE_CLASS_MAPPINGS)
    web_dirs = dict(nodes.EXTENSION_WEB_DIRS)
    assert _runtime.load_execution_provider() is True
    backend = _runtime.providers.execution_backend
    yield backend
    backend.stop()
    _runtime.providers.execution_backend = saved
    nodes.NODE_CLASS_MAPPINGS.clear()
    nodes.NODE_CLASS_MAPPINGS.update(before)
    nodes.EXTENSION_WEB_DIRS.clear()
    nodes.EXTENSION_WEB_DIRS.update(web_dirs)


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
    executor.execute(prompt, "oop-packs-test", {}, outputs)
    return executor


IMAGE = {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0x010101}}
PROMPT = {
    "1": IMAGE,
    "2": {"class_type": "PackDouble", "inputs": {"image": ["1", 0]}},
    "3": {"class_type": "PackReport", "inputs": {"image": ["2", 0], "pid": ["2", 1]}},
    "4": {"class_type": "V1Invert", "inputs": {"image": ["2", 0], "anything": 5}},
}


# --------------------------------------------------------------------------- #
# Finding packs
# --------------------------------------------------------------------------- #
def test_a_corpus_is_read_from_packs_json_and_loads_conversions(corpus):
    found = {p.name: p for p in packs.find_packs([str(corpus)])}
    assert sorted(found) == ["V1Pack", "V3Pack"]
    assert found["V3Pack"].entry.endswith(os.path.join("V3Pack-HEAD", "v2"))  # its conversion
    assert found["V1Pack"].entry.endswith("V1Pack-HEAD")  # unconverted: the pack itself


def test_a_flat_folder_is_a_custom_nodes_folder(tmp_path):
    (tmp_path / "plain" / "v2" / "engine").mkdir(parents=True)  # extensions only: not a conversion
    (tmp_path / "plain" / "__init__.py").write_text("")
    (tmp_path / "converted" / "v2").mkdir(parents=True)
    (tmp_path / "converted" / "__init__.py").write_text("")
    (tmp_path / "converted" / "v2" / "__init__.py").write_text("")
    (tmp_path / "off.disabled").mkdir()
    (tmp_path / "off.disabled" / "__init__.py").write_text("")
    (tmp_path / "single.py").write_text("")
    found = {p.name: p.entry for p in packs.find_packs([str(tmp_path)])}
    assert found == {"converted": os.path.realpath(tmp_path / "converted" / "v2"),
                     "plain": os.path.realpath(tmp_path / "plain")}


def test_descriptions_are_cached_per_version_of_a_pack(corpus, monkeypatch):
    monkeypatch.setattr(packs, "_cache_dir", lambda: str(corpus / "index-cache"))
    os.makedirs(corpus / "index-cache", exist_ok=True)
    pack = {p.name: p for p in packs.find_packs([str(corpus)])}["V1Pack"]
    first = packs.describe(pack)
    assert first["nodes"] == {"V1Invert": "Invert (V1)"}
    monkeypatch.setattr(server.PackServer, "describe", lambda entry: pytest.fail("not cached"))
    assert packs.describe(pack) == first


# --------------------------------------------------------------------------- #
# Nodes register before rendering, and run in the pack's process
# --------------------------------------------------------------------------- #
def test_no_pack_is_imported_or_started_until_a_prompt_needs_it(mode, corpus):
    import nodes

    mode.setup_packs([str(corpus)])
    assert "PackDouble" not in nodes.NODE_CLASS_MAPPINGS and "V1Invert" not in nodes.NODE_CLASS_MAPPINGS
    assert not mode.packs.started
    assert not any("V3Pack" in name for name in sys.modules)
    assert nodes.EXTENSION_WEB_DIRS["V3Pack"].endswith(os.path.join("v2", "web"))  # served from disk


def test_a_prompt_starts_its_packs_registers_their_nodes_and_runs_them_there(mode, corpus):
    import nodes

    mode.setup_packs([str(corpus)])
    executor = _run(PROMPT, ["3", "4"])
    assert executor.success, executor.status_messages
    ui = executor.history_result["outputs"]
    assert ui["3"]["pid"][0] == ui["3"]["here"][0] != os.getpid()  # in the V3 pack's process
    assert ui["3"]["sum"][0] == pytest.approx(8 * 8 * 3 * (1 / 255) * 2, rel=1e-5)  # its extension ran here
    assert ui["3"]["unique_id"] == ["3"]
    assert ui["4"]["pid"][0] not in (os.getpid(), ui["3"]["pid"][0])  # the V1 pack has its own process
    assert {"PackDouble", "PackReport", "V1Invert"} <= set(nodes.NODE_CLASS_MAPPINGS)
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["V1Invert"] == "Invert (V1)"
    assert sorted(mode.packs.started) == ["V1Pack", "V3Pack"]


def test_a_v3_node_is_registered_with_its_own_schema(mode, corpus, monkeypatch):
    import folder_paths
    import nodes

    mode.setup_packs([str(corpus)])
    asyncio.run(mode.ensure_node_types({"PackDouble"}))
    proxy = nodes.NODE_CLASS_MAPPINGS["PackDouble"]
    info = proxy.GET_NODE_INFO_V1()
    assert info["category"] == "test" and info["output"] == ["IMAGE", "INT"]
    assert info["python_module"] == "custom_nodes.V3Pack"
    # io.ModelFolder options are listed by ComfyUI, each time, not frozen by the pack
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda folder: ["new.safetensors"] if folder == "loras" else [])
    assert proxy.GET_NODE_INFO_V1()["input"]["optional"]["lora"][1]["options"] == ["new.safetensors"]


def test_a_v1_nodes_declarations_cross_as_plain_data(mode, corpus):
    import nodes

    mode.setup_packs([str(corpus)])
    asyncio.run(mode.ensure_node_types({"V1Invert"}))
    proxy = nodes.NODE_CLASS_MAPPINGS["V1Invert"]
    assert proxy.RETURN_TYPES == ("IMAGE", "*") and type(proxy.RETURN_TYPES[1]) is str
    assert proxy.INPUT_TYPES()["required"]["anything"][0] == "*"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def test_a_route_starts_its_pack_and_is_forwarded(mode, corpus, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    stand_in = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=stand_in)))
    mode.setup_packs([str(corpus)])
    assert not mode.packs.started

    async def go():
        app = web.Application()
        app.add_routes(stand_in.routes)
        api = web.RouteTableDef()  # ComfyUI also serves every custom route under /api
        for route in stand_in.routes:
            api.route(route.method, "/api" + route.path)(route.handler)
        app.add_routes(api)
        async with TestClient(TestServer(app)) as client:
            hello = await (await client.get("/pack_demo/hello?name=ben")).json()
            via_api = await (await client.get("/api/pack_demo/hello?name=api")).json()
            echo = await (await client.post("/pack_demo/echo", json={"a": [1, 2]})).json()
        return hello, via_api, echo
    hello, via_api, echo = asyncio.run(go())
    assert hello["hello"] == "ben" and hello["pid"] != os.getpid()
    assert via_api == {"hello": "api", "pid": hello["pid"]}
    assert echo == {"got": {"a": [1, 2]}, "pid": hello["pid"]}
    assert sorted(mode.packs.started) == ["V3Pack"]  # only the pack whose route was called


def test_v2_routes_live_under_the_packs_namespace(mode, corpus, monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    stand_in = types.SimpleNamespace(routes=web.RouteTableDef(), app=web.Application())
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=stand_in)))
    mode.setup_packs([str(corpus)])
    namespace = packs.pack_namespace("v3-pack")  # the corpus slug; the same derivation as the overlay's
    assert namespace == "v3-pack-8cb288b6" and mode.packs.namespaces[namespace] == "V3Pack"

    async def go():
        app = web.Application()
        app.add_routes(stand_in.routes)
        async with TestClient(TestServer(app)) as client:
            hello = await client.get(f"/ext/{namespace}/hello?name=ben")
            notes = await client.post(f"/ext/{namespace}/notes", data="a note")
            missing = await client.get(f"/ext/{namespace}/nope")
            other = await client.get("/ext/not-a-pack-00000000/hello")
            return (await hello.json(), notes.status, await notes.text(), notes.content_type,
                    missing.status, other.status)
    hello, status, text, content_type, missing, other = asyncio.run(go())
    assert hello["hello"] == "ben" and hello["path"] == "/hello" and hello["pid"] != os.getpid()
    assert (status, text, content_type) == (201, "saved a note", "text/plain")
    assert (missing, other) == (404, 404)
    assert sorted(mode.packs.started) == ["V3Pack"]


# --------------------------------------------------------------------------- #
# TTL
# --------------------------------------------------------------------------- #
def _wait(condition, seconds=30.0):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def test_an_unused_pack_stops_and_starts_again_when_needed(mode, corpus, monkeypatch):
    monkeypatch.setenv(packs.TTL_ENV, str(2 / 60))  # two seconds
    mode.setup_packs([str(corpus)])
    first = _run(PROMPT, ["3"])
    assert first.success, first.status_messages
    pid = first.history_result["outputs"]["3"]["pid"][0]
    assert _wait(lambda: not mode.packs.started), mode.packs.started
    assert not server.PackServer.status()  # the processes are gone
    again = _run(PROMPT, ["3"])
    assert again.success, again.status_messages
    assert again.history_result["outputs"]["3"]["pid"][0] != pid


def test_a_pack_in_use_is_not_stopped(mode, corpus, monkeypatch):
    monkeypatch.setenv(packs.TTL_ENV, str(1 / 60))
    mode.setup_packs([str(corpus)])
    asyncio.run(mode.ensure_node_types({"PackDouble"}))
    with mode.packs.using("V3Pack"):
        time.sleep(3)
        assert "V3Pack" in mode.packs.started
    assert _wait(lambda: "V3Pack" not in mode.packs.started)


def test_ttl_zero_never_stops(monkeypatch):
    monkeypatch.setenv(packs.TTL_ENV, "0")
    assert packs.PackManager(None).ttl == 0
    monkeypatch.delenv(packs.TTL_ENV)
    assert packs.PackManager(None).ttl == 15 * 60


# --------------------------------------------------------------------------- #
# Launchers
# --------------------------------------------------------------------------- #
LAUNCHER = r'''#!/usr/bin/env bash
# A drop-in launcher: keeps the contract, runs the pack server however it likes.
set -euo pipefail
here="$(dirname "$0")"
case "$1" in
  start)
    echo "start $2 $3" >> "$here/launcher.log"
    cd "$COMFY_ROOT"
    nohup "$PACK_PYTHON" -m comfy_api._impl.isolation.server _serve --pack "$2" --port "$3" --token-file "$4" \
      > "$here/pack-$3.log" 2>&1 &
    echo $! > "$here/pack-$3.pid" ;;
  stop)
    echo "stop $2" >> "$here/launcher.log"
    kill "$(cat "$here/pack-$2.pid")" 2>/dev/null || true ;;
esac
'''


def test_a_launcher_can_be_dropped_in(mode, corpus, tmp_path, monkeypatch):
    script = tmp_path / "launcher.sh"
    script.write_text(LAUNCHER)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(launcher.LAUNCHER_ENV, str(script))
    monkeypatch.setenv("PACK_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", server.COMFY_ROOT)
    mode.setup_packs([str(corpus)])
    assert isinstance(mode.packs.launcher, launcher.CommandLauncher)
    executor = _run(PROMPT, ["3"])
    assert executor.success, executor.status_messages
    mode.stop()
    log = (tmp_path / "launcher.log").read_text().splitlines()
    started = sorted(line.split()[1] for line in log if line.startswith("start "))
    assert [os.path.basename(os.path.dirname(e)) if e.endswith("v2") else os.path.basename(e) for e in started] == \
        ["V1Pack-HEAD", "V3Pack-HEAD"]  # every pack the prompt uses, through the launcher
    assert sum(line.startswith("stop ") for line in log) == 2
