import os as _os
import sys as _sys

_CONVERSION = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "comfy_api", "conversion"
)
_sys.path.insert(0, _CONVERSION)
_os.environ.setdefault(
    "COMFY_ROOT",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)
_os.environ["PYTHONPATH"] = _os.pathsep.join(
    p
    for p in (_CONVERSION, _os.environ["COMFY_ROOT"], _os.environ.get("PYTHONPATH", ""))
    if p
)
import ast
import asyncio
import contextvars
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

tree = ast.parse(
    (
        Path(__file__).resolve().parents[2] / "comfy_api" / "conversion" / "convert.py"
    ).read_text()
)
SHIM = next(
    ast.literal_eval(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "ENGINE_SHIM" for t in n.targets)
)


def build(native=None):
    defs = [
        n
        for n in ast.parse(SHIM).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in ("sync", "running", "get_filename_list", "run_v1")
    ]
    scope = {
        "asyncio": asyncio,
        "_LOOP": contextvars.ContextVar("testloop", default=None),
        "eng": NS(models=None),
        "_output": lambda x: x,
    }
    exec(  # noqa: S102
        compile(
            ast.Module(body=defs, type_ignores=[]),
            "<trusted-generated-functions>",
            "exec",
        ),
        scope,
    )
    return scope


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_runtime_constructor_once_hidden_and_cleanup(asynchronous):
    e = build()
    servicing = asyncio.get_running_loop()
    events = []
    hostlist = ["b.safetensors", "a.safetensors"]
    caller = threading.get_ident()

    async def names(folder):
        assert asyncio.get_running_loop() is servicing
        events.append(("broker", folder))
        return hostlist.copy()

    e["eng"].models = NS(list=names)

    class Node:
        def __init__(self):
            assert threading.get_ident() != caller
            events.append(("construct",))
            self.files = e["get_filename_list"](folder_name="checkpoints")

        def run(self, x, prompt=None):
            events.append(("method",))
            assert prompt == {"1": {}}
            return self.files, x

        async def arun(self, x, prompt=None):
            await asyncio.sleep(0)
            return self.run(x, prompt)

    cls = NS(hidden=NS(prompt={"1": {}}))
    out = await asyncio.wait_for(
        e["run_v1"](
            Node, "arun" if asynchronous else "run", cls, {"x": 3}, {"prompt": "prompt"}
        ),
        2,
    )
    assert out == (hostlist, 3)
    assert [x[0] for x in events].count("construct") == 1
    assert [x[0] for x in events].count("method") == 1
    assert e["_LOOP"].get() is None

    class Bad:
        def __init__(self):
            raise ValueError("constructor failed")

    with pytest.raises(ValueError, match="constructor failed"):
        await e["run_v1"](Bad, "run", cls, {}, {})
    assert e["_LOOP"].get() is None

    class BadMethod(Node):
        def run(self, **kwargs):
            raise RuntimeError("method failed")

    with pytest.raises(RuntimeError, match="method failed"):
        await e["run_v1"](BadMethod, "run", cls, {}, {})
    assert e["_LOOP"].get() is None
    assert (await e["run_v1"](Node, "run", cls, {"x": 4}, {"prompt": "prompt"}))[1] == 4


def test_original_signature_and_native_importtime(monkeypatch, tmp_path):
    import folder_paths
    from comfy_api.v4 import util

    e = build()
    assert (
        list(inspect.signature(folder_paths.get_filename_list).parameters)
        == list(inspect.signature(e["get_filename_list"]).parameters)
        == ["folder_name"]
    )
    (tmp_path / "bridge-visible.safetensors").write_bytes(b"")
    monkeypatch.setitem(
        folder_paths.folder_names_and_paths,
        "checkpoints",
        ([str(tmp_path)], {".safetensors"}),
    )
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    expected = folder_paths.get_filename_list(folder_name="checkpoints")
    assert "bridge-visible.safetensors" in expected
    assert e["get_filename_list"](folder_name="checkpoints") == expected
    calls = []

    def native(folder):
        calls.append(folder)
        return ["native-visible.safetensors"]

    monkeypatch.setattr(
        type(util.folders), "get_filename_list", lambda self, folder: native(folder)
    )
    assert e["get_filename_list"](folder_name="checkpoints") == [
        "native-visible.safetensors"
    ]
    assert calls == ["checkpoints"]

    def unavailable(self, folder):
        raise ModuleNotFoundError("folder_paths unavailable")

    monkeypatch.setattr(type(util.folders), "get_filename_list", unavailable)
    with pytest.raises(ModuleNotFoundError, match="unavailable"):
        e["get_filename_list"]("checkpoints")
    with pytest.raises(TypeError):
        e["get_filename_list"](folder="checkpoints")
