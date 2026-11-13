import logging
import os
import sys

import pytest

import nodes
from comfy.cli_args import args
from comfy_api.v4 import _pack_processes
from conftest import needs_pack_process

PACK = """\
import os

from comfy_api.v4 import ComfyExtension, io


class Where(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id={node_id!r}, outputs=[io.Int.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(os.getpid())


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [Where]


async def comfy_entrypoint():
    return Extension()
"""

REFUSED = (
    "--pack-process is not supported on Windows in this release; "
    "the packs it names are not loaded."
)


@pytest.mark.parametrize(
    "entry, named, expected",
    [
        ("ComfyUI-SD3.5-Latent-Size-Picker", "ComfyUI-SD3.5-Latent-Size-Picker", True),
        ("ComfyUI-SD3.5-Latent-Size-Picker", "ComfyUI-SD3", False),
        ("single_file.py", "single_file", True),
        ("single_file.py", "single_file.py", True),
        ("CasePack", "casepack", sys.platform == "win32"),
    ],
)
def test_pack_process_names_a_pack_as_comfyuis_loader_does_or_by_its_file_name(
    tmp_path, monkeypatch, entry, named, expected
):
    path = tmp_path / entry
    if entry.endswith(".py"):
        path.write_text("", encoding="utf-8")
    else:
        path.mkdir()
    monkeypatch.setattr(args, "pack_process", [named])

    assert _pack_processes.isolated(str(path)) is expected


@pytest.mark.asyncio
async def test_on_windows_the_packs_pack_process_names_are_not_loaded_and_the_rest_are(
    tmp_path, node_tables, monkeypatch, caplog
):
    monkeypatch.setattr(_pack_processes, "_met", set())
    monkeypatch.setattr(args, "pack_process", ["v4_first"])
    monkeypatch.setattr(sys, "platform", "win32")
    for name in ("v4_first", "v4_second"):
        (tmp_path / f"{name}.py").write_text(PACK.format(node_id=name), encoding="utf-8")
        monkeypatch.setitem(sys.modules, str(tmp_path / name), None)

    with caplog.at_level(logging.WARNING):
        loaded = [
            await nodes.load_custom_node(str(tmp_path / f"{name}.py"))
            for name in ("v4_first", "v4_second")
        ]
        _pack_processes.warn_at_start()

    messages = [r.getMessage() for r in caplog.records]
    assert loaded == [False, True]
    assert messages.count(REFUSED) == 1
    assert any("not supported on Windows" in m and "v4_first" in m for m in messages)
    assert not _pack_processes.processes
    assert "v4_first" not in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_CLASS_MAPPINGS["v4_second"].execute().args[0] == os.getpid()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform, first", [("linux", []), ("win32", [REFUSED])])
async def test_comfyui_warns_at_start_of_each_pack_process_name_no_loaded_pack_has(
    host, monkeypatch, caplog, tmp_path, platform, first
):
    (tmp_path / "v4_named.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(_pack_processes, "_met", set())
    monkeypatch.setattr(args, "pack_process", ["v4_named", "v4_not_there"])
    monkeypatch.setattr(sys, "platform", platform)
    _pack_processes.isolated(str(tmp_path / "v4_named.py"))

    with caplog.at_level(logging.WARNING):
        host.add_routes()

    warned = [
        r.getMessage() for r in caplog.records if "--pack-process" in r.getMessage()
    ]
    assert warned == [
        *first,
        "--pack-process v4_not_there: no custom node pack of that name was loaded.",
    ]


REPLACES = """\
from comfy_api.v4 import ComfyExtension, engine, io


class Extension(ComfyExtension):
    async def on_load(self):
        await engine.node_replacement.register(io.NodeReplace("LoadImage", "V4RemovedNode"))

    async def get_node_list(self):
        return []


async def comfy_entrypoint():
    return Extension()
"""


@needs_pack_process
@pytest.mark.asyncio
async def test_a_node_replacement_a_pack_process_registered_while_loading_goes_with_the_pack(
    host, tmp_path, monkeypatch
):
    from comfy_api.v4 import _packs

    monkeypatch.setattr(args, "pack_process", ["v4_replaces"])
    path = tmp_path / "v4_replaces.py"
    path.write_text(REPLACES, encoding="utf-8")
    try:
        assert await nodes.load_custom_node(str(path))
        registered = host.node_replace_manager.has_replacement("V4RemovedNode")
        _packs._remove_pack(str(path))
    finally:
        _pack_processes.unload(str(path))

    assert registered
    assert not host.node_replace_manager.has_replacement("V4RemovedNode")
