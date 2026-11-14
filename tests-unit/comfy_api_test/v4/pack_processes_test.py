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
    "the named packs run inside ComfyUI."
)


@needs_pack_process
@pytest.mark.parametrize(
    "entry, named, expected",
    [
        ("ComfyUI-SD3.5-Latent-Size-Picker", "ComfyUI-SD3.5-Latent-Size-Picker", True),
        ("ComfyUI-SD3.5-Latent-Size-Picker", "ComfyUI-SD3", False),
        ("single_file.py", "single_file", True),
        ("single_file.py", "single_file.py", False),
    ],
)
def test_pack_process_names_a_pack_as_comfyuis_loader_does(
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
async def test_on_windows_the_packs_pack_process_names_run_inside_comfyui_with_one_warning(
    load_pack, monkeypatch, caplog
):
    monkeypatch.setattr(_pack_processes, "_met", set())
    monkeypatch.setattr(args, "pack_process", ["v4_first", "v4_second"])
    monkeypatch.setattr(sys, "platform", "win32")

    with caplog.at_level(logging.WARNING):
        for name in ("v4_first", "v4_second"):
            await load_pack(PACK.format(node_id=name), name)

    assert [r.getMessage() for r in caplog.records].count(REFUSED) == 1
    assert not any(
        "v4_first" in p or "v4_second" in p for p in _pack_processes.processes
    )
    ran_in = [
        nodes.NODE_CLASS_MAPPINGS[n].execute().args[0]
        for n in ("v4_first", "v4_second")
    ]
    assert ran_in == [os.getpid()] * 2


@pytest.mark.asyncio
async def test_comfyui_warns_at_start_of_each_pack_process_name_no_loaded_pack_has(
    host, monkeypatch, caplog, tmp_path
):
    (tmp_path / "v4_named.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(_pack_processes, "_met", set())
    monkeypatch.setattr(args, "pack_process", ["v4_named", "v4_not_there"])
    _pack_processes.isolated(str(tmp_path / "v4_named.py"))

    with caplog.at_level(logging.WARNING):
        host.add_routes()

    warned = [
        r.getMessage() for r in caplog.records if "--pack-process" in r.getMessage()
    ]
    assert warned == [
        "--pack-process v4_not_there: no custom node pack of that name was loaded."
    ]
